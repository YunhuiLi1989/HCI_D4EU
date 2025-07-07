import torch
import os
import argparse
from architecture import *
from utils import dataparallel
import scipy.io as sio
import numpy as np
from torch.autograd import Variable

from torch.utils.tensorboard import SummaryWriter
from utils import *
from architecture import *
import time

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="PyTorch HSIFUSION")
parser.add_argument('--data_path', default='../dataset/TSA_real_data/Measurements/', type=str,help='path of data')
parser.add_argument('--mask_path', default='../dataset/TSA_real_data/mask.mat', type=str,help='path of mask')
parser.add_argument("--size", default=660, type=int, help='the size of trainset image')
parser.add_argument("--stage", default=3, type=str, help='Model scale')
parser.add_argument("--trainset_num", default=2000, type=int, help='total number of trainset')
parser.add_argument("--testset_num", default=5, type=int, help='total number of testset')
parser.add_argument("--seed", default=1, type=int, help='Random_seed')
parser.add_argument("--batch_size", default=1, type=int, help='batch_size')
parser.add_argument("--isTrain", default=False, type=bool, help='train or test')
parser.add_argument("--pretrained_model_path", default=None, type=str)

# Saving specifications
parser.add_argument('--outf', type=str, default='./output/', help='saving_path')
parser.add_argument("--learning_rate", type=float, default=0.0004)
parser.add_argument("--scheduler", type=str, default='MultiStepLR', help='MultiStepLR or CosineAnnealingLR')
parser.add_argument("--milestones", type=int, default=[50, 100, 150, 200, 250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
opt = parser.parse_args()
print(opt)

recon_path = opt.outf + '/recon/'
if not os.path.exists(recon_path):
    os.makedirs(recon_path)
model_path = '../train_code/output/model'

def prepare_data(path, file_num):
    HR_HSI = np.zeros((((220,714,file_num))))
    for idx in range(file_num):
        ####  read HrHSI
        path1 = os.path.join(path) + 'scene' + str(idx+1) + '.mat'
        data = sio.loadmat(path1)
        temp = data['meas_real']
        HR_HSI[:, :, idx] = temp[220-0:440-0, 0:714]
        HR_HSI[HR_HSI < 0] = 0.0
        HR_HSI[HR_HSI > 1] = 1.0
    return HR_HSI

def load_mask(path,size=660):
    ## load mask
    data = sio.loadmat(path)
    mask = data['mask']
    mask_3d = np.tile(mask[:, :, np.newaxis], (1, 1, 28))
    mask_3d_shift = np.zeros((size, size + (28 - 1) * 2, 28))
    mask_3d_shift[:, 0:size, :] = mask_3d
    for t in range(28):
        mask_3d_shift[:, :, t] = np.roll(mask_3d_shift[:, :, t], 2 * t, axis=1)
    mask_3d_shift_s = np.sum(mask_3d_shift ** 2, axis=2, keepdims=False)
    mask_3d_shift_s[mask_3d_shift_s == 0] = 1

    mask_3d_shift = mask_3d_shift[220-0:440-0, 0:714, :]
    mask_3d_shift_s = mask_3d_shift_s[220-0:440-0, 0:714]

    mask_3d_shift = torch.FloatTensor(mask_3d_shift.copy()).permute(2, 0, 1)
    mask_3d_shift_s = torch.FloatTensor(mask_3d_shift_s.copy())

    mask_3d = mask_3d[220-0:440-0, 0:660, :]
    mask_3d = torch.FloatTensor(mask_3d.copy())

    return mask_3d, mask_3d_shift.unsqueeze(0), mask_3d_shift_s.unsqueeze(0)


HR_HSI = prepare_data(opt.data_path, 5)
mask_3d, mask_3d_shift, mask_3d_shift_s = load_mask(opt.mask_path)

loss_func = nn.MSELoss()
model = D4EU(opt).cuda()

with torch.no_grad():
    for epoch in range(242, 243, 1):
        checkpoint = torch.load(os.path.join(model_path, 'model_%03d.pkl' % epoch))
        model.load_state_dict(checkpoint['model'])
        model.eval()
        epoch_loss = 0
        psnr_total = 0
        psnr_total_scene = 0

        for j in range(5):
            with torch.no_grad():
                meas = HR_HSI[:, : ,j]
                meas = meas / meas.max() * 0.8
                meas = torch.FloatTensor(meas)
                # meas = torch.FloatTensor(meas).unsqueeze(2).permute(2, 0, 1)
                input = meas.unsqueeze(0)
                input = Variable(input)
                input = input.cuda()
                mask_3d = mask_3d.cuda()
                mask_3d_shift = mask_3d_shift.cuda()
                mask_3d_shift_s = mask_3d_shift_s.cuda()

                # --20240911--生成输入网络数据
                g = input.unsqueeze(0)
                Phi_batch = mask_3d.permute(2, 0, 1).unsqueeze(0)
                Phi_s_batch = mask_3d_shift_s.unsqueeze(0)

                out = model(g=g, input_mask=(Phi_batch, Phi_s_batch))
                result = out[opt.stage - 1].clamp(min=0., max=1.)


            res = result.cpu().permute(2,3,1,0).squeeze(3).numpy()
            save_file = recon_path + f'{j+1}.mat'
            sio.savemat(save_file, {'res':res})









