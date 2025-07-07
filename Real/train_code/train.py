from architecture import *
from utils import *
from dataset import dataset
import torch.utils.data as tud
import torch
import torch.nn.functional as F
import time
import datetime
from torch.autograd import Variable
import os
from option import opt
from torch.utils.tensorboard import SummaryWriter
import scipy.io as scio


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
if not torch.cuda.is_available():
    raise Exception('NO GPU!')


Seed_Torch(opt.seed)

# load training data
CAVE = prepare_data_cave(opt.data_path_CAVE, 2)
KAIST = prepare_data_KAIST(opt.data_path_KAIST, 30)

# saving path
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

model_path = opt.outf + '/model/'
if not os.path.exists(model_path):
    os.makedirs(model_path)


# model
model = D4EU(opt).cuda()

## Load trained model
start_epoch = findLastCheckpoint(save_dir=model_path)
if start_epoch > 0:
    print('Load model: resuming by loading epoch %03d' % start_epoch)
    checkpoint = torch.load(os.path.join(model_path, 'model_%03d.pkl' % start_epoch))
    model.load_state_dict(checkpoint['model'])
    start_epoch = 1 + checkpoint['epoch']
else:
    start_epoch = 1

# optimizing
optimizer = torch.optim.Adam([{'params': model.parameters(), 'initial_lr': opt.learning_rate}], lr=opt.learning_rate, betas=(0.9, 0.999), eps=1e-8)
if opt.scheduler == 'MultiStepLR':
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.milestones, gamma=opt.gamma)
elif opt.scheduler == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, opt.max_epoch, eta_min=1e-6, last_epoch=start_epoch - 2)

mse = torch.nn.MSELoss().cuda()

def loss_f(loss_func, pred, lbl):
    return torch.sqrt(loss_func(pred, lbl))

if __name__ == "__main__":

    logger = gen_log(opt.outf)
    logger.info("Learning rate:{}, batch_size:{}, seed:{}.\n".format(opt.learning_rate, opt.batch_size, opt.seed))
    # writer = SummaryWriter(log_dir=tensorboard_path)

    ## pipline of training
    for epoch in range(1, 1 + opt.max_epoch):

        #--20240911--训练一个epoch
        model.train()
        epoch_loss = 0
        iter_num = int(np.floor(opt.epoch_sam_num / opt.batch_size))

        Dataset = dataset(opt, CAVE, KAIST)
        loader_train = tud.DataLoader(Dataset, num_workers=0, batch_size=opt.batch_size, shuffle=True)

        start_time = time.time()

        psnr_total = 0
        for i, (input, label, Mask, Phi, Phi_s) in enumerate(loader_train):
            input, label, Mask, Phi, Phi_s = Variable(input), Variable(label), Variable(Mask), Variable(Phi), Variable(Phi_s)
            input, label, Mask, Phi, Phi_s = input.cuda(), label.cuda(), Mask.cuda(), Phi.cuda(), Phi_s.cuda()

            g = input.unsqueeze(1)
            Phi_batch = Mask.permute(0, 3, 1, 2)
            Phi_s_batch = Phi_s.unsqueeze(1)
            out = model(g=g, input_mask=(Phi_batch, Phi_s_batch))

            psnr = compare_psnr(label.detach().cpu().numpy(), out[opt.stage - 1].detach().cpu().numpy(), data_range=1.0)
            psnr_total = psnr_total + psnr

            loss = loss_f(mse, out[opt.stage - 1], label) + 0.7 * loss_f(mse, out[opt.stage - 2], label) + \
            0.5 * loss_f(mse, out[opt.stage-3], label) + 0.3 * loss_f(mse, out[opt.stage-4], label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 50 == 0:
                print('%4d %4d / %4d loss = %.16f time = %s' % (
                    start_epoch + epoch, i, len(Dataset) // opt.batch_size,
                    epoch_loss / ((i + 1) * opt.batch_size),
                    datetime.datetime.now()))

        elapsed_time = time.time() - start_time
        scheduler.step()

        # tensorboard_show(writer, loss_avg, epoch)


        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch + 1}, Learning Rate: {current_lr:.6f}")

        print('epoch = %4d , loss = %.16f , Avg PSNR = %.4f ,time = %4.2f s' % (
            start_epoch + epoch, epoch_loss / len(Dataset), psnr_total / (i + 1), elapsed_time))
        state = {'model': model.state_dict(), 'epoch': start_epoch + epoch,
                 'loss': epoch_loss / len(Dataset), 'psnr': psnr_total / (i + 1)}
        torch.save(state, os.path.join(model_path, 'model_%03d.pkl' % (start_epoch + epoch)))


