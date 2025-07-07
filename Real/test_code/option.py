import argparse
import template

parser = argparse.ArgumentParser(description="HyperSpectral Image Reconstruction Toolbox")


# Hardware specifications
parser.add_argument("--gpu_id", type=str, default='0')

# Data specifications
parser.add_argument('--data_root', type=str, default='../dataset/', help='dataset directory')
parser.add_argument('--data_path_CAVE', default='../dataset/CAVE_512_28/', type=str, help='path of data')
parser.add_argument('--data_path_KAIST', default='../dataset/KAIST_CVPR2021/', type=str, help='path of data')
parser.add_argument('--mask_path', default='../dataset/TSA_real_data/mask.mat', type=str, help='path of mask')

# Saving specifications
parser.add_argument('--outf', type=str, default='./output/', help='saving_path')

# Model specifications
parser.add_argument('--pretrained_model_path', type=str, default=None, help='pretrained model directory')


# Training specifications
parser.add_argument("--size", default=660, type=int, help='cropped patch size')
parser.add_argument("--epoch_sam_num", default=5000, type=int, help='total number of trainset')
parser.add_argument('--batch_size', type=int, default=1, help='the number of HSIs per batch')
parser.add_argument("--max_epoch", type=int, default=300, help='total epoch')
parser.add_argument("--milestones", type=int, default=[50,100,150,200,250], help='milestones for MultiStepLR')
parser.add_argument("--gamma", type=float, default=0.5, help='learning rate decay for MultiStepLR')
parser.add_argument("--learning_rate", type=float, default=0.0004)

opt = parser.parse_args()
template.set_template(opt)

opt.trainset_num = 20000 // ((opt.size // 96) ** 2)

for arg in vars(opt):
    if vars(opt)[arg] == 'True':
        vars(opt)[arg] = True
    elif vars(opt)[arg] == 'False':
        vars(opt)[arg] = False