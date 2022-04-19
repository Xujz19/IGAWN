import argparse
import re
import os, glob, datetime, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from data import TrainData
from model import IGAWN
from losses import SSIM, MS_SSIM, TVLoss

# Params
parser = argparse.ArgumentParser()

parser.add_argument('--epoch', default=200, type=int, help='number of train epoches')
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--n_feats', default=32, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--save_dir', default='./experiment', type=str)
parser.add_argument('--w1', default=10, type=float)
parser.add_argument('--w2', default=0.1, type=float)

# Data
parser.add_argument('--batch_size', default=32, type=int, help='batch size')
parser.add_argument('--train_data', default='../../LOL_V2', type=str, help='path of train data')
parser.add_argument('--dataset', default='LowLight', type=str, help='dataset name')
parser.add_argument('--n_colors', default=3, type=int)
parser.add_argument('--patch_size', default=320, type=int)

# Optim
parser.add_argument('--lr', default=1e-4, type=float, help='initial learning rate for Adam')
parser.add_argument('--weight_decay', default=0, type=float)
parser.add_argument('--clip_grad_norm', default=2.5, type=float)


args = parser.parse_args()

cuda = torch.cuda.is_available()

args.save_dir = ('%s_%d_%.2f')%(args.save_dir, args.w1*10, args.w2)
model_save_dir = os.path.join(args.save_dir, 'models')
optim_save_dir = os.path.join(args.save_dir, 'optim')
results_save_dir = os.path.join(args.save_dir, 'results')


def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, 'model_*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*model_(.*).pth.*", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch


def count_params(model):
    count = 0
    for param in model.parameters():
        param_size = param.size()
        count_of_one_param = 1
        for i in param_size:
            count_of_one_param *= i
        count += count_of_one_param
    print('Total parameters: %d' % count)


def log(*args, **kwargs):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S:"), *args, **kwargs)


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    makedir(model_save_dir)
    makedir(optim_save_dir)
    makedir(results_save_dir)

    print('===> Building model')
    model = IGAWN(n_colors=args.n_colors, n_feats=args.n_feats)
    # print(model)
    count_params(model)

    ###############################################################

    initial_epoch = findLastCheckpoint(save_dir=model_save_dir)  # load the last model in matconvnet style
    if initial_epoch > 0:
        print('resuming by loading epoch %03d' % initial_epoch)
        # model.load_state_dict(torch.load(os.path.join(save_dir, 'model_%03d.pth' % initial_epoch)))
        model = torch.load(os.path.join(model_save_dir, 'model_%03d.pth' % initial_epoch))

    model.train()

    criterion = nn.MSELoss()
    criterion_ssim_E = MS_SSIM(data_range=1, channel=3)
    criterion_ssim_I = MS_SSIM(data_range=1, channel=1)
    # criterion_ssim = SSIM(data_range=1, channel=3)
    criterion_grad = TVLoss()
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
        criterion_ssim_E = criterion_ssim_E.cuda()
        criterion_ssim_I = criterion_ssim_I.cuda()
        criterion_grad = criterion_grad.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    TrainLoader = TrainData(args).get_loader()
    
    lrd = args.lr / (args.epoch//2)
    # for epoch in range(initial_epoch, args.epoch):
    for epoch in range(0, args.epoch):
        if epoch>args.epoch//2:
           lr = args.lr - lrd * (epoch-args.epoch//2)
           for param_group in optimizer.param_groups:
               param_group['lr'] = lr
        time_begin = time.time()

        epoch_intensity_l2_loss = 0
        epoch_intensity_ssim_loss = 0
        epoch_intensity_grad_loss = 0
        epoch_illumination_l2_loss = 0
        epoch_illumination_ssim_loss = 0
        epoch_illumination_grad_loss = 0
        epoch_total_loss = 0
        start_time = time.time()

        for n_count, (low, high, low_illumination) in enumerate(TrainLoader):
            optimizer.zero_grad()
            if cuda:
                low, high, low_illumination = low.cuda(), high.cuda(), low_illumination.cuda()
            pred_img, pred_illumination = model(low)

            intensity_l2_loss = criterion(pred_img, high)
            intensity_ssim_loss = -criterion_ssim_E(pred_img, high)
            intensity_grad_loss = criterion_grad(pred_img, high)
            intensity_loss = intensity_ssim_loss + args.w1 * intensity_grad_loss
            
            illumination_l2_loss = criterion(pred_illumination, low_illumination)
            illumination_ssim_loss = -criterion_ssim_I(pred_illumination, low_illumination)
            illumination_grad_loss = criterion_grad(pred_illumination, low_illumination)
            illumination_loss = illumination_ssim_loss + 10 *illumination_grad_loss

            total_loss = 10 * intensity_loss + args.w2 * illumination_ssim_loss

            epoch_intensity_l2_loss += intensity_l2_loss.item()
            epoch_intensity_ssim_loss += intensity_ssim_loss.item()
            epoch_intensity_grad_loss += intensity_grad_loss.item()
            epoch_illumination_l2_loss += illumination_l2_loss.item()
            epoch_illumination_ssim_loss += illumination_ssim_loss.item()
            epoch_illumination_grad_loss += illumination_grad_loss.item()
            epoch_total_loss += total_loss.item()
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
            optimizer.step()

        elapsed_time = time.time() - start_time
        log('epcoh = %4d , total_loss = %.6f , time = %4.2f s, w1=%d, w2=%.2f' % (epoch + 1, epoch_total_loss, elapsed_time, args.w1*10, args.w2))
        log('Intensity : L2_loss = %.6f , ssim_loss = %.6f , grad_loss = %.6f ' % (epoch_intensity_l2_loss, epoch_intensity_ssim_loss, epoch_intensity_grad_loss))
        log('Illumination : L2_loss = %.6f , ssim_loss = %.6f , grad_loss = %.6f ' % (
        epoch_illumination_l2_loss, epoch_illumination_ssim_loss, epoch_illumination_grad_loss))

        if (epoch + 1)  > 100:
            torch.save(model, os.path.join(model_save_dir, 'model_%03d.pth' % (epoch + 1)))
