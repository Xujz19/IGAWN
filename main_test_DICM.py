import argparse
import os
import time
import torch
import torch.nn as nn
import numpy as np
import imageio
import util
import pandas as pd

# Params
parser = argparse.ArgumentParser()

parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--save_dir', default='./experiment', type=str)
parser.add_argument('--test_data', default='../../LowLight_Testsets', type=str)
parser.add_argument('--dataset_name', default='LOL_V2_Synthetic', type=str)
parser.add_argument('--n_colors', default=3, type=int)
parser.add_argument('--save_results', default=True, type=bool)

parser.add_argument('--w1', default=10, type=float)
parser.add_argument('--w2', default=0.1, type=float)

def makedir(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def pad_tensor(input):
    height_org, width_org = input.shape[2], input.shape[3]
    divide = 8

    if width_org % divide != 0 or height_org % divide != 0:

        width_res = width_org % divide
        height_res = height_org % divide
        if width_res != 0:
            width_div = divide - width_res
            pad_left = int(width_div / 2)
            pad_right = int(width_div - pad_left)
        else:
            pad_left = 0
            pad_right = 0

        if height_res != 0:
            height_div = divide - height_res
            pad_top = int(height_div / 2)
            pad_bottom = int(height_div - pad_top)
        else:
            pad_top = 0
            pad_bottom = 0

        padding = nn.ReflectionPad2d((pad_left, pad_right, pad_top, pad_bottom))
        input = padding(input)
    else:
        pad_left = 0
        pad_right = 0
        pad_top = 0
        pad_bottom = 0

    height, width = input.data.shape[2], input.data.shape[3]
    assert width % divide == 0, 'width cant divided by stride'
    assert height % divide == 0, 'height cant divided by stride'

    return input, pad_left, pad_right, pad_top, pad_bottom


def pad_tensor_back(input, pad_left, pad_right, pad_top, pad_bottom):
    height, width = input.shape[2], input.shape[3]
    return input[:, :, pad_top: height - pad_bottom, pad_left: width - pad_right]


if __name__ == '__main__':

    IMG_EXTENSIONS = [
        '.jpg', '.JPG', '.jpeg', '.JPEG',
        '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    ]

    args = parser.parse_args()

    args.save_dir = ('%s_%d_%.2f') % (args.save_dir, args.w1 * 10, args.w2)

    model_path = os.path.join(args.save_dir, 'models')
    low_path = os.path.join(args.test_data, args.dataset_name)

    cuda = torch.cuda.is_available()

    # makedir(model_path)

    model_list = os.listdir(model_path)
    model_list.sort()
    # model_list = model_list[62:]

    ############### prepare train data ###############
    data_time = time.time()
    image_names = []
    for root, _, fnames in sorted(os.walk(low_path)):
        for fname in fnames:
            if any(fname.endswith(extension) for extension in IMG_EXTENSIONS):
                image_names.append(fname)
    image_names.sort()

    print('Finding {} test data file path'.format(len(image_names)))
    print('Reading images to memory..........')

    ###############################################################

    for model_name in model_list:
        begin_time = time.time()
        print(model_name)
        model = torch.load(os.path.join(model_path, model_name))
        model.eval()
        if cuda:
            model = model.cuda()

            with torch.no_grad():
                for img_name in image_names:
                    print(img_name)
                    low_name = img_name
                    low_image = imageio.imread(os.path.join(low_path, low_name))
                    low_image = util.uint2single(low_image)
                    low_input = util.single2tensor4(low_image)
                    low_input, pad_left, pad_right, pad_top, pad_bottom = pad_tensor(low_input)
                    if cuda:
                        low_input = low_input.cuda()
                    # print(low_input.shape)
                    pred_img, pred_illumination = model(low_input)

                    pred_illumination = pad_tensor_back(pred_illumination, pad_left, pad_right, pad_top, pad_bottom)
                    pred_img = pad_tensor_back(pred_img, pad_left, pad_right, pad_top, pad_bottom)

                    pred_illumination = util.tensor2single(pred_illumination)
                    pred_img = util.tensor2single(pred_img)

                    torch.cuda.synchronize()
                    pred_illumination = util.single2uint(pred_illumination)
                    pred_img = util.single2uint(pred_img)

                    if args.save_results:
                        result_path = os.path.join(args.save_dir, 'results', args.dataset_name, model_name)
                        makedir(result_path)
                        imageio.imsave(os.path.join(result_path, img_name[:-4] + '_IGAWN.png'), pred_img)
                        imageio.imsave(os.path.join(result_path, img_name[:-4] + '_illumination.png'),
                                       pred_illumination)
