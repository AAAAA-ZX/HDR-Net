import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
try:
    sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
    from utils.util import ProgressBar
except ImportError:
    pass

from argparse import ArgumentParser

parser=ArgumentParser()
parser.add_argument('-i', type=str, dest='input_folder_', default=None, help='input_folder')
parser.add_argument('-o', type=str, dest='save_folder_', default=None, help='save_folder')

options=parser.parse_args()

def main():
    """A multi-thread tool to crop sub imags."""

    input_folder = '/hy-tmp/Data/DIV2K_train_LR_bicubic_X4'
    # input_folder = 'E:/exp/dataset/Flickr2K/Flickr2K_LR_bicubic/X4'
    # input_folder = options.input_folder_

    # save_folder = 'E:/exp/dataset/Flickr2K/Flickr2K_HR_sub'
    # save_folder = 'E:/exp/dataset/Flickr2K/Flickr2K_LR_bicubic/X4_sub'
    # save_folder = 'F:/tempE/dataset/Flickr2K_DIV2K_train_LR_bicubic/X4_sub_120'
    save_folder = '/hy-tmp/Data/DIV2K_train_LR_bicubic_X4_sub_480'
    # save_folder = options.save_folder_

    n_thread = 20

    crop_sz = 480
    step = 240

    # crop_sz = 120
    # step = 60

    thres_sz = 0

    # n_thread = 20
    # # crop_sz = 320
    # # step = 160
    # crop_sz = 80
    # step = 40
    # thres_sz = 48
    compression_level = 3  # 3 is the default value in cv2
    # CV_IMWRITE_PNG_COMPRESSION from 0 to 9. A higher value means a smaller size and longer
    # compression time. If read raw images during training, use 0 for faster IO speed.

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)

    img_list = []
    for root, _, file_list in sorted(os.walk(input_folder)):
        path = [os.path.join(root, x) for x in file_list]  # assume only images in the input_folder
        img_list.extend(path)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(n_thread)
    for path in img_list:
        pool.apply_async(worker,
                         args=(path, save_folder, crop_sz, step, thres_sz, compression_level),
                         callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def worker(path, save_folder, crop_sz, step, thres_sz, compression_level):
    img_name = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    n_channels = len(img.shape)
    if n_channels == 2:
        h, w = img.shape
    elif n_channels == 3:
        h, w, c = img.shape
    else:
        raise ValueError('Wrong image shape - {}'.format(n_channels))

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            if n_channels == 2:
                crop_img = img[x:x + crop_sz, y:y + crop_sz]
            else:
                crop_img = img[x:x + crop_sz, y:y + crop_sz, :]
            crop_img = np.ascontiguousarray(crop_img)
            # var = np.var(crop_img / 255)
            # if var > 0.008:
            #     print(img_name, index_str, var)
            cv2.imwrite(
                os.path.join(save_folder, img_name.replace('.png', '_s{:03d}.png'.format(index))),
                crop_img, [cv2.IMWRITE_PNG_COMPRESSION, compression_level])
    return 'Processing {:s} ...'.format(img_name)


if __name__ == '__main__':
    main()
