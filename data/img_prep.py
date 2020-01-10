# resize whole slide images into proper sizes
import cv2
import os
from os import listdir, mkdir
from os.path import join, exists
import warnings
import numpy as np
from shutil import copyfile
from matplotlib import pyplot as plt
from IPython.core.display import display, HTML
import warnings


def process_folder(input_folder, output_folder, rescale_ratio=0.1, mode='grey'):
    """Processes all images in a folder, and store in a new folder.

    Resizes all images.

    Args:
        input_folder: path to the folder containing image files.
        output_folder: path to the folder that stores the outputs.
        rescale_ratio: the ratio between (0,inf), downsample if < 1.
        mode: the image channels, in ("grey", "color"), default: "grey"
    """
    files = listdir(input_folder)
    img_files = [x for x in files if x.split('.')[-1] in ('tif', 'jpg', 'png')]
    if mode == 'grey':
        imread_flag = cv2.IMREAD_GRAYSCALE
    elif mode == 'color':
        imread_flag = cv2.IMREAD_COLOR
    else:
        raise NotImplementedError
    if exists(output_folder):
        warnings.warn("output folder existed, might be overwitten!")
        for img_file in img_files:
            out_file = 'resized '+img_file
            if exists(join(output_folder, out_file)):
                warnings.warn(f"output file {out_file} already exists!")
                confirm = input("Do you really want to proceed? (y/n)")
                if confirm == "y":
                    break
                else:
                    raise Exception
    else:
        mkdir(output_folder)

    for img_file in img_files:
        img = cv2.imread(join(input_folder, img_file), imread_flag)
        img_res = cv2.resize(img, None, fx=rescale_ratio, fy=rescale_ratio)
        output_path = join(output_folder, "resized "+img_file)
        cv2.imwrite(output_path, img_res)


def select_img(input_folder, output_folder, key_words):
    """Selects image files according to names.

    Args:
        input_folder: path to the folder containing image files.
        output_folder: path to the folder that stores the outputs.
        key_words: a string filters image file names.
    """
    files = files = listdir(input_folder)
    img_files = [x for x in files if x.split('.')[-1] in ('tif', 'jpg', 'png')]
    img_files = [x for x in img_files if key_words in x]
    if exists(output_folder):
        warnings.warn("output folder existed, might be overwitten!")
        for img_file in img_files:
            warnings.warn(f"output file {img_file} already exists!")
            if input("Do you really want to proceed? (y/n)") == "y":
                break
            else:
                raise Exception
    else:
        mkdir(output_folder)

    for img_file in img_files:
        copyfile(join(input_folder, img_file), join(output_folder, img_file))


def process_data(input_folder, output_folder):
    """Forms a 'H&E slides to Hypoxia' dataset in the output folder.

    The dataset contains two folders. The img folder holds the source imgs. The 
    mask folder holds the label imgs. Corresponding pairs should have the same 
    size. Usually, the input folders contains raw slide images.

    Args:
        input_folder: path to the folder containing image files.
        output_folder: path to the output dataset.
    """
    # preprocess imgs
    process_folder(input_folder, output_folder, rescale_ratio=0.1)
    # select imgs
    img_folder = join(output_folder, 'img')
    select_img(output_folder, img_folder, 'HE-green')

    mask_folder = join(output_folder, 'mask')
    select_img(output_folder, mask_folder, '_EF5')


if __name__ == '__main__':
    data_folders = ["/home/haotian/Downloads/hypoxia image datasets/DC 201-226",
                    "/home/haotian/Downloads/hypoxia image datasets/DC 10B-12E",
                    "/home/haotian/Downloads/hypoxia image datasets/DC 250-262"
                    ]
    for data_folder in data_folders:
        print(f"processing {data_folder}")
        process_data(input_folder=data_folder,
                     output_folder='/home/haotian/Code/vessel_segmentation/data/hypoxia img')
