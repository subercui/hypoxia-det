# resize whole slide images into proper sizes
import cv2
import os
import argparse
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
    files = listdir(input_folder)
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


def process_data(output_folder):
    """Forms a 'H&E slides to Hypoxia' dataset in the output folder.

    The dataset contains two folders. The img folder holds the source imgs. The
    mask folder holds the label imgs. Corresponding pairs should have the same
    size. Usually, the input folders contains raw slide images.

    Args:
        input_folder: path to the folder containing image files.
        output_folder: path to the output dataset.
    """
    # select imgs
    img_folder = join(output_folder, 'img')
    select_img(output_folder, img_folder, 'HE-green')

    mask_folder = join(output_folder, 'mask')
    select_img(output_folder, mask_folder, '_EF5')


class DataParser(object):
    def __init__(self, source_folder, rescale_ratio=0.1):
        self.imgs_folder = source_folder
        self.rescale_ratio = rescale_ratio
        return

    def parse_all(self, target_folder):
        trial_folder_list = listdir(self.imgs_folder)
        trial_folder_list.remove('.DC 274-297')
        trial_folder_list = [join(self.imgs_folder, folder)
                             for folder in trial_folder_list]
        for trial_folder in trial_folder_list:
            sample_name_list = self.samples_in_trial(trial_folder)
            for s_name in sample_name_list:
                self.form_sample_folder(trial_folder, target_folder, s_name)
                self.save_color_img(target_folder, s_name)

    def samples_in_trial(self, trial_folder):
        """
        Gets sample names in a trial folder and return in a list.
        """
        files = listdir(join(trial_folder, 'Background'))
        img_files = [x for x in files if x.split(
            '.')[-1] in ('tif', 'jpg', 'png')]
        sample_name_list = [x.split('.')[1] for x in img_files]
        return sample_name_list  # like ['12B',]

    def process_img(self, img_dir, rescale_ratio, mode='grey'):
        if mode == 'grey':
            imread_flag = cv2.IMREAD_GRAYSCALE
        elif mode == 'color':
            imread_flag = cv2.IMREAD_COLOR
        else:
            raise NotImplementedError
        img = cv2.imread(img_dir, imread_flag)
        img_res = cv2.resize(img, None, fx=rescale_ratio, fy=rescale_ratio)
        return img_res

    def save_color_img(self, target_folder, sample_name):
        sample_dir = join(target_folder, sample_name)
        img_B = cv2.imread(join(sample_dir, 'HE-blue.png'),
                           cv2.IMREAD_GRAYSCALE)
        img_G = cv2.imread(join(sample_dir, 'HE-green.png'),
                           cv2.IMREAD_GRAYSCALE)
        img_R = cv2.imread(join(sample_dir, 'HE-red.png'),
                           cv2.IMREAD_GRAYSCALE)
        img = np.stack([255 - img_B, 255 - img_G, 255 - img_R], axis=2)
        cv2.imwrite(join(sample_dir, 'HE.png'), img)

    def form_sample_folder(self, input_folder, target_folder, sample_name):
        """
        makes a sample folder containing the input images and ground truth label img.

        For example, in the hypoxia img directory, create a folder DC_9D for the DC_9D
        slide sample. The folder DC_9D shall contains four rescaled imgs, 1. the H&E 2.
        the necrosis mask 3. the perfusion mask 4. the ground truth EF5 img as label.

        Args:
            input_folder: the folder contains the sample images
            target_folder: the path to the folder where the sample folders locate
            sample_name: used to search and assign file names.
        """
        print(f'processing {sample_name} folder.')
        # first make a subfolder to contain the images - e.g. 'target_folder/sample_name'
        sample_dir = join(target_folder, sample_name)
        if not os.path.exists(sample_dir):
            mkdir(sample_dir)
        # resize and move the mask images - e.g. 'target_folder/sample_name/imgs_necrosis.png'
        img_file_nec = join(input_folder, 'Necrosis',
                            'Tissue Slides.'+sample_name+'.png')
        img_res = self.process_img(img_file_nec, self.rescale_ratio)
        img_nec = img_res.copy()
        cv2.imwrite(join(sample_dir, 'necrosis.png'), img_res)

        img_file_perf = join(input_folder, 'Perfusion',
                             'Tissue Slides.'+sample_name+'.png')
        img_res = self.process_img(img_file_perf, self.rescale_ratio)
        cv2.imwrite(join(sample_dir, 'perfusion.png'), img_res)

        # resize and move the maker HE and EF5 images
        files = listdir(input_folder)
        img_files = [x for x in files if x.split(
            '.')[-1] in ('tif', 'jpg', 'png')]
        for img_file in img_files:
            if (sample_name+'_' in img_file) or (sample_name+'-' in img_file):
                if ('HE-G' in img_file) or ('HE-green' in img_file) or ('HEgreen' in img_file):
                    img_res = self.process_img(
                        join(input_folder, img_file), self.rescale_ratio)
                    if not os.path.exists(join(sample_dir, 'HE-green.png')):
                        cv2.imwrite(join(sample_dir, 'HE-green.png'), img_res)
                    else:
                        warnings.warn(
                            f"file already exists, while processing {img_file}")
                elif ('HE-R' in img_file) or ('HE-red' in img_file) or ('HEred' in img_file):
                    img_res = self.process_img(
                        join(input_folder, img_file), self.rescale_ratio)
                    if not os.path.exists(join(sample_dir, 'HE-red.png')):
                        cv2.imwrite(join(sample_dir, 'HE-red.png'), img_res)
                    else:
                        warnings.warn(
                            f"file already exists, while processing {img_file}")
                elif ('HE-B' in img_file) or ('HE-blue' in img_file) or ('HE-blue' in img_file):
                    img_res = self.process_img(
                        join(input_folder, img_file), self.rescale_ratio)
                    if not os.path.exists(join(sample_dir, 'HE-blue.png')):
                        cv2.imwrite(join(sample_dir, 'HE-blue.png'), img_res)
                    else:
                        warnings.warn(
                            f"file already exists, while processing {img_file}")
                elif 'EF5' in img_file:
                    img_res = self.process_img(
                        join(input_folder, img_file), self.rescale_ratio)
                    img_ef5 = img_res.copy()
                    if not os.path.exists(join(sample_dir, 'EF5.png')):
                        cv2.imwrite(join(sample_dir, 'EF5.png'), img_res)
                    else:
                        warnings.warn(
                            f"file already exists, while processing {img_file}")

        masked_ef5 = (img_ef5 * (img_nec <= 0)).astype(img_ef5.dtype)
        cv2.imwrite(join(sample_dir, 'EF5_masked.png'), masked_ef5)
        assert len(listdir(sample_dir)) == 7
        return


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--source-folder', type=str,
                            help='the path to slide images')
    argparser.add_argument('--target-folder', type=str,
                            help='the folder to store parsed images')
    args = argparser.parse_args()
    parser = DataParser(
        source_folder=args.source_folder, rescale_ratio=0.1)
    parser.parse_all(target_folder=args.target_folder)
