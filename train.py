# System and other libs
import os
import time
from matplotlib import pyplot as plt
import cv2
import random
import argparse
from distutils.version import LooseVersion
import math
import warnings
# Numerical libs
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import Adam
# Our libs
from models.models import ModelBuilder, SegmentationModule
from data.dataset import DRIVEData
from utils import AverageMeter, parse_devices, accuracy, intersectionAndUnion
from lib.nn import UserScatteredDataParallel, async_copy_to,  user_scattered_collate, patch_replication_callback
import lib.utils.data as torchdata
from lib.utils import as_numpy
import numpy as np
import pandas as pd
import csv

from tensorboardX import SummaryWriter
writer = SummaryWriter()
'''
I took care of eval() and train(), don't worry about implementing
anything in these unless there is an error, please fix it!
Errors that may appear may be dimension mismatching or
moving tensors onto the gpu.
'''


def infer(img_folder, segmentation_module, args, mode='patch'):
    """Given an input slide img, predicts the hypoxia outputs.

    This works in test setting.

    Arguments:
        img {np.ndarray} -- the input image used to predict hypoxia
        segmentation_module {nn.Module} -- the model to use
        args -- the arguments
        mode {str} -- in {'patch', 'whole'}, whether using patched
        img, or directly predict on wholeimg.

    Returns:
        {np.ndarray} -- the output pixel-wise prediction, which has
        the same size as the input img.
    """
    def _img2patches(image, patch_size, if_norm=True):
        assert len(image.shape) == 3
        width, height, channels = image.shape
        if if_norm:
            image = image.astype(np.float) / 255.
        patches = []
        coordinates = []
        for i in range(0, width-patch_size, patch_size):
            for j in range(0, height-patch_size, patch_size):
                patches.append(image[i:i+patch_size, j:j+patch_size])
                coordinates.append((i, j))
        patches = np.stack(patches, axis=0)  # (N, patch_size, patch_size, cs)

        # modify data type and dims
        # (N, C_in, patch_size, patch_size)
        patches = torch.tensor(patches).permute(0, 3, 1, 2)
        if torch.cuda.is_available():
            patches = patches.cuda()
        return patches, coordinates

    def _patches2img(patches, coordinates, patch_size, scale_factor=None):
        # modify data type
        patches = patches.detach().cpu().numpy().copy()

        # modify shapes
        assert patches.ndim == 4 and patches.shape[1] == 1
        patches = patches[:, 0, :, :]

        width, height = coordinates[-1][0] + \
            patch_size, coordinates[-1][1] + patch_size
        img = np.zeros((width, height))
        for i in range(len(coordinates)):
            w, h = coordinates[i]
            img[w:w+patch_size, h:h+patch_size] = patches[i]

        # if the output label has been scaled, scale it back
        if scale_factor:
            img = (img * scale_factor).astype('int')
        # assert img.max() <= 255
        if img.max() > 255:
            warnings.warn(
                f"the model output exceeds the maximum grayscale value 255 - img.max() = {img.max()}")

        img = img.astype(np.uint8)
        return img
    '''
    def top_bottom_pad(w, desired_w):
        pad_size_one = math.ceil((w-desired_w)/2.)
        pad_size_two = math.floor((w-desired_w)/2.)
        return pad_size_one, pad_size_two
    '''
    def _scale_whole_img(image, if_norm=True):
        assert len(image.shape) == 3
        width, height, _ = image.shape
        scaled_w = 2**(math.ceil(math.log(max(width, height), 2)))
        img = np.zeros((1, 3, scaled_w, scaled_w))
        w_start = math.floor((scaled_w-width)/2.)
        h_start = math.floor((scaled_w-height)/2.)
        #image = np.pad(image, (scaled_w-width, scaled_w-height), 'constant')
        #image = np.expand_dims(image, axis=0)
        img[0, :, w_start:w_start+width, h_start:h_start +
            height] = image.transpose(2, 0, 1)
        if if_norm:
            img = img.astype(np.float) / 255.
        img = torch.tensor(img)
        if torch.cuda.is_available():
            img = img.cuda()
        return img

    im_img = cv2.imread(os.path.join(
        img_folder, 'HE-green.png'), cv2.IMREAD_GRAYSCALE)
    im_nec = cv2.imread(os.path.join(
        img_folder, 'necrosis.png'), cv2.IMREAD_GRAYSCALE)
    im_perf = cv2.imread(os.path.join(
        img_folder, 'perfusion.png'), cv2.IMREAD_GRAYSCALE)
    img = np.stack([im_img, im_nec, im_perf], axis=2)
    if mode == 'patch':
        patches, coordinates = _img2patches(img, patch_size=args.patch_size)
        patch_preds = segmentation_module.infer(patches)
        pred = _patches2img(patch_preds, coordinates,
                            patch_size=args.patch_size,
                            scale_factor=args.scale_factor)
    elif mode == 'whole':
        scaled_img = _scale_whole_img(img)
        pred_ = segmentation_module.infer(scaled_img)
        pred = pred_.squeeze().detach().cpu().numpy().copy()
        if args.scale_factor:
            pred = (pred * args.scale_factor).astype('int')
        if pred.max() > 255:
            warnings.warn(
                f"the model output exceeds the maximum grayscale value 255 - img.max() = {img.max()}")
        pred = pred.astype(np.uint8)
    else:
        raise NotImplementedError

    return pred


def eval(loader_val, segmentation_module, args, crit, n_epoch):
    # intersection_meter = AverageMeter()
    # union_meter = AverageMeter()
    loss_meter = AverageMeter()
    ave_mse = AverageMeter()
    ave_pred_pct = AverageMeter()
    ave_label_pct = AverageMeter()

    segmentation_module.eval()
    # import pudb; pudb.set_trace()
    for idx, batch_data in enumerate(loader_val):
        #batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['mask'])
        img_resized_list = batch_data['image']
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.no_grad():
            # bug fix: segSize = size of last two dimensions
            segSize = (seg_label.shape[2], seg_label.shape[3])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            if torch.cuda.is_available():
                scores = async_copy_to(scores, args.gpu)

            for i in range(img_resized_list.shape[0]):
                feed_dict = batch_data.copy()
                feed_dict['image'] = torch.tensor(img_resized_list[i, :, :, :])
                feed_dict['mask'] = torch.tensor(seg_label[i, :, :, :])

                if torch.cuda.is_available():
                    feed_dict['image'].cuda()
                    feed_dict['mask'].cuda()
                    del feed_dict['name']
                    feed_dict = async_copy_to(feed_dict, args.gpu)
                # print(torch.max(feed_dict['image']))

                # forward pass
                scores_tmp, loss, metric = segmentation_module(
                    feed_dict, mode='test')
                #scores = scores.float() + scores_tmp.float()
                loss_meter.update(loss)
                ave_mse.update(metric[0].data.item())
                ave_pred_pct.update(metric[1].data.item())
                ave_label_pct.update(metric[2].data.item())
                _, pred = torch.max(scores_tmp, dim=1)
                pred = as_numpy(pred.squeeze(0).cpu())
                seg_label_temp = as_numpy(feed_dict['mask'].squeeze(0).cpu())

                # # calculate accuracy
                # intersection, union = intersectionAndUnion(pred, seg_label_temp, args.num_class)
                # intersection_meter.update(intersection)
                # union_meter.update(union)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    # summary
    # iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    # for i, _iou in enumerate(iou):
    #     if i == 1:
    #         print('class [{}], IoU: {:.4f}'.format(i, _iou))
    print('loss: {:.4f}, MSE: {:.4f}, pred-pct: {:.4f}, label-pct: {:.4f}'.format(
        loss_meter.average(), ave_mse.average(), ave_pred_pct.average(), ave_label_pct.average()))
    # wandb.log({"Test IoU": iou[i], "Test Loss": loss_meter.average()})
    writer.add_scalar("Test Loss", loss_meter.average(), n_epoch)
    writer.add_scalar("Test MSE", ave_mse.average(), n_epoch)
    writer.add_scalar("Test Pct", ave_pred_pct.average(), n_epoch)
    return - loss_meter.average()  # iou[1]

# train one epoch


def train(segmentation_module, loader_train, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_mse = AverageMeter()
    ave_pred_pct = AverageMeter()
    ave_label_pct = AverageMeter()
    #ave_acc = AverageMeter()
    #ave_jaccard = AverageMeter()
    #ave_acc_all = AverageMeter()

    segmentation_module.train(not args.fix_bn)

    # main loop
    tic = time.time()

    for batch_data in loader_train:
        data_time.update(time.time() - tic)

        if torch.cuda.is_available():
            batch_data["image"] = batch_data["image"].cuda()
            batch_data["mask"] = batch_data["mask"].cuda()

        segmentation_module.zero_grad()
        # forward pass
        loss, acc = segmentation_module(batch_data, mode='train')
        loss = loss.mean()
        #jaccard = acc[1].float().mean()
        #acc_all = acc[2].float().mean()
        #acc = acc[0].float().mean()

        # Backward
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - tic)
        tic = time.time()

        # update average loss and acc
        ave_mse.update(acc[0].data.item())
        ave_pred_pct.update(acc[1].data.item())
        ave_label_pct.update(acc[2].data.item())
        ave_total_loss.update(loss.data.item())
        # ave_acc.update(acc.data.item()*100)
        # ave_jaccard.update(jaccard.data.item()*100)
        # ave_acc_all.update(acc_all.data.item()*100)
    # calculate accuracy, and display
    print('Epoch: [{}/{}], Time: {:.2f}, Data: {:.2f},'
          ' lr_model: {:.6f}, Loss: {:.6f}, MSE: {:.6f},'
          ' pred-pct: {:.6f}, label-pct: {:.6f}'
          .format(epoch, args.max_iters,
                  batch_time.average(),
                  data_time.average(),
                  args.lr,
                  ave_total_loss.average(),
                  ave_mse.average(),
                  ave_pred_pct.average(),
                  ave_label_pct.average()
                  ))
    writer.add_scalar("Train Loss", ave_total_loss.average(), epoch)
    writer.add_scalar("Train MSE", ave_mse.average(), epoch)
    writer.add_scalars("Train Pct", {'pred-pct': ave_pred_pct.average(),
                                     'label-pct': ave_label_pct.average()}, epoch)

    # args.running_lr_encoder

    history['train']['epoch'].append(epoch)
    history['train']['loss'].append(loss.data.item())
    history['train']['mse'].append(ave_mse.average())
    # history['train']['acc'].append(acc.data.item())
    # history['train']['jaccard'].append(ave_jaccard.average())
    # history['train']['acc_all'].append(ave_acc_all.average())
    # adjust learning rate
    # adjust_learning_rate(optimizers, epoch, args)


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')

    (model, crit) = nets

    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    # torch.save(history,
    #            '{}/history_{}'.format(args.ckpt, suffix_latest))
    dict_model = model.state_dict()
    torch.save(dict_model,
               '{}/model_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    groups = [dict(params=group_decay), dict(
        params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (model, crit) = nets
    '''
    We will use vanilla SGD by default.
    Feel free to implement any other optimizer
    if you wish!
    '''
    # optimizer_model = torch.optim.SGD(
    #         group_weight(model),
    #         lr=args.lr,
    #         momentum=args.beta1,
    #         weight_decay=args.weight_decay)

    # optimizer_model = torch.optim.AdamW(
    #         group_weight(model),
    #         lr=args.lr,
    #         weight_decay=args.weight_decay)
    # return [optimizer_model]
    optimizer_model = Adam(
        group_weight(model),
        lr=args.lr,
        weight_decay=args.weight_decay)
    # optimizer_model = RAdam(params=group_weight(model),
    #                         lr=args.lr,
    #                         weight_decay=args.weight_decay)
    return [optimizer_model]


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = (
        (1. - float(cur_iter) / (args.max_iters)) ** args.lr_pow)
    args.running_lr = args.lr * scale_running_lr

    optimizer_model = optimizers[0]
    for param_group in optimizer_model.param_groups:
        param_group['lr'] = args.running_lr


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        # create one hot from labels
        t = np.bool_(labels.cpu().numpy())
        t_i = np.invert(t)
        targets = np.concatenate((t_i.astype(int), t.astype(int)), axis=1)
        targets = torch.tensor(targets).float()

        # no need to softmax, torch.nn.BCEWithLogitsLoss() has one internally
        logits = logits.cpu().float()
        return self.loss_function(logits, targets)


class diceLoss(nn.Module):
    # adapted from https://github.com/pytorch/pytorch/issues/1249
    # Dice Loss article: https://www.jeremyjordan.me/semantic-segmentation/
    # try with sqaured dice loss
    def __init__(self, epsilon=1e-6):
        super(diceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, logits, labels):
        # calculate dice coff on each class of each training example
        # along W*H axes
        axes = tuple(range(2, len(logits.shape)))

        # convert target label to one hot for dice calculation
        t = np.bool_(labels.cpu().numpy())
        t_i = np.invert(t)
        targets = np.concatenate((t_i.astype(int), t.astype(int)), axis=1)
        targets = torch.tensor(targets).float()

        logits = nn.functional.softmax(logits, dim=1)
        logits = logits.cpu().float()

        # soft dice loss
        numerator = 2. * (targets * logits).sum(axis=axes)
        denominator = (logits.pow(2) + targets.pow(2)).sum(axis=axes)
        return 1 - (numerator / (denominator + self.epsilon)).mean()


def lossFunction():
    '''
    Return the loss function to use.
    Hint: The predicted outputs should be
    logits in the shape of NCHW.
    The target mask is a class indexed tensor
    of shape NHW.
    You can use any loss function which works
    for this task, and/or use a combination.
    (Combinations can sometimes lead to better
    performance.)
    '''
    # return BCELoss()
    return nn.MSELoss()


def main(args):
    # Network Builders
    builder = ModelBuilder()

    model = builder.build_model(args,
                                arch=args.arch,
                                weights=args.weights)

    '''
    Implement the loss function in the function above.
    '''
    crit = lossFunction()

    segmentation_module = SegmentationModule(
        model, crit)

    # Dataset and Loader
    dataset_train = DRIVEData(
        root=args.data_root,
        split='train',
        patch_size=args.patch_size)

    loader_train = data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=False,
        pin_memory=True)

    '''
    Implement the validation dataset loaders
    so we can evaluate on the validation set
    after every training epoch.
    '''
    dataset_val = DRIVEData(
        root=args.data_root,
        split='test',
        patch_size=args.patch_size)
    loader_val = data.DataLoader(
        dataset_val,
        batch_size=8,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=False,
        pin_memory=True)

    # load nets into gpu
    if torch.cuda.is_available():
        segmentation_module.cuda()

    # Set up optimizers
    nets = (model, crit)
    optimizers = create_optimizers(nets, args)

    '''
    You should ideally create an object to keep track of training and validation results
    + best validation scores along with its respective epoch. You should also use the
    checkpoint() function to save the network parameters
    '''
    # initialize history
    history = {'train': {'epoch': [], 'loss': [], 'mse': []}}
    best_val_accuracy = 0
    # create log
    # train_acc_P = Precision
    # train_acc_A = acc
    logname = (f'result/log_{args.id}.csv')
    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(
                ['epoch', 'train_loss', 'train_mse', 'valid_loss', 'valid_mse'])

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, loader_train,
              optimizers, history, epoch, args)
        # eval() returns IoU so we can store it in our history that you implement above.
        iou_val = eval(loader_val, segmentation_module, args, crit, epoch)
        '''
        Update the log you implemented above here.
        '''
        # inference on a whole slide
        if args.test_img:
            folder_name = args.test_img.split('/')[-1]
            pred = infer(args.test_img, segmentation_module,
                         args, mode='whole')

            plt.figure(figsize=[12.8, 9.6], dpi=300)
            plt.imshow(pred, cmap='gray', vmin=0, vmax=255)
            plt.colorbar()
            # plt.show()
            plt.savefig(os.path.join(
                '/home/haotian/Code/vessel_segmentation/visuliz_img', f'{folder_name}-{epoch}.png'), dpi=300)
            plt.close()

        # update log
        train_loss = history['train']['loss']
        train_acc = history['train']['mse']
        #train_acc_all = history['train']['acc_all']
        #train_jaccard = history['train']['jaccard']
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc, iou_val])

        checkpoint(nets, history, args, epoch)
        # save best model
        if iou_val >= best_val_accuracy:
            torch.save(model.state_dict(), os.path.join(
                'ckpt/{}best_model_{}.pth'.format(args.id, epoch)))
            best_val_accuracy = iou_val

    print('Training Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    '''
    Edit DATA_ROOT variable depending on where the data is stored on your env.
    Its relative directory is ./data/img.
    '''
    DATA_ROOT = "/home/haotian/Code/vessel_segmentation/data/hypoxia img"
    DATASET_NAME = "DRIVE"

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', default='hypoxia',
                        help="a name for identifying the model")
    parser.add_argument('--arch', default='default',
                        help="Model architecture")
    parser.add_argument('--weights', default='',
                        help="weights to finetune the model")
    parser.add_argument('--in-channels', default=3, type=int,
                        help="number of input channels")
    parser.add_argument('--out-channels', default=1,
                        type=int, help="number of out channels")

    # Path related arguments
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)
    parser.add_argument('--test-img', type=str,
                        default='/home/haotian/Code/vessel_segmentation/data/hypoxia img/training/img/resized DC 201 L1_HE-green.tif',
                        help='the image in visulize during training process, set to \'\' if not using visulization')

    # optimization related arguments
    parser.add_argument('--gpus', default='0',
                        help='gpus to use (indexes), e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size', default=4, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=190, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--lr', default=3e-4, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-5, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    # Data related argument
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--dataset-name', type=str, default="DRIVE")
    parser.add_argument('--patch-size', default=256, type=int,
                        help='image patch size')
    parser.add_argument('--scale-factor', default=255, type=float,
                        help='scale on the label mask: the label mask is divided by scale-factor')

    # Misc arguments
    parser.add_argument('--seed', default=304, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # Parse gpu ids
    all_gpus = parse_devices(args.gpus)
    all_gpus = [x.replace('gpu', '') for x in all_gpus]
    args.gpus = [int(x) for x in all_gpus]
    num_gpus = len(args.gpus)
    args.gpu = 0

    args.max_iters = args.num_epoch
    args.running_lr = args.lr

    args.arch = args.arch.lower()

    args.id += '-' + str(args.arch)

    # args.id += '-ngpus' + str(num_gpus)
    args.id += '-batchSize' + str(args.batch_size)

    args.id += '-lr' + str(args.lr)

    args.id += '-epoch' + str(args.num_epoch)
    if args.fix_bn:
        args.id += '-fixBN'

    args.id += time.strftime('%b%d_%H-%M', time.localtime())
    print('Model ID: {}'.format(args.id))

    args.ckpt = os.path.join(args.ckpt, args.id)
    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    main(args)
