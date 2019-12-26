# System libs
import os
import time
import argparse
from distutils.version import LooseVersion
# Numerical libs
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
# Our libs
from data.dataset import DRIVEData
from models.models import ModelBuilder, SegmentationModule
from utils import AverageMeter, colorEncode, accuracy, intersectionAndUnion, confusion_matrix
from lib.nn import user_scattered_collate, async_copy_to
from lib.utils import as_numpy
import lib.utils.data as torchdata
import cv2
from tqdm import tqdm

# confusion matrix
import matplotlib.pyplot as plt
#########from loss import ACLoss

from PIL import Image

def visualize_result(data, pred, pred_color, args):
    (img, seg, info) = data
    img = img*255
    seg = seg*255
    pred = pred*255
    pred_color = pred_color*255
    
    # create collage for img, mask and pred
    for i in range(args.batch_size):
        name = info[i]
        
        img_collage = np.concatenate((img[i,:,:,:], seg[i,:,:,:], pred[i,:,:,:], pred_color[i,:,:,:]), axis=-1)
        img_collage_t = np.transpose(img_collage, (1, 2, 0))
        img_collage_t = Image.fromarray(img_collage_t.astype('uint8'), 'RGB')
        
        img_collage_t.save(os.path.join(args.result, '{}_collage.png'.format(name)), 'PNG')
          
    '''
    Write a function that saves an image
    containing the original img (img), the ground
    truth segmentation (seg), and the predicted seg
    (pred). The img should be saved in the directory
    of args.result.
    '''

def plot_confusion_matrix(cm, classes, args,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(os.path.join(args.result, 'confusion_matrix.png'))
'''
evaluate() is already completely implemented for you.
If any bugs occur, please fix it however.
'''

def evaluate(segmentation_module, loader, args):
    acc_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    time_meter = AverageMeter()

    tp_meter = AverageMeter()
    fp_meter = AverageMeter()
    fn_meter = AverageMeter()
    tn_meter = AverageMeter()
    
    segmentation_module.eval()
    #pbar = tqdm(total=len(loader))
    for batch_data in loader:
        # process data
        # batch_data = batch_data[0]
        seg_label = as_numpy(batch_data['mask'])
        img_resized_list = batch_data['image']
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tic = time.perf_counter()
        
        # initialize the array to store the reconstruction of img, mask and preds of the batch
        factor = int(args.img_size/args.patch_size)
        batch_preds = np.zeros([args.batch_size,3,args.img_size,args.img_size])
        batch_preds_color = np.zeros([args.batch_size,3,args.img_size,args.img_size])
        batch_mask = np.zeros([args.batch_size,3,args.img_size,args.img_size])
        batch_img = np.zeros([args.batch_size,3,args.img_size,args.img_size])
        
        feed_dict = batch_data.copy()
        
        with torch.no_grad():
            segSize = (seg_label.shape[2], seg_label.shape[3])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            if torch.cuda.is_available():
                scores = async_copy_to(scores, args.gpu)
                
            for i in range(img_resized_list.shape[0]):
                for j in range(factor**2):
                    feed_dict['image'] = torch.tensor(img_resized_list[i,j,:,:,:])
                    feed_dict['mask'] =torch.tensor(seg_label[i,j,:,:,:])
                    
                    if torch.cuda.is_available():
                        feed_dict['image'].cuda()
                        feed_dict['mask'].cuda()
                        #del feed_dict['name']
                        feed_dict = async_copy_to(feed_dict, args.gpu)

                    # forward pass
                    scores_tmp, loss = segmentation_module(feed_dict, mode='result')
                    #scores = scores + scores_tmp / len(args.imgSize)
                    _, pred = torch.max(scores_tmp, dim=1)
                    
                    # update tp, fp, fn, tn
                    tp, fp, fn, tn = confusion_matrix(pred, feed_dict['mask'].long())
                    tp_meter.update(tp.sum())
                    fp_meter.update(fp.sum())
                    fn_meter.update(fn.sum())
                    tn_meter.update(tn.sum())
                
                    pred = as_numpy(pred.squeeze(0).cpu())
                    seg_label_temp = as_numpy(feed_dict['mask'].squeeze(0).cpu())
                    # update acc and IoU
                    acc, pix = accuracy(pred, seg_label_temp)
                    intersection, union = intersectionAndUnion(pred, seg_label_temp, args.num_class)
                    intersection_meter.update(intersection)
                    union_meter.update(union)
                    acc_meter.update(acc, pix)

                    # recombine the patches to the whole img, mask and pred
                    tp_pred = as_numpy(tp.squeeze(0).cpu())
                    fp_pred = as_numpy(fp.squeeze(0).cpu())
                    fn_pred = as_numpy(fn.squeeze(0).cpu())

                    # i: # in the batch
                    # n: row subscript
                    # k: column subscript
                    n = int(j/factor)
                    k = int(j%factor)

                    batch_preds[i, :, n*args.patch_size:(n+1)*args.patch_size, 
                                 k*args.patch_size:(k+1)*args.patch_size] = pred[:,:]

                    # RGB
                    batch_preds_color[i, :, n*args.patch_size:(n+1)*args.patch_size, 
                                 k*args.patch_size:(k+1)*args.patch_size] = tp_pred[:,:]
                    # fp red
                    batch_preds_color[i, 1:2, n*args.patch_size:(n+1)*args.patch_size, 
                                 k*args.patch_size:(k+1)*args.patch_size] = tp_pred[:,:] + fp_pred[:,:]
                    # fn green
                    batch_preds_color[i, 0:1, n*args.patch_size:(n+1)*args.patch_size, 
                                 k*args.patch_size:(k+1)*args.patch_size] = tp_pred[:,:] + fn_pred[:,:]                  

                    batch_mask[i, :, n*args.patch_size:(n+1)*args.patch_size, 
                                 k*args.patch_size:(k+1)*args.patch_size] = feed_dict['mask'][:,:]
                    batch_img[i, :, n*args.patch_size:(n+1)*args.patch_size, 
                                 k*args.patch_size:(k+1)*args.patch_size] = feed_dict['image'][:,:,:]
            

        if True:# args.visualize
            visualize_result((batch_img, batch_mask, batch_data['name']), batch_preds, batch_preds_color, args)
                    #visualize_result((batch_data['image'], seg_label, batch_data['name']), pred, args)
                    
                
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        time_meter.update(time.perf_counter() - tic)
        #pbar.update(1)
    # confusion matrix

    # summary
    iou = intersection_meter.sum / (union_meter.sum + 1e-10)
    for i, _iou in enumerate(iou):
        print('class [{}], IoU: {:.4f}'.format(i, _iou))

    print('[Eval Summary]:')
    print('Mean IoU: {:.4f}, Accuracy: {:.2f}%, Inference Time: {:.4f}s'
          .format(iou.mean(), acc_meter.average()*100, time_meter.average()))
 
    
    acc_all = float((tp_meter.sum.item()+tn_meter.sum.item()))/(tp_meter.sum.item()+fp_meter.sum.item()+fn_meter.sum.item()+tn_meter.sum.item())
    sen = float(tp_meter.sum.item())/(tp_meter.sum.item()+fn_meter.sum.item())
    spe = float(tn_meter.sum.item())/(fp_meter.sum.item()+tn_meter.sum.item())
    print('Overall acc: {:.4f}, Sensitivity: {:.4f}, Specificity: {:.4f}'.format(acc_all, sen, spe))

    cm = np.asarray([[tp_meter.sum, fp_meter.sum],[fn_meter.sum, tn_meter.sum]])
    classes = ['1','0']
    plot_confusion_matrix(cm, classes, args, normalize=True,
                      title='Confusion matrix')
        
class BCELoss(nn.Module):
    # pixel-wise cross entropy loss
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_function = torch.nn.BCEWithLogitsLoss()

    def forward(self, logits, labels):
        t =  np.bool_(labels.cpu().numpy())
        t_i = np.invert(t)
        targets = np.concatenate((t_i.astype(int), t.astype(int)), axis = 1)
        targets = torch.tensor(targets).float()
        logits = logits.cpu().float()
        return self.loss_function(logits, targets)

class diceLoss(nn.Module):
    # adapted from https://github.com/pytorch/pytorch/issues/1249
    # Dice Loss article: https://www.jeremyjordan.me/semantic-segmentation/
    def __init__(self, epsilon=1e-6):
        super(diceLoss, self).__init__()
        self.epsilon = epsilon
    
    def forward(self, logits, labels):
        # calculate dice coff on each class of each training example
        # along W*H axes
        axes = tuple(range(2, len(logits.shape)))
        
        # convert target label to one hot for dice calculation
        t =  np.bool_(labels.cpu().numpy())
        t_i = np.invert(t)
        targets = np.concatenate((t_i.astype(int), t.astype(int)), axis = 1)
        targets = torch.tensor(targets).float()
        logits = nn.functional.softmax(logits, dim=1)
        logits = logits.cpu().float()
        numerator = 2. * (targets * logits).sum(axis=axes)
        denominator = (logits.pow(2) + targets.pow(2)).sum(axis = axes)
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

    If you've already implemented this in train.py,
    just copy-and-paste the code over.
    '''
    
    return diceLoss()
    

def main(args):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # Network Builders
    builder = ModelBuilder()
    model = builder.build_model(num_class=args.num_class,
        arch=args.arch,
        weights=args.weights)

    crit = lossFunction()

    segmentation_module = SegmentationModule(model, crit)

    '''
    Implement the dataset variables the same as you
    did in train.py.
    '''
    # Dataset and Loader
    dataset_val = DRIVEData(
            root=args.data_root,
            split='result')
    loader_val = data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    if torch.cuda.is_available():
        segmentation_module.cuda()

    # Main loop
    evaluate(segmentation_module, loader_val, args)

    print('Evaluation Done!')


if __name__ == '__main__':
    assert LooseVersion(torch.__version__) >= LooseVersion('0.4.0'), \
        'PyTorch>=0.4.0 is required'

    '''
    Edit DATA_ROOT variable depending on where the data is stored on your env.
    Its relative directory is ./data/img.
    '''
    DATA_ROOT = os.getenv('DATA_ROOT', './data/img')

    parser = argparse.ArgumentParser()
    # Model related arguments
    parser.add_argument('--id', required=True,
                        help="a name for identifying the model to load")
    parser.add_argument('--arch', default='default',
                        help='Model architecture?')

    # Data related arguments
    parser.add_argument('--num_val', default=-1, type=int,
                        help='number of images to evalutate')
    parser.add_argument('--num_class', default=2, type=int,
                        help='number of classes')
    parser.add_argument('--batch_size', default=1, type=int,
                        help='batchsize. current only supports 1')
    parser.add_argument('--checkpoint', type=str, required=True, help="checkpoint path")
    parser.add_argument('--test-split', type=str, default='test')
    parser.add_argument('--data-root', type=str, default=DATA_ROOT)

    # Misc argument
    parser.add_argument('--ckpt', default='./ckpt',
                        help='folder to output checkpoints')
    parser.add_argument('--visualize', action='store_true',
                        help='output visualization?')
    parser.add_argument('--result', default='./result',
                        help='folder to output visualization results')
    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu id for evaluation')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--patch_size', default=0, type=int,
                        help='patch size for evaluation')
    parser.add_argument('--img_size', default=0, type=int,
                        help='img size for reconstruction')
    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    # absolute paths of model weights
    args.weights = args.checkpoint
    assert os.path.exists(args.weights), 'checkpoint does not exist!'

    args.result = os.path.join(args.result, args.id)
    if not os.path.isdir(args.result):
        os.makedirs(args.result)

    main(args)
