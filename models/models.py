import torch
import torch.nn as nn
import torchvision
import numpy as np
#from . import resnet, resnext, mobilenet, dpn, drn
from lib.nn import SynchronizedBatchNorm2d
import math
from collections import OrderedDict

'''
I have already implemented the classes SegmentationModuleBase,
SegmentationModule, and ModelBuilder. Your task is to write the
code for your model of choice in the Model class.
'''


class SegmentationModuleBase(nn.Module):
    def __init__(self):
        super(SegmentationModuleBase, self).__init__()

    def pixel_acc(self, pred, label):
        _, preds = torch.max(pred, dim=1)
        preds = preds.unsqueeze(1)
        valid = (label >= 1).long()
        acc_sum = torch.sum(valid * (preds == label).long())
        pixel_sum = torch.sum(valid)
        valid_neg = (label < 1).long()
        acc_sum_neg = torch.sum(valid_neg * (preds == label).long())
        acc_all = (acc_sum.float() + acc_sum_neg.float()) / \
            (preds.shape[-1]*preds.shape[-1]*preds.shape[0])

        # When you +falsePos, acc == Jaccard.
        acc = acc_sum.float() / (pixel_sum.float() + 1e-10)

        # class 1
        v1 = (label == 1).long()
        pred1 = (preds == 1).long()
        anb1 = torch.sum(v1 * pred1)
        try:
            j1 = anb1.float() / (torch.sum(v1).float() +
                                 torch.sum(pred1).float() - anb1.float() + 1e-10)
        except:
            j1 = 0

        j1 = j1 if j1 <= 1 else 0

        jaccard = j1
        return acc, jaccard, acc_all

    # ACCURACY THAT TAKES INTO ACCOUNT BOTH TP AND FP.
    def jaccard(self, pred, label):
        AnB = torch.sum(pred.long() & label)  # TAKE THE AND
        return AnB/(pred.view(-1).sum().float() + label.view(-1).sum().float() - AnB)

    # MSE metrics
    def mse(self, pred, label):
        return torch.sum((pred - label) ** 2)

class SegmentationModule(SegmentationModuleBase):
    def __init__(self, model, crit):
        super(SegmentationModule, self).__init__()
        self.model = model
        self.crit = crit

    def forward(self, feed_dict,  *,  mode='train'):
        assert mode in ['train', 'test', 'result']
        # training
        if mode == 'train':
            '''
            Note: since we want the logits to use in the loss function,
            we do not softmax pred.
            '''
            pred = self.model(feed_dict['image'])  # (4,1,64,64)
            loss = self.crit(pred, feed_dict['mask'])
            #acc = self.pixel_acc(torch.round(nn.functional.softmax(
            #    pred, dim=1)).long(), feed_dict['mask'].long())
            metric = self.mse(pred, feed_dict['mask'])
            return loss, metric
        # inference
        else:
            p = self.model(feed_dict['image'].unsqueeze(0))
            loss = self.crit(p, feed_dict['mask'].unsqueeze(0))
            '''
            Note: we softmax the pred after calculating the validation loss.
            The values in pred are now in the range [0, 1].
            '''
            metric = self.mse(p, feed_dict['mask'])
            pred = nn.functional.softmax(p, dim=1)
            return pred, loss, metric

    def infer(self, input):
        pred = self.model(input)
        return pred


class ModelBuilder():
    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def build_model(self, args, arch='default', weights='i'):
        arch = arch.lower()

        if arch == 'default':
            model = Model(in_channels=1, out_channels=1)
        else:
            raise Exception('Architecture undefined!')

        if len(weights) > 0:

            model.load_state_dict(
                torch.load(weights, map_location=lambda storage, loc: storage), strict=False)

            print("Loaded pretrained model weights.")
        print('Loaded weights for model.')
        return model.double()


class Model(nn.Module):
    '''
    Implement any model you wish here.
    Do some research on some standard
    models used in medical imaging segmentation.
    Let us know why you chose the model you chose.
    Also let us know the pros and cons of the model
    you chose.
    '''
    # code adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch/blob/master/unet.py

    def __init__(self, in_channels=3, out_channels=1, init_features=32):
        super(Model, self).__init__()

        features = init_features
        self.encoder1 = Model._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = Model._block(features, features * 2, name="enc2")
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = Model._block(features * 2, features * 4, name="enc3")
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = Model._block(features * 4, features * 8, name="enc4")
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.bottleneck = Model._block(
            features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose2d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = Model._block(
            (features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose2d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = Model._block(
            (features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose2d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = Model._block(
            (features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose2d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = Model._block(features * 2, features, name="dec1")

        self.conv = nn.Conv2d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))

        bottleneck = self.bottleneck(self.pool4(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)
        # other funcs can be tested, tanh, relu, softplus
        # return nn.functional.sigmoid(self.conv(dec1))

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv2d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm2d(num_features=features)),
                    (name + "relu1", nn.ReLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv2d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm2d(num_features=features)),
                    (name + "relu2", nn.ReLU(inplace=True)),
                ]
            )
        )
