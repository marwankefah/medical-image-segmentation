# -*- coding: utf-8 -*-
"""
Created on Mon Jul 12 12:47:42 2021

@author: Prinzessin
"""
import configparser
import os
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss, BCELoss

import monai
import segmentation_models_pytorch as smp

from monai.transforms import (
    Activations,
    AddChanneld,
    AsDiscrete,
    Compose,
    LoadImaged,
    RandFlipd,
    RandRotated,
    RandZoomd,
    ScaleIntensityd,
    EnsureTyped,
    Resized,
    RandGaussianNoised,
    RandGaussianSmoothd,
    Rand2DElasticd,
    RandAffined,
    OneOf,
    NormalizeIntensity,
    AsChannelFirstd,
    EnsureType,
    LabelToMaskd
)
from monai.data.image_reader import PILReader
from monai.metrics import DiceMetric

import torch
import numpy as np


class Configs:
    def __init__(self, filename):

        # =============================================================================
        # Readable ini file
        # =============================================================================
        self.config_filename = filename
        config_file = configparser.ConfigParser(allow_no_value=True)
        config_file.read(self.config_filename)

        self.root_path = config_file.get('path', 'root_path', fallback='../data/FETA/')
        self.linux_gpu_id = config_file.get('path', 'linux_gpu_id', fallback=0)
        self.linux = config_file.getboolean('path', 'linux', fallback=False)

        self.img_root_path = config_file.get('path', 'img_root_path', fallback='../data/FETA/')

        self.exp = config_file.get('path', 'exp', fallback='FETA/Mean_Teacher')
        self.model_name = config_file.get('path', 'model_name', fallback='unetResnet34')

        self.multi_class = config_file.getboolean('network', 'multi_class', fallback=True)

        self.optim = config_file.get('network', 'optim', fallback='adam')

        self.psuedoLabelsGenerationEpoch = config_file.getint('network', 'psuedoLabelsGenerationEpoch', fallback=3)
        self.mean_teacher_epoch = config_file.getint('network', 'mean_teacher_epoch', fallback=3)
        self.num_workers = config_file.getint('network', 'num_workers', fallback=0)

        self.val_batch_size = config_file.getint('network', 'val_batch_size', fallback=16)

        self.generationLowerThreshold = config_file.getfloat('network', 'generationLowerThreshold', fallback=0.05)
        self.generationHigherThreshold = config_file.getfloat('network', 'generationHigherThreshold', fallback=0.02)

        if self.linux == True:
            os.environ['CUDA_VISIBLE_DEVICES'] = self.linux_gpu_id

        self.backbone = config_file.get('network', 'backbone', fallback='resnet34')
        self.max_iterations = config_file.getint('network', 'max_iterations', fallback=30000)
        self.batch_size = config_file.getint('network', 'batch_size', fallback=16)
        self.labeled_bs = config_file.getint('network', 'labeled_bs', fallback=8)
        self.deterministic = config_file.getint('network', 'deterministic', fallback=1)
        self.base_lr = config_file.getfloat('network', 'base_lr', fallback=0.01)

        patch_size = config_file.get('network', 'patch_size', fallback='[256, 256]')
        self.patch_size = [int(number) for number in patch_size[1:-1].split(',')]

        self.seed = config_file.getint('network', 'seed', fallback=1337)
        self.num_classes = config_file.getint('network', 'num_classes', fallback=2)
        self.in_channels = config_file.getint('network', 'in_channels', fallback=1)

        # costs
        self.ema_decay = config_file.getfloat('network','ema_decay',fallback=0.99)
        self.consistency_type = config_file.get('network','consistency_type',fallback='mse')
        self.consistency = config_file.getfloat('network','consistency',fallback=0.1)
        self.consistency_rampup = config_file.getfloat('network','consistency_rampup',fallback= 200.0)

        # Model

        aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            # dropout=0.5,  # dropout ratio, default is None
            # activation='sigmoid',  # activation function, default is None
            classes=self.num_classes,  # define number of output labels
        )

        def create_model(ema=False):
            model = smp.Unet(
                encoder_name=self.backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
                encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
                in_channels=self.in_channels,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
                classes=self.num_classes,  # model output channels (number of classes in your dataset)
                aux_params=aux_params)
            if ema:
                for param in model.parameters():
                    param.detach_()
            return model

        self.model = create_model()
        self.ema_model = create_model(ema=True)

        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # TODO abstract and add more optimizers
        if self.optim.lower() == 'sgd':

            self.optimizer = optim.SGD(self.model.parameters(), lr=self.base_lr,
                                       momentum=0.9, weight_decay=0.0001)
        elif self.optim.lower() == 'adam':

            self.optimizer = optim.Adam(self.model.parameters(), lr=self.base_lr)
        else:
            raise Exception("Optimizer is not supported")

        # TODO include background?
        self.criterion = monai.losses.DiceLoss(include_background=True, softmax=True,to_onehot_y=True)
        self.criterion_1 = CrossEntropyLoss()
        # writers
        self.train_writer = None
        self.val_writer = None

        image_loader = None
        channel_transform = None
        if self.in_channels == 1:
            image_loader = PILReader(converter=lambda image: image.convert("L"))
            channel_transform = AddChanneld(keys=["image", "label"])
        elif self.in_channels == 3:
            image_loader = PILReader(converter=lambda image: image.convert("RGB"))
            channel_transform = AsChannelFirstd(keys=["image", "label"])
        else:
            raise Exception("input channel is not supported")

        deform = Rand2DElasticd(
            keys=["image", "label"],
            prob=0.5,
            spacing=(7, 7),
            magnitude_range=(1, 2),
            rotate_range=(np.pi / 6,),
            scale_range=(0.2, 0.2),
            translate_range=(20, 20),
            padding_mode="zeros",
            # device=self.device,
        )

        affine = RandAffined(
            keys=["image", "label"],
            prob=0.5,
            rotate_range=(np.pi / 6),
            scale_range=(0.2, 0.2),
            translate_range=(20, 20),
            padding_mode="zeros",
            # device=self.device
        )

        # TODO joaquin check transforms again
        self.train_transform = Compose(
            [
                LoadImaged(keys=["image", "label"], reader=image_loader),

                channel_transform,

                LabelToMaskd(keys=["label"], select_labels=[1]),

                ScaleIntensityd(keys=["image", "label"]),

                Resized(keys=["image", "label"], spatial_size=(self.patch_size[0], self.patch_size[1])),
                RandRotated(keys=["image", "label"], range_x=(-np.pi / 6, np.pi / 6), prob=0.5, keep_size=True),

                RandFlipd(keys=["image", "label"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image", "label"], spatial_axis=1, prob=0.5),

                RandZoomd(keys=["image", "label"], min_zoom=0.9, max_zoom=1.1, prob=0.5),

                RandGaussianSmoothd(keys=["image"], prob=0.1, sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5)),
                RandGaussianNoised(keys=["image"], mean=0, std=0.1, prob=0.5),

                # TODO check this
                OneOf(transforms=[affine, deform], weights=[0.8, 0.2]),
                # NormalizeIntensity(subtrahend=None, divisor=None, channel_wise=False),

                EnsureTyped(keys=["image", "label"], ),
            ]
        )

        self.teacher_transform = Compose(
            [
                LoadImaged(keys=["image"], reader=image_loader),

                #TODO abstarct for teacher to use
                # AsChannelFirstd(keys=["image"]),
                AddChanneld(keys=["image"]),

                ScaleIntensityd(keys=["image"]),

                Resized(keys=["image"], spatial_size=(self.patch_size[0], self.patch_size[1])),

                RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),
                RandFlipd(keys=["image"], spatial_axis=1, prob=0.5),

                EnsureTyped(keys=["image"] ),
            ]
        )
        self.val_transform = Compose(
            [
                LoadImaged(keys=["image", "label"], reader=image_loader),

                channel_transform,

                LabelToMaskd(keys=["label"], select_labels=[1]),

                ScaleIntensityd(keys=["image", "label"]),

                # NormalizeIntensity(subtrahend=None, divisor=None, channel_wise=False),

                Resized(keys=["image", "label"], spatial_size=(self.patch_size[0], self.patch_size[1])),
                EnsureTyped(keys=["image", "label"])
            ])

        self.y_pred_trans = Compose([EnsureType(), Activations(softmax=True), AsDiscrete(argmax=True, to_onehot=self.num_classes)])

        self.y_trans = AsDiscrete(threshold=0.1, to_onehot=self.num_classes)

        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

    # TODO add more scheduling techniques for other optimizers?
    def update_lr(self, iter_num):
        if self.optim.lower() == 'sgd':
            lr_ = self.base_lr * (1.0 - iter_num / self.max_iterations) ** 0.9
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr_
