import argparse
import logging
import os
import random
import shutil
import sys
import time

from monai.data.utils import decollate_batch
from tensorboardX import SummaryWriter
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from configs.configs_mean_teacher import *
from dataloaders.dataset import (BaseFetaDataSets,TwoStreamBatchSampler)
from configs.configs_mean_teacher import Configs
from monai.visualize import plot_2d_or_3d_image
from medpy import metric
from utils import ramps
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return configs.consistency * ramps.sigmoid_rampup(epoch, configs.consistency_rampup)

def train(configs, snapshot_path):
    configs.train_writer = SummaryWriter(snapshot_path + '/log')
    configs.val_writer = SummaryWriter(snapshot_path + '/log_val')

    configs.model.to(configs.device)
    configs.ema_model.to(configs.device)
    db_train = BaseFetaDataSets(configs=configs, split='train', transform=configs.train_transform,teacher_transform=configs.teacher_transform)
    db_val = BaseFetaDataSets(configs=configs, split='val', transform=configs.val_transform)

    labeled_idxs = db_train.labeled_idxs
    unlabeled_idxs = db_train.unlabeled_idxs


    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, configs.batch_size, configs.batch_size-configs.labeled_bs)


    trainloader = DataLoader(db_train, num_workers=configs.num_workers,batch_sampler=batch_sampler,
                             pin_memory=True)

    valloader = DataLoader(db_val, batch_size=configs.val_batch_size, shuffle=False,
                           num_workers=configs.num_workers)





    configs.model.train()

    writer = configs.train_writer
    writer_val = configs.val_writer

    logging.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0

    max_epoch = configs.max_iterations // len(trainloader) + 1

    # TODO AUC best_performance = 0.5 min to sasve the model
    best_performance = float(0.5)

    # TODO create a running loss to make sure loss is calculated corretly
    iterator = tqdm(range(max_epoch), ncols=70)

    for epoch_num in iterator:
        y_pred_list = []
        y_true_list = []

        y_true_val_list = []
        y_pred_val_list = []

        train_loss_list = []
        val_loss_list = []

        for i_batch, sampled_batch in enumerate(trainloader):

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.to(configs.device), label_batch.to(configs.device)

            unlabeled_volume_batch = volume_batch[configs.labeled_bs:]

            noise = torch.clamp(torch.randn_like(
                unlabeled_volume_batch) * 0.1, -0.2, 0.2)

            ema_inputs = unlabeled_volume_batch + noise
            with torch.no_grad():
                ema_outputs, ema_classification = configs.ema_model(ema_inputs)
                ema_output_soft = torch.softmax(ema_outputs, dim=1)

            # print(db_train.y_test[sampled_batch['idx'].squeeze().detach().cpu().numpy()])


            outputs, classification_head_output = configs.model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)

            loss_dice = configs.criterion(outputs[:configs.labeled_bs],
                                          label_batch[:][:configs.labeled_bs].long())

            loss_ce = configs.criterion_1(outputs[:configs.labeled_bs],
                                          label_batch[:][:configs.labeled_bs].squeeze(1).long())

            if epoch_num < configs.mean_teacher_epoch:
                consistency_loss = 0.0
            else:
                consistency_loss = torch.mean(
                    (outputs_soft[configs.labeled_bs:] - ema_output_soft) ** 2)


            supervised_loss = 0.5 * (loss_ce + loss_dice)

            consistency_weight = get_current_consistency_weight(iter_num//150)

            loss= supervised_loss + consistency_weight * consistency_loss

            configs.optimizer.zero_grad()
            loss.backward()
            configs.optimizer.step()

            train_loss_list.append(loss.detach().cpu().numpy())

            iter_num = iter_num + 1

            configs.update_lr(iter_num)
            lr_ = configs.optimizer.param_groups[0]['lr']

            writer.add_scalar('info/lr', lr_, iter_num)

            writer.add_scalar('info/consistency_loss',
                              consistency_loss, iter_num)
            writer.add_scalar('info/consistency_weight',
                              consistency_weight, iter_num)

            logging.info(
                'iteration %d : loss : %f, loss_ce: %f, loss_dice: %f' %
                (iter_num, loss.item(), loss_ce.item(), loss_dice.item()))

            if iter_num % 20 == 0:
                plot_2d_or_3d_image(volume_batch, iter_num + 1, writer, index=0, tag="image")
                plot_2d_or_3d_image(label_batch, iter_num + 1, writer, index=0, tag="label")

                output_mask = configs.y_pred_trans(outputs[0])

                plot_2d_or_3d_image(output_mask, iter_num + 1, writer, index=1, tag="output")

        configs.model.eval()
        medpy_dice = 0

        for i_batch, sampled_batch in enumerate(valloader):
            val_images, val_labels = sampled_batch["image"].to(configs.device), sampled_batch["label"].to(
                configs.device)

            val_outputs, val_classification_output = configs.model(val_images)

            val_dice_loss = configs.criterion(val_outputs, val_labels.long())

            val_ce_loss = configs.criterion_1(val_outputs, val_labels.squeeze(1).long())

            val_loss = 0.5 * (val_ce_loss + val_dice_loss)
            val_loss_list.append(val_loss.detach().cpu().numpy())

            y_onehot = [configs.y_trans(i) for i in decollate_batch(val_labels)]
            y_pred_act = [configs.y_pred_trans(i) for i in decollate_batch(val_outputs)]

            configs.dice_metric(y_pred_act, y_onehot)

            medpy_dice += metric.binary.dc(y_pred_act[0][1].detach().cpu().numpy(),
                                           val_labels.squeeze().detach().cpu().numpy() > 0.5)

        medpy_dice = medpy_dice / len(valloader)

        val_dice_metric = configs.dice_metric.aggregate().item()

        print(medpy_dice, val_dice_metric)

        configs.dice_metric.reset()

        val_loss_mean = np.mean(val_loss_list, axis=0)
        #
        train_loss_mean = np.mean(train_loss_list)

        writer.add_scalar('info/model_total_loss', train_loss_mean, epoch_num)
        writer_val.add_scalar('info/model_total_loss', val_loss_mean, epoch_num)
        writer_val.add_scalar('info/val_dice', val_dice_metric, epoch_num)

        plot_2d_or_3d_image(val_images, epoch_num + 1, writer_val, index=0, tag="image")
        plot_2d_or_3d_image(val_labels, epoch_num + 1, writer_val, index=0, tag="label")
        plot_2d_or_3d_image(y_pred_act[0], epoch_num + 1, writer_val, index=1, tag="output")

        logging.info(
            'iteration %d : val_loss : %f val_dice : %f' % (iter_num, val_loss_mean, val_dice_metric))
        #
        performance = val_dice_metric

        if performance > best_performance:
            best_performance = performance
            save_mode_path = os.path.join(snapshot_path,
                                          'epoch_{}_val_dice_{}.pth'.format(
                                              epoch_num, round(best_performance, 4)))
            # TODO save to best performance path
            save_best = os.path.join(snapshot_path,
                                     '{}_best_model.pth'.format(configs.model_name))
            torch.save(configs.model.state_dict(), save_mode_path)
            torch.save(configs.model.state_dict(), save_best)
        #
        if iter_num >= configs.max_iterations:
            break
        configs.model.train()
        print(epoch_num + 1, ' epoch finished')

    writer.close()
    writer_val.close()
    return "Training Finished!"


if __name__ == "__main__":

    configs = Configs('./configs/mean_teacher.ini')

    if not configs.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(configs.seed)
    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)

    log_time = int(time.time())


    snapshot_path = "../model/{}_labeled/{}/{}".format(
        configs.exp, configs.model_name, log_time)

    # TODO create data similar to data_split_csv.py every run
    main_dir = '/mnt/mia_images/breast/omi-db/iceberg_selection/HOLOGIC/'
    data_output_dir = os.path.join(snapshot_path, '/data/')
    # roi_dir = main_dir + 'roi/'

    #
    # if not os.path.exists(data_output_dir):
    #     os.makedirs(data_output_dir)
    #

    # create_splits(data_output_dir,roi_dir,[80,20,20])

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')

    shutil.copytree('.', snapshot_path + '/code',
                    shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path + "/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(vars(configs)))
    train(configs, snapshot_path)
