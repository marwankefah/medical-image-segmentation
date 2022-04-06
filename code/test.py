from configs.configs_mean_teacher import Configs
from dataloaders.dataset import (BaseFetaDataSets, RandomGenerator, ResizeTransform, TwoStreamBatchSampler)
from medpy import metric
from monai.data.utils import decollate_batch
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import random

configs = Configs('./configs/mean_teacher.ini')
db_test = BaseFetaDataSets(configs=configs, split='test', transform=configs.val_transform)

valloader = DataLoader(db_test, batch_size=1, shuffle=False)

medpy_dice_sum = 0

val_loss_list = []

configs.model.to(configs.device)

configs.model.eval()
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

medpy_dice_list=[]
with torch.no_grad():
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
        medpy_dice = metric.binary.dc(y_pred_act[0][1].detach().cpu().numpy(),
                                      val_labels.squeeze().detach().cpu().numpy() > 0.5)

        medpy_dice_list.append(medpy_dice)
        medpy_dice_sum += medpy_dice

medpy_dice_sum = medpy_dice_sum / len(valloader)

val_dice_metric = configs.dice_metric.aggregate().item()

print(np.mean(medpy_dice_list),np.std(medpy_dice_list),medpy_dice_sum, val_dice_metric)

configs.dice_metric.reset()

val_loss_mean = np.mean(val_loss_list, axis=0)

print('dataset loss ', val_loss_mean, ' dice score_medpy: ', medpy_dice_sum, ' dice score monai', val_dice_metric)
