# -*- coding: utf-8 -*-

import numpy as np
import torch
import warnings
import time
from dataProcess import get_dataloader, cal_val_iou, split_train_val_old
import segmentation_models_pytorch as smp
import glob
from segmentation_models_pytorch.losses import DiceLoss, SoftCrossEntropyLoss, LovaszLoss
from pytorch_toolbelt import losses as L
# from model import seg_hrnet_ocr
from torch.optim.swa_utils import AveragedModel, SWALR

warnings.filterwarnings('ignore')

torch.backends.cudnn.enabled = True

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu' 


def train(EPOCHES, BATCH_SIZE, train_image_paths, train_label_paths, 
          val_image_paths, val_label_paths, channels, optimizer_name,
          model_path, swa_model_path, addNDVI, loss, early_stop):
    
    train_loader = get_dataloader(train_image_paths, train_label_paths, 
                                  "train", addNDVI, BATCH_SIZE, shuffle=True, num_workers=8)
    valid_loader = get_dataloader(val_image_paths, val_label_paths, 
                                  "val", addNDVI, BATCH_SIZE, shuffle=False, num_workers=8)
    
    model = smp.UnetPlusPlus(
            encoder_name="timm-regnety_320",#timm-resnest101e
            encoder_weights="imagenet",
            in_channels=channels,
            classes=10,
    )
    model.to(DEVICE);

    if(optimizer_name == "sgd"):
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=1e-4, weight_decay=1e-3, momentum=0.9)
    # 采用AdamM优化器
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=1e-4, weight_decay=1e-3)
    # 余弦退火调整学习率
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=2, # T_0就是初始restart的epoch数目
            T_mult=2, # T_mult就是重启之后因子,即每个restart后，T_0 = T_0 * T_mult
            eta_min=1e-5 # 最低学习率
            )

    
    if(loss == "SoftCE_dice"):

        DiceLoss_fn=DiceLoss(mode='multiclass')
        SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
        loss_fn = L.JointLoss(first=DiceLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()
    else:

        LovaszLoss_fn = LovaszLoss(mode='multiclass')

        SoftCrossEntropy_fn=SoftCrossEntropyLoss(smooth_factor=0.1)
        loss_fn = L.JointLoss(first=LovaszLoss_fn, second=SoftCrossEntropy_fn,
                              first_weight=0.5, second_weight=0.5).cuda()
    
    header = r'Epoch/EpochNum | TrainLoss | ValidmIoU | Time(m)'
    raw_line = r'{:5d}/{:8d} | {:9.3f} | {:9.3f} | {:9.2f}'
    print(header)

    best_miou = 0
    best_miou_epoch = 0
    train_loss_epochs, val_mIoU_epochs, lr_epochs = [], [], []
    # 开始训练
    for epoch in range(1, EPOCHES+1):
        # print("Start training the {}st epoch...".format(epoch))
        # 存储训练集每个batch的loss
        losses = []
        start_time = time.time()
        model.train()
        model.to(DEVICE);
        for batch_index, (image, target) in enumerate(train_loader):
            # print(target)

            image, target = image.to(DEVICE), target.to(DEVICE)
            # 在反向传播前要手动将梯度清零
            optimizer.zero_grad()
            # 模型推理得到输出
            output = model(image)
            # 求解该batch的loss
            loss = loss_fn(output, target)

            # 反向传播求解梯度
            loss.backward()
            # 更新权重参数
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        # 计算验证集IoU
        val_iou = cal_val_iou(model, valid_loader)

        train_loss_epochs.append(np.array(losses).mean())
        val_mIoU_epochs.append(np.mean(val_iou))
        lr_epochs.append(optimizer.param_groups[0]['lr'])
        # 输出进程
        print(raw_line.format(epoch, EPOCHES, np.array(losses).mean(), 
                              np.mean(val_iou), 
                              (time.time()-start_time)/60**1), end="")    
        if best_miou < np.stack(val_iou).mean(0).mean():
            best_miou = np.stack(val_iou).mean(0).mean()
            best_miou_epoch = epoch
            torch.save(model.state_dict(), model_path)
            print("  valid mIoU is improved. the model is saved.")
        else:
            print("")
            if (epoch - best_miou_epoch) >= early_stop:
                break

    return train_loss_epochs, val_mIoU_epochs, lr_epochs



if __name__ == '__main__':
    EPOCHES = 100
    BATCH_SIZE = 16
    #
    image_paths = glob.glob(r'..\sampleData\*.tif')
    label_paths = glob.glob(r'..\sampleData\*.png')


    val_index = 0
    upsample = True
    # 每5个数据的第val_index个数据为验证集
    train_image_paths, train_label_paths, val_image_paths, val_label_paths = split_train_val_old(image_paths,
                                                                                             label_paths,
                                                                                             val_index,
                                                                                             upsample)
    loss = "SoftCE_dice"

    channels = 10 #16 20
    addNDVI = False
    if(addNDVI): 
        channels += 1
    optimizer_name = "adamw"
    model_path = "../model_data/unetplusplus_dpn131"
    if(upsample):
        model_path += "_upsample"
    if(addNDVI):
        model_path += "_ndvi"
    model_path += "_" + loss
    swa_model_path = model_path + "_swa.pth"
    model_path += ".pth"
    early_stop = 10
    train_loss_epochs, val_mIoU_epochs, lr_epochs = train(EPOCHES, 
                                                          BATCH_SIZE, 
                                                          train_image_paths, 
                                                          train_label_paths, 
                                                          val_image_paths, 
                                                          val_label_paths, 
                                                          channels, 
                                                          optimizer_name,
                                                          model_path, 
                                                          swa_model_path, 
                                                          addNDVI,
                                                          loss,
                                                          early_stop)


    if(True):    
        import matplotlib.pyplot as plt
        epochs = range(1, len(train_loss_epochs) + 1)
        plt.plot(epochs, train_loss_epochs, 'r', label = 'train loss')
        plt.plot(epochs, val_mIoU_epochs, 'b', label = 'val mIoU')
        plt.title('train loss and val mIoU')
        plt.legend()
        plt.savefig("train loss and val mIoU.png",dpi = 300)
        plt.figure()
        plt.plot(epochs, lr_epochs, 'r', label = 'learning rate')
        plt.title('learning rate')
        plt.legend()
        plt.savefig("learning rate.png", dpi = 300)
        plt.show() 
