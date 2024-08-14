import time
from tkinter import Image
from torch.autograd import Variable

from glo import *
# from global_ import *
import torch
import torch.nn as nn
import torch.utils.data
import os
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch.nn.functional as F

from global_annos import annos_list

class dice_loss(nn.Module):
    def __init__(self, c_num=2):
        super(dice_loss, self).__init__()

    def forward(self, data, label):
        n = data.size(0)
        dice_list = []
        all_dice = 0.
        for i in range(n):
            my_label11 = label[i]
            my_label1 = torch.abs(1 - my_label11)
            my_data1 = data[i][0]
            my_data11 = data[i][1]
            m1 = my_data1.view(-1)
            m2 = my_label1.view(-1)
            m11 = my_data11.view(-1)
            m22 = my_label11.view(-1)
            dice = 0.
            dice += (1 - ((2. * (m1 * m2).sum() + 1) / (m1.sum() + m2.sum() + 1)))
            dice += (1 - ((2. * (m11 * m22).sum() + 1) / (m11.sum() + m22.sum() + 1)))
            dice_list.append(dice)
        for i in range(n):
            all_dice += dice_list[i]
        dice_loss = all_dice / n
        return dice_loss

class FocalLoss(nn.Module):

    def __init__(self, weight=None, reduction='mean', gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(weight=weight, reduction=reduction)

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        # p = torch.exp(-logp)
        pt = torch.sigmoid(input)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

Loss = nn.CrossEntropyLoss().to(DEVICE)

def train_model(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_need = []
    tqdr = tqdm(enumerate(train_loader))
    for batch_index, (data, target) in tqdr:
        time.sleep(0.01)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = Loss(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss = loss.item()
        loss_need.append(train_loss)
        tqdr.set_description("Train Epoch : {} \t train Loss : {:.6f} ".format(epoch, loss.item()))
    train_loss = np.mean(loss_need)
    return train_loss, loss_need

def tst_model(model, device, test_loader, epoch, test):
    model.eval()
    test_loss = 0.0
    PA = IOU = DICE = P = R = F1 = 0
    tqrr = tqdm(enumerate(test_loader))
    with torch.no_grad():
        for batch_index, (data, target) in tqrr:
            if test:
                data_cpu = data.clone().cpu()
                my_label_cpu = target.clone().cpu()
                for i in range(len(data_cpu)):
                    true_img_tensor = data_cpu[i][0]
                    true_label_tensor = my_label_cpu[i]
                    use_plot_2d(true_img_tensor, true_label_tensor, batch_index=batch_index, i=i,
                                true_label=True)
            data, target = data.to(device), target.to(device)
            torch.cuda.empty_cache()
            output = model(data)
            loss = Loss(output, target)
            test_loss += loss.item()
            PA0, IOU0, DICE0, P0, R0, F10, tn, fp, fn, tp = zhibiao(output, target)
            PA += PA0
            IOU += IOU0
            DICE += DICE0
            P += P0
            R += R0
            F1 += F10
            if test:
                name = 'Test'
            else:
                name = 'Valid'
            tqrr.set_description(
                "{} Epoch : {} \t {} Loss : {:.6f} \t tn, fp, fn, tp:  {:.0f}  {:.0f}  {:.0f}  {:.0f} ".format(name,
                                                                                                               epoch,
                                                                                                               name,
                                                                                                               loss.item(),
                                                                                                               tn, fp,
                                                                                                               fn, tp))
            if test:
                data_cpu = data.clone().cpu()
                my_output_cpu = output.clone().cpu()
                for i in range(len(data_cpu)):
                    img_tensor = data_cpu[i][0]  # 96 * 96 * 96
                    label_tensor = torch.gt(my_output_cpu[i][1], my_output_cpu[i][0])  # 96 * 96 * 96
                    use_plot_2d(img_tensor, label_tensor, batch_index=batch_index, i=i)
        test_loss /= len(test_loader)
        PA /= len(test_loader)
        IOU /= len(test_loader)
        DICE /= len(test_loader)
        P /= len(test_loader)
        R /= len(test_loader)
        F1 /= len(test_loader)

        print(
            " Epoch : {} \t {} Loss : {:.6f} \t DICE :{:.6f} PA: {:.6f} P: {:.6f} R: {:.6f} IOU: {:.6f} F1: {:.6f}".format(
                epoch, name, test_loss, DICE, PA, P, R, IOU, F1))

        return test_loss, [PA, IOU, DICE, P, R, F1]

class myDataset(Dataset):

    def __init__(self, data_list, label_list):
        self.data_list = data_list
        self.label_list = label_list
    def __getitem__(self, index):
        img = np.load(self.data_list[index])
        label = np.load(self.label_list[index])
        arr = label
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                for k in range(arr.shape[2]):
                    if arr[i][j][k] < 0 or arr[i][j][k] > 1:
                        arr[i][j][k] = 0
        img = np.expand_dims(img, axis=0)
        img = torch.tensor(img)
        img = img.type(torch.FloatTensor)
        label = torch.Tensor(label).long()
        torch.cuda.empty_cache()
        return img, label

    def __len__(self):
        return len(self.data_list)

def zhibiao(data, label):
    n = data.size(0)
    PA, IOU, DICE, P, R, F1, TN, FP, FN, TP = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for i in range(n):

        empty_data = torch.gt(data[i][1], data[i][0])
        empty_data = empty_data.long()  # pred label
        my_data = empty_data
        my_label = label[i]
        my_data = my_data.cpu().numpy()
        my_data = numpy_list(my_data)
        # print("my_data:", my_data)

        my_label = my_label.cpu().numpy()
        my_label = numpy_list(my_label)
        confuse = confusion_matrix(my_label, my_data, labels=[0, 1])
        tn, fp, fn, tp = confusion_matrix(my_label, my_data, labels=[0, 1]).ravel()
        all = tn + fp + fn + tp
        diag = torch.diag(torch.from_numpy(confuse))
        b = 0
        for ii in diag:
            b += ii
        diag = b

        PA += float(torch.true_divide(diag, all))
        # IOU += float(torch.true_divide(diag,(2 * all - diag)))
        # DICE += float(2 * torch.true_divide(diag,2 * all))
        IOU += float(torch.true_divide(tp, tp + fp + fn))
        # DICE += float(2 * torch.true_divide(diag,2 * all))
        DICE += float(torch.true_divide(2 * tp, fp + fn + 2 * tp))
        if tp + fp == 0:
            P += tp / (tp + fp + 1)
        else:
            P += tp / (tp + fp)

        if tp + fn == 0:
            R += tp / (tp + fn + 1)
        else:
            R += tp / (tp + fn)
        TN += tn
        FP += fp
        FN += fn
        TP += tp
    TN /= n
    FP /= n
    FN /= n
    TP /= n

    PA = PA / n
    IOU = IOU / n
    DICE = DICE / n
    P = P / n
    R = R / n
    if P + R == 0:
        F1 += 2 * P * R / (P + R + 1)
    else:
        F1 += 2 * P * R / (P + R)
    return PA, IOU, DICE, P, R, F1, TN, FP, FN, TP


def numpy_list(numpy):
    x = []
    numpy_to_list(x, numpy)
    return x


def numpy_to_list(x, numpy):
    for i in range(len(numpy)):
        if type(numpy[i]) is np.ndarray:
            numpy_to_list(x, numpy[i])
        else:
            x.append(numpy[i])

def show_loss(loss_list, STR, path):
    x1 = range(0, EPOCH)
    y1 = loss_list
    plt.plot(x1, y1, "-", label=STR)
    plt.legend()
    plt.savefig(path + '/%s.jpg' % STR)
    plt.close()

def use_plot_2d(image, output, batch_index=0, i=0, true_label=False):
    for j in range(image.shape[0]):
        p = image[j, :, :] + 0.25
        p = torch.unsqueeze(p, dim=2)

        q = output[j, :, :]
        q = (q * 0.2).float()
        q = torch.unsqueeze(q, dim=2)
        q = p + q

        q[q > 1] = 1
        r = p

        cat_pic = torch.cat([r, q, p], dim=2)
        plt.imshow(cat_pic)
        path = zhibiao_path
            if not os.path.exists(path + fengefu + 'true_pic'):
                os.mkdir(path + fengefu + 'true_pic')
            plt.savefig(path + '/true_pic/%d_%d_%d.jpg' % (batch_index, i, j))
        else:
            if not os.path.exists(path + fengefu + 'pic'):
                os.mkdir(path + fengefu + 'pic')
            plt.savefig(path + '/pic/%d_%d_%d.jpg' % (batch_index, i, j))
        plt.close()
