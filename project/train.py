import os
from torch.autograd import Variable
from torch.optim import Adam, lr_scheduler
from torch.utils.tensorboard import SummaryWriter

from glo import *
from global_annos import *
from train_def import *


from EMC-UNet import UNet
import time
import torch.utils.data
import torch.optim as optim
import torch

EPOCH = 100
BATCH_SIZE = 16
valid_epoch_each = 5
model = UNet()
model = model.to(DEVICE)
train_img_list = []
for i in range(0, 8):
    img_list = os.listdir(cut_img_path + fengefu + "subset%d" % i)
    for ii in img_list:
        train_img_list.append(cut_img_path + fengefu + 'subset%d' % i + fengefu + ii)
train_img_list.sort()
train_label_list = []
for i in range(0, 8):
    label_list = os.listdir(cut_msk_path + fengefu + "subset%d" % i)
    for ii in label_list:
        train_label_list.append(cut_msk_path + fengefu + 'subset%d' % i + fengefu + ii)
train_label_list.sort()
dataset_train = myDataset(train_img_list, train_label_list)
train_loader = torch.utils.data.DataLoader(dataset_train,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=16)
print("train_dataloader_ok")

valid_img_list = []
for i in range(8, 9):
    img_list = os.listdir(cut_img_path + fengefu + "subset%d" % i)
    for ii in img_list:
        valid_img_list.append(cut_img_path + fengefu + 'subset%d' % i + fengefu + ii)
valid_img_list.sort()
valid_label_list = []
for i in range(8, 9):
    label_list = os.listdir(cut_msk_path + fengefu + "subset%d" % i)
    for ii in label_list:
        valid_label_list.append(cut_msk_path + fengefu + 'subset%d' % i + fengefu + ii)
valid_label_list.sort()
dataset_valid = myDataset(valid_img_list, valid_label_list)
valid_loader = torch.utils.data.DataLoader(dataset_valid,
                                           batch_size=BATCH_SIZE,
                                           shuffle=False,
                                           num_workers=16)
print("valid_dataloader_ok")

test_img_list = []
for i in range(9, 10):
    img_list = os.listdir(cut_img_path + fengefu + "subset%d" % i)
    for ii in img_list:
        test_img_list.append(cut_img_path + fengefu + 'subset%d' % i + fengefu + ii)
test_img_list.sort()
test_label_list = []
for i in range(9, 10):
    label_list = os.listdir(cut_msk_path + fengefu + "subset%d" % i)
    for ii in label_list:
        test_label_list.append(cut_msk_path + fengefu + 'subset%d' % i + fengefu + ii)
test_label_list.sort()
dataset_test = myDataset(test_img_list, test_label_list)
test_loader = torch.utils.data.DataLoader(dataset_test,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False,
                                          num_workers=16)
print("Test_dataloader_ok")

minnum = 0
mome = 0.99
lr = 1e-3

start = time.perf_counter()
train_loss_list = []
valid_loss_list = []
total_train_step = 0
total_valid_step = 0
train_loss_item = 0.0
writer = SummaryWriter("./logs_loss")
for epoch in range(1, EPOCH + 1):
    print("-------the {} epoch--------".format(epoch))
    if epoch == 90:
        mome = 0.9
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = lr_scheduler.MultiStepLR(optimizer, [50, 70, 90], gamma=0.1, last_epoch=-1)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1, last_epoch=-1)
    train_loss = train_model(model, DEVICE, train_loader, optimizer, epoch)
    train_loss_item = train_loss[0]
    total_train_step = total_train_step + 1
    writer.add_scalar("train_loss", train_loss_item, total_train_step)
    train_loss_list.append(train_loss)
    train_loss_pd = pd.DataFrame(train_loss_list)
    train_loss_pd.to_excel(zhibiao_path + "/the%d epoch_trainloss.xlsx" % epoch, engine='openpyxl')
    torch.save(model, model_path + fengefu + 'train_model.pth')
    torch.cuda.empty_cache()
    if epoch % valid_epoch_each == 0:
        valid_loss, valid_zhibiao = tst_model(model, DEVICE, valid_loader, epoch, test=False)
        dice1 = valid_zhibiao[2]
        total_valid_step = total_valid_step + 1
        writer.add_scalar("valid_loss", valid_loss, total_valid_step)
        valid_loss_list.append(valid_loss)
        valid_loss_pd = pd.DataFrame(valid_loss_list)
        valid_loss_pd.to_excel(zhibiao_path + "/the%d epoch val_loss.xls" % epoch, engine='openpyxl')
        if epoch == valid_epoch_each:
            torch.save(model, model_path + fengefu + 'best_model.pth')
            minnum = valid_loss
            print("minnum", minnum)
        elif valid_loss < minnum:
            print("valid_loss < minnum", valid_loss, "<", minnum)
            minnum = valid_loss
            torch.save(model, model_path + fengefu + 'best_model.pth')
            zhibiao = valid_zhibiao
            zhibiao_pd = pd.DataFrame(zhibiao)
            zhibiao_pd.to_excel(
                zhibiao_path + "/best：[PA, IOU, DICE, P, R, F1].xls" % epoch,
                engine='openpyxl')
        else:
            pass
writer.close()

end = time.perf_counter()
train_time = end - start
print('Running time: %s Seconds' % train_time
time_list = list([train_time])
train_time_pd = pd.DataFrame(time_list)
train_time_pd.to_excel(zhibiao_path + "/TIME.xls", engine='openpyxl')
test_start = time.perf_counter()
# torch.cuda.empty_cache()

test_loss_list = []
test_zhibiao_list = []

model = torch.load(model_path + fengefu + 'best_model.pth')
model = model.to(DEVICE)

test_loss, test_zhibiao = tst_model(model, DEVICE, test_loader, EPOCH, test=True)

test_loss_list.append(test_loss)
test_zhibiao_list.append(test_zhibiao)

test_loss_pd = pd.DataFrame(test_loss_list)
test_loss_pd.to_excel(zhibiao_path + "/TESTLOSS.xls")
test_zhibiao_pd = pd.DataFrame(test_zhibiao_list)  # 存成excel格式
test_zhibiao_pd.to_excel(zhibiao_path + "/TEST[PA, IOU, DICE, P, R, F1].xls")

test_end = time.perf_counter()
test_time = test_end - test_start
print('Running time: %s Seconds' % test_time)
test_time_list = list([test_time])
test_time_pd = pd.DataFrame(test_time_list)
test_time_pd.to_excel(zhibiao_path + "/TESTTIME.xls", engine='openpyxl')
