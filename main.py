import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.io as scio
import numpy as np
import random
import torch.utils.data as data
import os
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score,recall_score
from cross_verity import EcgDataset, TrainDataSet, TestDataSet
from sklearn.model_selection import StratifiedKFold
import csv


# In this section, we will apply an CNN to extract features and implement a classification task.
# Firstly, we should build the model by PyTorch. We provide a baseline model here.
# You can use your own model for better performance
class Doubleconv_33(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_33, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, padding=1, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_35(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_35, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, padding=2, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, padding=2, kernel_size=5),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Doubleconv_37(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Doubleconv_37, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, padding=3, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, padding=3, kernel_size=7),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Tripleconv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Tripleconv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch_in, ch_out, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, padding=1, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.Conv1d(ch_out, ch_out, padding=1, kernel_size=3),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class MLP(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(ch_in, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, ch_out),
        )

    def forward(self, input):
        return self.fc(input)


class Mscnn(nn.Module):
    # TODO: Build a better model
    def __init__(self, ch_in, ch_out):
        super(Mscnn, self).__init__()
        # stream1
        self.conv11 = Doubleconv_33(ch_in, 64)
        self.pool11 = nn.MaxPool1d(3, stride=3)
        self.conv12 = Doubleconv_33(64, 128)
        self.pool12 = nn.MaxPool1d(3, stride=3)
        self.conv13 = Tripleconv(128, 256)
        self.pool13 = nn.MaxPool1d(2, stride=2)
        self.conv14 = Tripleconv(256, 512)
        self.pool14 = nn.MaxPool1d(2, stride=2)
        self.conv15 = Tripleconv(512, 512)
        self.pool15 = nn.MaxPool1d(2, stride=2)
        # stream2
        self.conv21 = Doubleconv_37(ch_in, 64)
        self.pool21 = nn.MaxPool1d(3, stride=3)
        self.conv22 = Doubleconv_37(64, 128)
        self.pool22 = nn.MaxPool1d(3, stride=3)
        self.conv23 = Tripleconv(128, 256)
        self.pool23 = nn.MaxPool1d(2, stride=2)
        self.conv24 = Tripleconv(256, 512)
        self.pool24 = nn.MaxPool1d(2, stride=2)
        self.conv25 = Tripleconv(512, 512)
        self.pool25 = nn.MaxPool1d(2, stride=2)

        self.out = MLP(1024 * 33, ch_out)

    def forward(self, x):
        # stream1
        c11 = self.conv11(x)
        p11 = self.pool11(c11)
        c12 = self.conv12(p11)
        p12 = self.pool12(c12)
        c13 = self.conv13(p12)
        p13 = self.pool13(c13)
        c14 = self.conv14(p13)
        p14 = self.pool14(c14)
        c15 = self.conv15(p14)
        p15 = self.pool15(c15)
        # stream2
        c21 = self.conv21(x)
        p21 = self.pool21(c21)
        c22 = self.conv22(p21)
        p22 = self.pool22(c22)
        c23 = self.conv23(p22)
        p23 = self.pool23(c23)
        c24 = self.conv24(p23)
        p24 = self.pool24(c24)
        c25 = self.conv25(p24)
        p25 = self.pool25(c25)
        concatenation = torch.cat([p15, p25], dim=1)
        merge = concatenation.view(concatenation.size()[0], -1)
        output = self.out(merge)
        output = F.sigmoid(output)
        return output


# Now, we will build the pipeline for deep learning based training.
# These functions may be useful :)
def save_loss(fold, value):
    path = 'loss' + str(fold) + '.txt'
    file = open(path, mode='a+')
    file.write(str(value) + '\n')


if __name__ == "__main__":
    # We will use GPU if cuda is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build pre-processing transformation
    # Note this pre-processing is in PyTorch
    x_transforms = transforms.Compose([
        transforms.ToTensor(),
    ])
    y_transforms = transforms.ToTensor()

    # TODO: fine tune hyper-parameters
    batch_size = 64
    csv_reader = csv.reader(
        open("C:/Users/13120/Desktop/现代信号处理-pro2-fall2022/现代信号处理-pro2-fall2022/data/train_data.csv"))
    train_dir = 'C:/Users/13120/Desktop/现代信号处理-pro2-fall2022/现代信号处理-pro2-fall2022/data/train/'
    filenamelist = []
    labellist = []
    # Read csv file by rows
    for row in csv_reader:
        filename = row[0]
        label_name = row[1]
        label = 0
        # Convert label from str to int
        if label_name == 'N':
            label = 1
        elif label_name == 'A':
            label = 0
        else:
            continue
        absfilename = train_dir + filename + ".mat"
        filenamelist.append(absfilename)
        labellist.append(label)
    skf = StratifiedKFold(n_splits=5, random_state=2, shuffle=True)
    train_work_cnt = 0
    for train_index, test_index in skf.split(filenamelist, labellist):
        # print("TRAIN:", train_index, "TEST:", test_index)
        trainfilenamelist, testfilenamelist = np.array(filenamelist)[train_index], np.array(filenamelist)[test_index]
        trainlabellist, testlabellist = np.array(labellist)[train_index], np.array(labellist)[test_index]
        train_data_set = TrainDataSet(trainfilenamelist, 2400, transform=x_transforms, target_transform=y_transforms)
        dataloader = DataLoader(train_data_set, batch_size=batch_size, shuffle=True, num_workers=0)
        test_data_set = TestDataSet(testfilenamelist, 2400, transform=x_transforms, target_transform=y_transforms)
        testdataloader = DataLoader(test_data_set, batch_size=1)
        model = Mscnn(1, 2).to(device)  # ch_in, ch_out
        train_weight = torch.Tensor([5,1]).cuda()
        criterion = torch.nn.CrossEntropyLoss(train_weight)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        num_epochs = 100
        # Start training !
        for epoch in range(1, num_epochs + 1):
            model.train()
            print('Epoch {}/{}'.format(epoch, num_epochs))
            # Write your code here
            # dt_size = len(dataloader.dataset)
            #     print(dt_size)
            epoch_loss = 0
            step = 0
            process = tqdm(dataloader)
            for x, y in process:
                step += 1
                inputs = x.to(device)
                labels = y.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
                process.set_description(
                    "epoch: %d, train_loss:%0.8f" % (epoch, epoch_loss / step)
                )
            if epoch % 5 == 0:
                # Set model's mode lto eval
                model.eval()

                # TODO: add more metrics for evaluation?
                # Evaluate
                predict = []
                target = []
                with torch.no_grad():
                    for x, mask in testdataloader:
                        y = model(x.to(device))
                        y = torch.argmax(y,dim=1)
                        predict.append(torch.squeeze(y).cpu().numpy())
                        target.append(torch.squeeze(torch.argmax(mask)).cpu().numpy())
                acc = accuracy_score(target, predict)
                auc = roc_auc_score(target, predict)
                pre=precision_score(target,predict)
                rec=recall_score(target,predict)
                # print('Auc: {}'.format(auc))
                # print('Accuracy: {}'.format(acc))
                print('Pre:{}'.format(pre))
                print('Rec:{}'.format(rec))
                if auc > 0.7:
                    model_name = "Auc_".format(auc)+"Acc_".format(acc)+"epoch_".format(epoch)+"step_".format(train_work_cnt)+".pth"
                    torch.save(model.state_dict(), model_name)

            epoch_loss /= step

            save_loss(100, epoch_loss)
        # Save model
        # if
        # torch.save(model.state_dict(), 'weights10_%d.pth' % (epoch))
        # Build test dataset
        train_work_cnt = train_work_cnt + 1
