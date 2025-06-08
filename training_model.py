import time
import os
from tqdm import tqdm

import pandas as pd
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
%matplotlib inline


import warnings
warnings.filterwarnings("ignore")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device', device)

from torchvision import transforms


train_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                     ])


test_transform = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize(
                                         mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
                                    ])

dataset_dir = 'new'

train_path = os.path.join(dataset_dir, 'train')
test_path = os.path.join(dataset_dir, 'valid')
print('trainpath', train_path)
print('testpath', test_path)

from torchvision import datasets

train_dataset = datasets.ImageFolder(train_path, train_transform)

test_dataset = datasets.ImageFolder(test_path, test_transform)


class_names = train_dataset.classes
n_class = len(class_names)

train_dataset.class_to_idx

idx_to_labels = {y:x for x,y in train_dataset.class_to_idx.items()}

idx_to_labels

np.save('idx_to_labels.npy', idx_to_labels)
np.save('labels_to_idx.npy', train_dataset.class_to_idx)


from torch.utils.data import DataLoader

BATCH_SIZE = 32


train_loader = DataLoader(train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=4
                         )


test_loader = DataLoader(test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False,
                         num_workers=4
                        )

from torchvision import models
import torch.optim as optim
from torch.optim import lr_scheduler

model = models.resnet50(pretrained=False)
optimizer = optim.Adam(model.parameters())
model = model.to(device)
criterion = nn.CrossEntropyLoss()
EPOCHS = 100
lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


def train_one_batch(images, labels):
    images = images.to(device)
    labels = labels.to(device)
    outputs = model(images)
    loss = criterion(outputs, labels)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    _, preds = torch.max(outputs, 1)
    preds = preds.cpu().numpy()
    loss = loss.detach().cpu().numpy()
    outputs = outputs.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    log_train = {}
    log_train['epoch'] = epoch
    log_train['batch'] = batch_idx
    log_train['train_loss'] = loss
    log_train['train_accuracy'] = accuracy_score(labels, preds)
    return log_train


def evaluate_testset():

    loss_list = []
    labels_list = []
    preds_list = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            loss = criterion(outputs, labels)
            loss = loss.detach().cpu().numpy()
            outputs = outputs.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            loss_list.append(loss)
            labels_list.extend(labels)
            preds_list.extend(preds)
    log_test = {}
    log_test['epoch'] = epoch
    log_test['test_loss'] = np.mean(loss_list)
    log_test['test_accuracy'] = accuracy_score(labels_list, preds_list)
    log_test['test_precision'] = precision_score(labels_list, preds_list, average='macro')
    log_test['test_recall'] = recall_score(labels_list, preds_list, average='macro')
    log_test['test_f1-score'] = f1_score(labels_list, preds_list, average='macro')
    return log_test

epoch = 0
batch_idx = 0
best_test_accuracy = 0
df_train_log = pd.DataFrame()
log_train = {}
log_train['epoch'] = 0
log_train['batch'] = 0
images, labels = next(iter(train_loader))
log_train.update(train_one_batch(images, labels))
df_train_log = df_train_log.append(log_train, ignore_index=True)
df_test_log = pd.DataFrame()
log_test = {}
log_test['epoch'] = 0
log_test.update(evaluate_testset())
df_test_log = df_test_log.append(log_test, ignore_index=True)

for epoch in range(1, EPOCHS + 1):

    print(f'Epoch {epoch}/{EPOCHS}')


    model.train()
    for images, labels in tqdm(train_loader):
        batch_idx += 1
        log_train = train_one_batch(images, labels)
        df_train_log = df_train_log.append(log_train, ignore_index=True)

    lr_scheduler.step()

    model.eval()
    log_test = evaluate_testset()
    df_test_log = df_test_log.append(log_test, ignore_index=True)

    if log_test['test_accuracy'] > best_test_accuracy:
        old_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(best_test_accuracy)
        if os.path.exists(old_best_checkpoint_path):
            os.remove(old_best_checkpoint_path)
        best_test_accuracy = log_test['test_accuracy']
        new_best_checkpoint_path = 'checkpoint/best-{:.3f}.pth'.format(log_test['test_accuracy'])
        torch.save(model, new_best_checkpoint_path)


df_train_log.to_csv('trainnote.csv', index=False)
df_test_log.to_csv('testnote.csv', index=False)