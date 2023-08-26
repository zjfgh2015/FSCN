import argparse
import math
import random
from copy import deepcopy
import os

from plot import plot_multiclass_roc

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
# os.environ["HF_DATASETS_OFFLINE"] = '1'
# os.environ["TRANSFORMERS_OFFLINE"] = '1'
import torch
import numpy as np
import torch.nn as nn
from data_loader import grab_all_data
from model import Model
from transformers import AutoTokenizer
from tqdm import tqdm
import torch.nn.functional as F
from sklearn.metrics import classification_report
from balanced_loss import Loss

device = torch.device('cuda' if torch.cuda.is_available() else "cpu")


def set_seed(seed):
    # np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


set_seed(8888)



def main(cfg):
    model = Model()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=cfg.lr, weight_decay=0.01)

    model.to(device)
    checkpoint = 'hfl/chinese-bert-wwm'
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    data = grab_all_data(cfg)
    random.shuffle(data)

    samples_per_class = [0, 0, 0]
    for item in data[:-200]:
        samples_per_class[item['label'] + 1] += 1
    criterion = Loss(
        loss_type='cross_entropy',
        samples_per_class=samples_per_class,
        class_balanced=True
    )

    for epoch in range(cfg.epochs):
        loss_epoch = train(cfg, model, data[:-200], criterion, optimizer, tokenizer, epoch)
        print("Loss epoch {}".format(loss_epoch))
        acc = eval(cfg, model, data[-200:], criterion, tokenizer, epoch)
        print("Acc epoch {}".format(acc))


def cal_acc(probs, labels):
    index = torch.argmax(probs, dim=1)
    right_count = 0
    for i in range(len(index)):
        if index[i] == labels[i]:
            right_count += 1
    return right_count / len(index)


def train(cfg, model, train_data, criterion, optimizer, tokenizer, epoch):
    model.train()
    train_loader = tqdm(range(0, int(len(train_data) / cfg.batch_size)), desc="Epoch [{0}] Training".format(epoch))
    loss_epoch = []
    for batch_index in train_loader:
        batch_data = train_data[batch_index * cfg.batch_size:(batch_index + 1) * cfg.batch_size]
        batch_label = []
        batch_data_copy = []
        for i, item in enumerate(batch_data):
            try:
                item['content_new_token'] = tokenizer(item['content_new'], padding='longest',
                                                      add_special_tokens=True,
                                                      return_tensors="pt").to(device)
                item['content_hot_token'] = tokenizer(item['content_hot'], padding='longest',
                                                      add_special_tokens=True,
                                                      return_tensors="pt").to(device)
                batch_label.append(batch_data[i]['label'] + 1)
                batch_data_copy.append(item)
            except Exception:
                continue

        probs = model(batch_data_copy)
        probs_numpy = probs.detach().cpu().numpy()
        optimizer.zero_grad()
        loss = criterion(probs, torch.tensor(batch_label).to(device))
        loss.backward()
        loss_epoch.append(loss.item())
        optimizer.step()
    return np.mean(loss_epoch)


def eval(cfg, model, test_data, criterion, tokenizer, epoch):
    model.eval()
    test_loader = tqdm(range(0, math.ceil(len(test_data) / cfg.batch_size)), desc="Epoch [{0}] Testing".format(epoch))
    all_label = []
    with torch.no_grad():
        for batch_index in test_loader:
            batch_data = test_data[batch_index * cfg.batch_size:(batch_index + 1) * cfg.batch_size]

            batch_data_copy = []
            for i, item in enumerate(batch_data):
                try:
                    item['content_new_token'] = tokenizer(item['content_new'], padding='longest',
                                                          add_special_tokens=True,
                                                          return_tensors="pt").to(device)
                    item['content_hot_token'] = tokenizer(item['content_hot'], padding='longest',

                                                          add_special_tokens=True,
                                                          return_tensors="pt").to(device)
                    all_label.append(batch_data[i]['label'] + 1)
                    batch_data_copy.append(item)
                except Exception:
                    continue

            probs = model(batch_data_copy)
            if batch_index == 0:
                probs_all = probs

            else:
                probs_all = torch.cat((probs_all, probs), dim=0)
    probs_numpy = probs_all.detach().cpu().numpy()

    # plot_multiclass_roc(all_label, probs_numpy)

    print(classification_report(all_label, np.argmax(probs_numpy, axis=1)))

    acc = cal_acc(probs_all, labels=torch.tensor(all_label).to(device))

    # loss = criterion(probs, torch.tensor(batch_label).to(device))
    # loss.backward()
    # loss_epoch.append(loss.item())
    # optimizer.step()
    return np.mean(acc)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', default=50, type=int)
    parser.add_argument("--aug_num", default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--lr', default=0.0001, type=float)

    cfg = parser.parse_args()
    main(cfg)
