import argparse
import numpy as np
import torch
from model import TextRCNN
from dataset import WeiboData
from torch.utils.data import DataLoader
import torch.nn as nn
import time
from utils import set_logger
import logging

SEED = 1234
np.random.seed(SEED)
torch.random.manual_seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluation(model, test_dataloader, criterion, n_test, test_steps):
    model.eval()
    test_loss = 0.
    correct = 0
    count = 0
    for item_seq, target in test_dataloader:
        item_seq, target = item_seq.to(device), target.squeeze().to(device)
        y_pred = model(item_seq)
        loss = criterion(y_pred, target)
        test_loss += loss.item()
        correct += (y_pred.argmax(dim=1) == target).sum().item()
        if count == 5:
            break

    test_loss /= test_steps
    test_acc = correct / n_test

    return test_acc, test_loss


def train(config):

    train_dataset = WeiboData(config.train_data, config.voc_path)
    test_dataset = WeiboData(config.test_data, config.voc_path)

    max_len = max(train_dataset.max_len, test_dataset.max_len)
    config.max_len = max_len
    train_dataset.max_len = max_len
    test_dataset.max_len = max_len

    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config.batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config.batch_size)

    batch_size = config.batch_size
    n_train, n_test = len(train_dataset), len(test_dataset)
    train_steps = int(n_train / batch_size) + 1
    test_steps = int(n_test / batch_size) + 1

    logging.info(f'train size: {n_train} test size: {n_test}')

    model = TextRCNN(config).to(device)
    CELoss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_acc = 0.
    best_epoch = 0
    count = 0
    early_stopping = config.es_epoch

    for epoch in range(config.num_epoch):
        c1 = time.time()
        epoch_loss = 0.

        model.train()
        for item_seq, label in train_dataloader:
            item_seq, label = item_seq.to(device), label.squeeze().to(device)
            y_pred = model(item_seq)
            loss = CELoss(y_pred, label)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_loss /= train_steps
        c2 = time.time()
        logging.info('Epoch:%d [%.1fs] train loss: %.4f' % (epoch + 1, c2 - c1, epoch_loss))

        test_acc, test_loss = evaluation(model, test_dataloader, CELoss, n_test, test_steps)

        logging.info('test acc: %.4f test loss: %.4f' % (test_acc, test_loss))
        logging.info('Evaluation time: %.1fs' % (time.time() - c2))

        count += 1
        if test_acc <= best_acc and count == early_stopping:
            logging.info('Early stopping is trigger at epoch: %d best acc:%.4f' % (best_epoch + 1, best_acc))
            break
        if test_acc > best_acc:
            count = 0
            best_acc = test_acc
            best_epoch = epoch


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Hyper Parameters')
    # Data
    parser.add_argument('--train_data', default='data/train_data_0.2.txt', type=str,
                        help='total meta train data path')
    parser.add_argument('--test_data', default='data/test_data_0.2.txt', type=str,
                        help='total meta test data path')
    parser.add_argument('--voc_path', default='data/voc_dict_0.2.txt', type=str,
                        help='vocabulary dict path')
    parser.add_argument('--num_classes', default=2, type=int,
                        help='class number')
    # Model
    parser.add_argument('--embed_size', default=128, type=int,
                        help='embedding size')
    parser.add_argument('--hidden_size', default=128, type=int,
                        help='hidden layer embedding size')
    parser.add_argument('--num_layers', default=3, type=int,
                        help='lstm layer number')
    parser.add_argument('--voc_size', default=1002, type=int,
                        help='vocabulary dict size')
    parser.add_argument('--max_len', default=None, type=int,
                        help="Do not manually set")
    parser.add_argument('--dropout', default=0.8, type=float,
                        help='lstm drop out rate')
    # Experiment
    parser.add_argument('--model_name', default='TextRCNN', type=str,
                        help='running model name')
    parser.add_argument('--run_time', default=0, type=int,
                        help='number of model runs')
    parser.add_argument('--data_ratio', default=0.2, type=float,
                        help='proportion of data used in all data')
    parser.add_argument('--log_save', default='log', type=str,
                        help='log save directory')
    parser.add_argument('--batch_size', default=256, type=int,
                        help='batch size')
    parser.add_argument('--num_epoch', default=100, type=int,
                        help='max epoch number')
    parser.add_argument('--lr', default=1.e-3, type=float,
                        help='learning rate')
    parser.add_argument('--device', default=device, type=str,
                        help='training on gpu or cpu')
    parser.add_argument('--es_epoch', default=5, type=int,
                        help='early stopping epoch num')
    config = parser.parse_args()
    set_logger(config)

    train(config)
