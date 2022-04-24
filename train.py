import datetime
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dgl.data import CoraGraphDataset
from dgl.dataloading import GraphDataLoader
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from config import Parameters
from dataset import GCNDataset
from models import Classifier, GCNClassifier, NodeClassifier, GATClassifier


def _train(dataset, model, criterion, optim, batch_size, epochs, device):

    train_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model = model.to(device)
    criterion = criterion
    optim = optim

    min_loss = 1e9

    model.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        total = 0
        correct = 0
        for batched_graph, label in train_dataloader:
            optim.zero_grad()

            feats = batched_graph.ndata['feat'].to(device)
            outputs = model(batched_graph, feats)
            label = label.to(device)

            loss = criterion(outputs, label)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

            loss.backward()
            optim.step()

        print('epoch    :{}'.format(epoch+1))
        print('loss     :{}'.format(running_loss))
        print('accuracy :{} %'.format(correct/total*100))

        if min_loss >= running_loss:
            model_path = './data/weight'
            torch.save(model.state_dict(), model_path)

def train(dataset, model, criterion, optim, batch_size, epochs, device, pred_rank, out_dir,n_splits=5):
    kf = KFold(n_splits=n_splits)
    model = model.to(device)
    criterion = criterion
    optim = optim

    min_loss = 1e9

    for epoch in tqdm(range(epochs)):
        train_loss = 0
        valid_loss = 0
        train_total = 0
        train_correct = 0
        valid_total = 0
        valid_correct = 0
        
        for _fold, (train_index, valid_index) in enumerate(kf.split(dataset)):
            train_dataset = Subset(dataset, train_index).dataset
            train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            valid_dataset   = Subset(dataset, valid_index).dataset
            valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
            
            model.train()
            for batched_graph, label in train_dataloader:
                optim.zero_grad()

                batched_graph = batched_graph.to(device)
                feats = batched_graph.ndata['feat'].to(device)
                outputs = model(batched_graph, feats)
                label = label.to(device)

                loss = criterion(outputs, label)

                train_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)

                train_total += label.size(0)
                train_correct += (predicted == label).sum().item()

                loss.backward()
                optim.step()

            model.eval()
            with torch.no_grad():
                for batched_graph, label in valid_dataloader:

                    batched_graph = batched_graph.to(device)
                    feats = batched_graph.ndata['feat'].to(device)
                    outputs = model(batched_graph, feats)
                    label = label.to(device)

                    loss = criterion(outputs, label)

                    valid_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)

                    valid_total += label.size(0)
                    valid_correct += (predicted == label).sum().item()

        print('\n')
        print('train loss     :{}'.format(train_loss/kf.n_splits))
        print('train accuracy :{} %'.format(train_correct/train_total*100))
        print('valid loss     :{}'.format(valid_loss/kf.n_splits))
        print('valid accuracy :{} %'.format(valid_correct/valid_total*100))

        if min_loss >= valid_loss:
            model_path = f'{out_dir}/weight_rank_{pred_rank}'
            torch.save(model.state_dict(), model_path)
    model_path = f'{out_dir}/weight_rank_{pred_rank}_end'
    torch.save(model.state_dict(), model_path)

def predict_race(dataset, model_dir, weight, device='cpu'):
    model = GCNClassifier(pool_type='attention')
    model.load_state_dict(torch.load(f'data/{model_dir}/{weight}', map_location=torch.device('cpu')))

    model.eval()
    with torch.no_grad():
        feats = dataset.ndata['feat'].to(device)
        outputs = model(dataset, feats)

        _, predicted = torch.max(outputs.data, 1)

        prob = outputs.data[0]

        sort_pred = list(np.sort(prob))[::-1]
        sort_index = list(np.argsort(prob))[::-1]

        for pred, index in zip(sort_pred[:5], sort_index[:5]):
            print(f'{index+1}番 : {pred}')


def eval(dataset, model, pred_rank, out_dir,device='cpu'):
    model = model.to(device)
    model.load_state_dict(torch.load(f'{out_dir}/weight_rank_{pred_rank}', map_location=torch.device(device)))

    dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)
    model.eval()
    with torch.no_grad():
        total = 0
        true_correct = 0
        third_correct = 0
        five_correct = 0
        for graph, label in dataloader:
            feats = graph.ndata['feat'].to(device)
            outputs = model(graph, feats)

            label = label.to(device)

            prob = outputs.data[0]

            sort_pred = list(np.sort(prob))[::-1]
            sort_index = list(np.argsort(prob))[::-1]

            _, predicted = torch.max(outputs.data, 1)

            total += label.size(0)

            if label.item() == sort_index[0]:
                true_correct += 1
            if label.item() in sort_index[:3]:
                third_correct += 1
            if label.item() in sort_index[:5]:
                five_correct += 1
        print('正しく一位を予測できた確率 : {}%'.format(true_correct/total*100))
        print('一位が予測上位3頭以内の確率: {}%'.format(third_correct/total*100))
        print('一位が予測上位5頭以内の確率: {}%'.format(five_correct/total*100))


def node_train(dataset, model, criterion, optim, batch_size, epochs, device):
    train_dataloader = GraphDataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model = model.to(device)
    criterion = criterion
    optim = optim

    min_loss = 1e9

    model.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        total = 0
        correct = 0
        for batched_graph in train_dataloader:
            optim.zero_grad()

            feats = batched_graph.ndata['feat'].to(device)
            labels = batched_graph.ndata['label'].to(device)
            outputs = model(batched_graph, feats)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss.backward()
            optim.step()

        print('epoch :{}'.format(epoch+1))
        print('loss     :{}'.format(running_loss))
        print('accuracy :{} %'.format(correct/total*100))

        if min_loss >= running_loss:
            model_path = './data/weight'
            torch.save(model.state_dict(), model_path)

def node_eval(dataset, device='cpu'):
    model = GCNClassifier()
    model.load_state_dict(torch.load('data/weight', map_location=torch.device('cpu')))

    dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    with torch.no_grad():
        total = 0
        correct = 0
        for batched_graph in dataloader:
            optim.zero_grad()

            feats = batched_graph.ndata['feat'].to(device)
            labels = batched_graph.ndata['label'].to(device)
            outputs = model(batched_graph, feats)

            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print('eval accuracy: {}%'.format(correct/total*100))

def nn_train(dataset, model, criterion, optim, batch_size, epochs, device):
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True, shuffle=True)

    model = model.to(device)
    criterion = criterion
    optim = optim

    min_loss = 1e9

    model.train()

    for epoch in tqdm(range(epochs)):
        running_loss = 0
        total = 0
        correct = 0
        for data, label in dataloader:
            optim.zero_grad()

            inputs = data.to(device)
            label = label.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, label)

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)

            total += label.size(0)
            correct += (predicted == label).sum().item()

            loss.backward()
            optim.step()

        print('epoch    :{}'.format(epoch+1))
        print('loss     :{}'.format(running_loss))
        print('accuracy :{} %'.format(correct/total*100))

        if min_loss >= running_loss:
            model_path = './data/weight'
            torch.save(model.state_dict(), model_path)

def main(params, args):
    pred_rank = args.pred_rank
    if args.out_dir:
        out_dir = f'data/{args.out_dir}'
        os.makedirs(out_dir, exist_ok=True)
    else:
        now = datetime.datetime.now()
        out_dir = f"data/{now.strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(out_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = 'cuda:1'
    else:
        device = 'cpu'

    train_dataset = GCNDataset(device, pred_rank=pred_rank,nn_type='graph', csv_path='train_dataset.csv')
    test_dataset = GCNDataset(device='cpu', pred_rank=pred_rank, nn_type='graph', csv_path='test_dataset.csv')

    model = GCNClassifier(pool_type='attention')
    # NUM_LAYERS = 3
    # NUM_HEADS = 3
    # heads = [NUM_HEADS for _ in range(NUM_LAYERS)]
    # model = GATClassifier(num_heads=heads, pool_type='attention')
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    train(train_dataset, model, criterion, optimizer, batch_size=256, epochs=50, device=device, pred_rank=pred_rank, out_dir=out_dir)
    eval(test_dataset, model, pred_rank=pred_rank, out_dir=out_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--pred_rank', '-r', default=1)
    parser.add_argument('--out_dir', '-o', default=None)

    args = parser.parse_args()
    params = Parameters(ARGS=args)

    main(params, args)
