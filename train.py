import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dgl.data import CoraGraphDataset
from dgl.dataloading import GraphDataLoader

from tqdm import tqdm

from models import GCNClassifier, NodeClassifier, Classifier
from dataset import GCNDataset

def train(dataset, model, criterion, optim, batch_size, epochs, device):

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
            print(outputs)

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


def main():

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    dataset = GCNDataset(nn_type='nn')

    model = Classifier(in_feats=49)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    nn_train(dataset, model, criterion, optimizer, batch_size=64, epochs=300, device=device)

if __name__ == '__main__':
    main()