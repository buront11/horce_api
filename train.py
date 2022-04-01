import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from dgl.data import CoraGraphDataset
from dgl.dataloading import GraphDataLoader

from tqdm import tqdm
from sklearn.model_selection import train_test_split, KFold

from models import GCNClassifier, NodeClassifier, Classifier
from dataset import GCNDataset

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

def train(dataset, model, criterion, optim, batch_size, epochs, device, n_splits=5):
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
            model_path = './data/weight'
            torch.save(model.state_dict(), model_path)

def predict_race(dataset, device='cpu'):
    model = GCNClassifier()
    model.load_state_dict(torch.load('data/weight', map_location=torch.device('cpu')))

    with torch.no_grad():
        feats = dataset.ndata['feat'].to(device)
        outputs = model(dataset, feats)

        print(outputs)

        _, predicted = torch.max(outputs.data, 1)
        print(predicted)

def eval(dataset, device='cpu'):
    model = GCNClassifier()
    model.load_state_dict(torch.load('data/weight', map_location=torch.device('cpu')))

    dataloader = GraphDataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    with torch.no_grad():
        total = 0
        correct = 0
        for graph, label in dataloader:
            feats = graph.ndata['feat'].to(device)
            outputs = model(graph, feats)

            label = label.to(device)

            _, predicted = torch.max(outputs.data, 1)
            
            total += label.size(0)
            correct += (predicted == label).sum().item()
        print('accuracy: {}%'.format(correct/total*100))


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

    device = 'cpu'

    train_dataset = GCNDataset(nn_type='graph', csv_path='train_dataset.csv')
    test_dataset = GCNDataset(nn_type='graph', csv_path='test_dataset.csv')

    model = GCNClassifier()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    train(train_dataset, model, criterion, optimizer, batch_size=32, epochs=100, device=device)
    eval(test_dataset, device)

if __name__ == '__main__':
    main()