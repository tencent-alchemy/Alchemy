#!/usr/bin/env python
# encoding: utf-8
# File Name: mpnn.py
# Author: Jiezhong Qiu
# Create Time: 2019/04/23 17:38
# TODO:

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

import pandas as pd
import os

class Complete(object):
    def __call__(self, data):
        device = data.edge_index.device

        row = torch.arange(data.num_nodes, dtype=torch.long, device=device)
        col = torch.arange(data.num_nodes, dtype=torch.long, device=device)

        row = row.view(-1, 1).repeat(1, data.num_nodes).view(-1)
        col = col.repeat(data.num_nodes)
        edge_index = torch.stack([row, col], dim=0)

        edge_attr = None
        if data.edge_attr is not None:
            idx = data.edge_index[0] * data.num_nodes + data.edge_index[1]
            size = list(data.edge_attr.size())
            size[0] = data.num_nodes * data.num_nodes
            edge_attr = data.edge_attr.new_zeros(size)
            edge_attr[idx] = data.edge_attr

        edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
        data.edge_attr = edge_attr
        data.edge_index = edge_index

        return data


transform = T.Compose([Complete(), T.Distance(norm=False)])
train_dataset = TencentAlchemyDataset(root='data-bin', mode='dev', transform=transform).shuffle()
test_dataset = TencentAlchemyDataset(root='data-bin', mode='test', transform=transform)
valid_dataset = TencentAlchemyDataset(root='data-bin', mode='valid', transform=transform)


valid_loader = DataLoader(valid_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class MPNN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=12,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6):
        super(MPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim)
                )
        nego_edge_network = nn.Sequential(
                nn.Linear(node_hidden_dim * node_hidden_dim, edge_hidden_dim),
                nn.ReLU(),
                nn.Linear(edge_hidden_dim, edge_input_dim)
                )
        self.conv = NNConv(node_hidden_dim, node_hidden_dim, edge_network, nego_edge_network, aggr='mean', root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)
        fg_loss_all = 0

        for i in range(self.num_step_message_passing):
            m, fg_loss = self.conv(out, data.edge_index, data.edge_attr)
            m = F.relu(m)
            fg_loss_all += fg_loss
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out, fg_loss_all/self.num_step_message_passing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MPNN(node_input_dim=train_dataset.num_features).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0
    fg_loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_model, fg_loss = model(data)
        loss = F.mse_loss(y_model, data.y)
        loss_with_fg = loss + 0.001 * fg_loss
        loss_with_fg.backward()
        loss_all += loss.item() * data.num_graphs
        fg_loss_all += fg_loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset), fg_loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()

    targets = dict()
    for data in loader:
        data = data.to(device)
        y_pred, fg_loss = model(data)
        #print("data.sdfid", data.sdfid)
        for i in range(len(data.sdfid)):
            #print("data.sdfid[i].item()", data.sdfid[i].item())
            #print("y_pred[i]", y_pred[i])
            targets[data.sdfid[i].item()] = y_pred[i].tolist()
    return targets

def save_model(epoch):
    save_file = "./model/alchemy.ckp-"+ str(epoch % 5 + 1)
    torch.save(model.state_dict(), save_file)

def listdir(path, list_name):
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            listdir(file_path, list_name)
        else:
            list_name.append(file_path)

def sort_file_by_time(file_path):
    files = os.listdir(file_path)
    if not files:
        return
    else:
        files = sorted(files, key=lambda x: os.path.getmtime(os.path.join(file_path, x)), reverse=True)
        return files


def load_model():
    files = sort_file_by_time("model")

    if (files == None):
        print("no saved model params file.")
        return

    print("Loading model params:", files[0])
    model.load_state_dict(torch.load("model/"+files[0]))
    print("Loading model params done.")


def test_mse(loader):
    model.eval()
    loss_all = 0
    fg_loss_all = 0

    for data in loader:
        data = data.to(device)
        y_model, fg_loss = model(data)
        loss = F.mse_loss(y_model, data.y)
        loss_with_fg = loss + 0.001 * fg_loss
        loss_all += loss.item() * data.num_graphs
        fg_loss_all += fg_loss.item() * data.num_graphs
    return loss_all / len(loader.dataset), fg_loss_all / len(loader.dataset)

#epoch = 400
print("Generating test result for mae use...")

load_model()

targets = test(test_loader)
df_targets = pd.DataFrame.from_dict(targets, orient="index", columns=['property_%d' % x for x in range(12)])
df_targets.sort_index(inplace=True)
df_targets.to_csv('mytest.csv', index_label='gdb_idx')
print("Generating mytest.csv done.")
