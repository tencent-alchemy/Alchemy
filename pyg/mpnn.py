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
from tdataset import TencentAlchemyDataset
from torch_geometric.nn import NNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

from datetime import datetime
import time
import logging

import pandas as pd


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


class MPNN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 edge_input_dim=5,
                 output_dim=1,
                 node_hidden_dim=64,
                 edge_hidden_dim=128,
                 num_step_message_passing=6,
                 num_step_set2set=6):
        super(MPNN, self).__init__()
        self.num_step_message_passing = num_step_message_passing
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        edge_network = nn.Sequential(
            nn.Linear(edge_input_dim, edge_hidden_dim), nn.ReLU(),
            nn.Linear(edge_hidden_dim, node_hidden_dim * node_hidden_dim))
        self.conv = NNConv(node_hidden_dim,
                           node_hidden_dim,
                           edge_network,
                           aggr='mean',
                           root_weight=False)
        self.gru = nn.GRU(node_hidden_dim, node_hidden_dim)

        self.set2set = Set2Set(node_hidden_dim,
                               processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        h = out.unsqueeze(0)

        for i in range(self.num_step_message_passing):
            m = F.relu(self.conv(out, data.edge_index, data.edge_attr))
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)

        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
        return out


def run(prop="homo", gpuid="0", epoch=500, dataset="t2", size=100000):

    # set logger
    task_name = "MPNN_%s_%s_%s" % (dataset, prop,
                                   datetime.now().strftime("%m%d_%H%M%S"))
    logname = "./logs/%s.log" % (task_name)
    log = logging.getLogger(task_name)
    log.setLevel(logging.INFO)
    fmt = "%(asctime)-s %(levelname)s %(filename)s %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    handler = logging.FileHandler(filename=logname)  # output to file
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(fmt, datefmt))
    log.addHandler(handler)
    chler = logging.StreamHandler()  # print to console
    chler.setFormatter(logging.Formatter(fmt, datefmt))
    chler.setLevel(logging.INFO)
    log.addHandler(chler)
    log.info("Experiment of model: %s, dataset size: %d" % (task_name, size))

    device = torch.device("cuda:%s" % (gpuid))
    transform = T.Compose([Complete(), T.Distance(norm=False)])
    dataset = TencentAlchemyDataset(root='./tdata/',
                                    mode='dev',
                                    dataset=dataset,
                                    prop=prop,
                                    transform=transform).shuffle()
    dataset = dataset[:size]
    trainset = dataset[:size - 20000]
    valset = dataset[size - 20000:size - 10000]
    testset = dataset[size - 10000:]
    train_loader = DataLoader(trainset, batch_size=64)
    val_loader = DataLoader(valset, batch_size=64)
    test_loader = DataLoader(testset, batch_size=64)
    model = MPNN(node_input_dim=trainset.num_features).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002)
    model.train()
    loss_all = 0
    loss = 0
    mae = 0

    st = time.time()
    best_valid = float("inf")
    for it in range(epoch):

        # train
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            y_model = model(data)
            loss = F.mse_loss(y_model, data.y)
            mae += F.l1_loss(y_model, data.y).item()
            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        loss = loss_all / len(train_loader)
        train_score = mae / len(train_loader)
        mae = 0
        for data in val_loader:
            data = data.to(device)
            y_model = model(data)
            mae += F.l1_loss(y_model, data.y).item()
        valid_score = mae / len(val_loader)
        mae = 0
        for data in test_loader:
            data = data.to(device)
            y_model = model(data)
            mae += F.l1_loss(y_model, data.y).item()
        test_score = mae / len(test_loader)

        log.info("Epoch {:2d}, train loss {:.7f}, test loss no, \
                 train mae {:.7f}, val mae {:.7f}, test mae {:.7f}".format(
            it, loss, train_score, valid_score, test_score))

        if valid_score < best_valid:
            best_valid = valid_score
            related_test = test_score
            ed = time.time()

    log.info(
        "Best val mae: {:.7f}  Related test mae: {:.7f}  Time cost: {:.0f}".
        format(best_valid, related_test, ed - st))


if __name__ == "__main__":
    import sys
    gpuid = sys.argv[1]
    size = int(sys.argv[2])
    dataset = sys.argv[3]
    prop = sys.argv[4]
    run(prop, gpuid, 500, dataset, size)
