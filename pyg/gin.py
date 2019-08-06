
import torch
import torch.nn as nn
import torch.nn.functional as F

from Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import GINConv, Set2Set
from torch_geometric.data import DataLoader

import pandas as pd


train_dataset = TencentAlchemyDataset(root='data-bin', mode='dev').shuffle()
valid_dataset = TencentAlchemyDataset(root='data-bin', mode='valid')


valid_loader = DataLoader(valid_dataset, batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

        
class GIN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 output_dim=12,
                 node_hidden_dim=64,
                 num_step_prop=6,
                 num_step_set2set=6):
        super(GIN, self).__init__()
        self.num_step_prop = num_step_prop
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        self.mlps = torch.nn.ModuleList()
        self.convs = torch.nn.ModuleList()
        for i in range(num_step_prop):
            self.mlps.append(nn.Sequential(nn.Linear(node_hidden_dim, node_hidden_dim), nn.BatchNorm1d(node_hidden_dim), nn.ReLU(),
                                           nn.Linear(node_hidden_dim, node_hidden_dim), nn.BatchNorm1d(node_hidden_dim), nn.ReLU()))
            self.convs.append(GINConv(self.mlps[i], eps=0, train_eps=False))
        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        
        for i in range(self.num_step_prop):
            out = self.convs[i](out, data.edge_index)
   
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
               
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GIN(node_input_dim=train_dataset.num_features).to(device)
print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        y_model = model(data)
        loss = F.mse_loss(y_model, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()

    targets = dict()
    for data in loader:
        data = data.to(device)
        y_pred = model(data)
        for i in range(len(data.y)):
            targets[data.y[i].item()] = y_pred[i].tolist()
    return targets

epoch = 1
print("training...")
for epoch in range(epoch):
    loss = train(epoch)
    print('Epoch: {:03d}, Loss: {:.7f}'.format(epoch, loss))

targets = test(valid_loader)
df_targets = pd.DataFrame.from_dict(targets, orient="index", columns=['property_%d' % x for x in range(12)])
df_targets.sort_index(inplace=True)
df_targets.to_csv('targets.csv', index_label='gdb_idx')
