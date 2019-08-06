
import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
from Alchemy_dataset import TencentAlchemyDataset
from torch_geometric.nn import RGCNConv, Set2Set
from torch_geometric.data import DataLoader
from torch_geometric.utils import remove_self_loops

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


transform = T.Compose([Complete()]) # This transform is executed in dataloader rather than pretransform.
train_dataset = TencentAlchemyDataset(root='data-bin', mode='dev', transform=transform).shuffle()
valid_dataset = TencentAlchemyDataset(root='data-bin', mode='valid', transform=transform)
train_dataset.data.edge_attr = train_dataset.data.edge_attr.argmax(dim=1)+1 # one-hot -> scalar, e.g., [1,0,0,0] -> 1
valid_dataset.data.edge_attr = valid_dataset.data.edge_attr.argmax(dim=1)+1

valid_loader = DataLoader(valid_dataset, batch_size=64)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

class RGCN(torch.nn.Module):
    def __init__(self,
                 node_input_dim=15,
                 num_edge_type=5,
                 output_dim=12,
                 node_hidden_dim=64,
                 num_basis=-1,
                 num_step_prop=6,
                 num_step_set2set=6):
        super(RGCN, self).__init__()
        self.num_step_prop = num_step_prop
        self.lin0 = nn.Linear(node_input_dim, node_hidden_dim)
        if num_basis<0:
            self.conv = RGCNConv(node_hidden_dim, node_hidden_dim, num_edge_type, num_edge_type)
        else:
            self.conv = RGCNConv(node_hidden_dim, node_hidden_dim, num_edge_type, num_basis)
            
        self.set2set = Set2Set(node_hidden_dim, processing_steps=num_step_set2set)
        self.lin1 = nn.Linear(2 * node_hidden_dim, node_hidden_dim)
        self.lin2 = nn.Linear(node_hidden_dim, output_dim)

    def forward(self, data):
        out = F.relu(self.lin0(data.x))
        
        for i in range(self.num_step_prop):
            out = F.relu(self.conv(out, data.edge_index, data.edge_attr))
   
        out = self.set2set(out, data.batch)
        out = F.relu(self.lin1(out))
        out = self.lin2(out)
                       
        return out  

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = RGCN(node_input_dim=train_dataset.num_features).to(device)
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
