#!/usr/bin/env python
# encoding: utf-8
# File Name: Alchemy_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/01/09 16:23
# TODO:

import os.path as osp
import networkx as nx
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import dgl
from dgl.data.utils import get_download_dir
from dgl.data.utils import download
from dgl.data.utils import extract_archive
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from collections import namedtuple
import pathlib
import pandas as pd

_urls = {
        'Alchemy': 'https://alchemy.tencent.com/data/'
        }

AlchemyBatcher = namedtuple('AlchemyBatch', ['graph', 'label'])

def batcher(device):
    def batcher_dev(batch):
        graphs, labels = zip(*batch)
        batch_graphs = dgl.batch(graphs)
        labels = torch.stack(labels, 0)
        return AlchemyBatcher(graph=batch_graphs, label=labels)
    return batcher_dev

class TencentAlchemyDataset(Dataset):

    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def alchemy_nodes(self, g):
        feat = []
        for n, d in g.nodes(data=True):
            h_t = []
            # Atom type (One-hot H, C, N, O F)
            h_t += [int(d['a_type'] == x) for x in ['H', 'C', 'N', 'O', 'F', 'S', 'Cl']]
            # Atomic number
            h_t.append(d['a_num'])
            # Acceptor
            h_t.append(d['acceptor'])
            # Donor
            h_t.append(d['donor'])
            # Aromatic
            h_t.append(int(d['aromatic']))
            # Hybradization
            h_t += [int(d['hybridization'] == x) \
                    for x in (Chem.rdchem.HybridizationType.SP, \
                        Chem.rdchem.HybridizationType.SP2,
                        Chem.rdchem.HybridizationType.SP3)]
            h_t.append(d['num_h'])
            feat.append((n, torch.FloatTensor(h_t)))

        nx.set_node_attributes(g, dict(feat), "n_feat")

    def alchemy_edges(self, g):
        e={}
        for n1, n2, d in g.edges(data=True):
            e_t = [float(d['b_type'] == x)
                    for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]
            e[(n1, n2)] = e_t
        nx.set_edge_attributes(g, e, "e_feat")

    # sdf file reader for Alchemy dataset
    def sdf_graph_reader(self, sdf_file):

        with open(sdf_file, 'r') as f:
            sdf_string = f.read()
        mol = Chem.MolFromMolBlock(sdf_string, removeHs=False)
        if mol is None:
            print("rdkit can not parsing", sdf_file)
            return None
        feats = self.chem_feature_factory.GetFeaturesForMol(mol)

        g = nx.DiGraph()
        l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()) \
                if self.mode == 'dev' else torch.LongTensor([int(sdf_file.stem)])

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                    aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                    num_h=atom_i.GetTotalNumHs(),
                    pos=torch.FloatTensor(geom[i]))

        for i in range(len(feats)):
            if feats[i].GetFamily() == 'Donor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['donor'] = 1
            elif feats[i].GetFamily() == 'Acceptor':
                node_list = feats[i].GetAtomIds()
                for i in node_list:
                    g.node[i]['acceptor'] = 1
        # Read Edges
        for i in range(mol.GetNumAtoms()):
            for j in range(mol.GetNumAtoms()):
                e_ij = mol.GetBondBetweenAtoms(i, j)
                if e_ij is not None:
                    g.add_edge(i, j, b_type=e_ij.GetBondType())

        self.alchemy_nodes(g)
        self.alchemy_edges(g)
        ret = dgl.DGLGraph()
        ret.from_networkx(g, node_attrs=['n_feat', 'pos'], edge_attrs=['e_feat'])

        return ret, l


    def __init__(self, mode='dev', transform=None):
        assert mode in ['dev', 'valid', 'test'], "mode should be dev/valid/test"
        self.mode = mode
        self.transform = transform
        self.file_dir = pathlib.Path(get_download_dir(), mode)
        self.zip_file_path = pathlib.Path(get_download_dir(), '%s.zip' % mode)
        download(_urls['Alchemy'] + "%s.zip" % mode, path=self.zip_file_path)
        extract_archive(str(self.zip_file_path), self.file_dir)

        self._load()

    def _load(self):
        if self.mode == 'dev':
            target_file = pathlib.Path(self.file_dir, "train.csv")
            self.target = pd.read_csv(target_file, index_col=0,
                    usecols=['gdb_idx',] + ['property_%d' % x for x in range(12)])
            self.target = self.target[['property_%d' % x for x in range(12)]]

        sdf_dir = pathlib.Path(self.file_dir, "sdf")
        self.graphs, self.labels = [], []
        for sdf_file in sdf_dir.glob("**/*.sdf"):
            result = self.sdf_graph_reader(sdf_file)
            if result is None:
                continue
            self.graphs.append(result[0])
            self.labels.append(result[1])
        print(len(self.graphs), "loaded!")

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        g, l = self.graphs[idx], self.labels[idx]
        if self.transform:
            g = self.transform(g)
        return g, l
