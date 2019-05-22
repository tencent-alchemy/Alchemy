#!/usr/bin/env python
# encoding: utf-8
# File Name: Alchemy_dataset.py
# Author: Jiezhong Qiu
# Create Time: 2019/05/08 14:55
# TODO:

import os.path as osp
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import networkx as nx
import pathlib
import pandas as pd

_urls = {
        'dev': 'https://alchemy.tencent.com/data/dev.zip',
        'valid': 'https://alchemy.tencent.com/data/valid.zip',
        'test': 'https://alchemy.tencent.com/data/test.zip',
        }

class TencentAlchemyDataset(InMemoryDataset):
    fdef_name = osp.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)

    def __init__(self, root, mode='dev', transform=None, pre_transform=None, pre_filter=None):
        self.mode = mode
        assert mode in _urls
        super(TencentAlchemyDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [self.mode + '/sdf', self.mode + '/train.csv'] if self.mode == 'dev' else [self.mode + '/sdf', ]

    @property
    def processed_file_names(self):
        return 'TencentAlchemy_%s.pt' % self.mode

    def download(self):
        raise NotImplementedError('please download and unzip dataset from %s, and put it at %s' % (_urls[self.mode], self.raw_dir))

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
            feat.append((n, h_t))
        feat.sort(key=lambda item: item[0])
        node_attr = torch.FloatTensor([item[1] for item in feat])
        return node_attr

    def alchemy_edges(self, g):
        e={}
        for n1, n2, d in g.edges(data=True):
            e_t = [int(d['b_type'] == x)
                    for x in (Chem.rdchem.BondType.SINGLE, \
                            Chem.rdchem.BondType.DOUBLE, \
                            Chem.rdchem.BondType.TRIPLE, \
                            Chem.rdchem.BondType.AROMATIC)]
            e[(n1, n2)] = e_t

        edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
        edge_attr = torch.FloatTensor(list(e.values()))
        return edge_index, edge_attr

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

        # for training set, we store its target
        # otherwise, we store its molecule id
        l = torch.FloatTensor(self.target.loc[int(sdf_file.stem)].tolist()).unsqueeze(0) \
                if self.mode == 'dev' else torch.LongTensor([int(sdf_file.stem)])

        # Create nodes
        assert len(mol.GetConformers()) == 1
        geom = mol.GetConformers()[0].GetPositions()
        for i in range(mol.GetNumAtoms()):
            atom_i = mol.GetAtomWithIdx(i)
            g.add_node(i, a_type=atom_i.GetSymbol(), a_num=atom_i.GetAtomicNum(), acceptor=0, donor=0,
                    aromatic=atom_i.GetIsAromatic(), hybridization=atom_i.GetHybridization(),
                    num_h=atom_i.GetTotalNumHs())

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

        node_attr = self.alchemy_nodes(g)
        edge_index, edge_attr = self.alchemy_edges(g)
        data = Data(
                x=node_attr,
                pos=torch.FloatTensor(geom),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=l,
                )
        return data

    def process(self):
        if self.mode == 'dev':
            self.target = pd.read_csv(self.raw_paths[1], index_col=0,
                    usecols=['gdb_idx',] + ['property_%d' % x for x in range(12)])
            self.target = self.target[['property_%d' % x for x in range(12)]]
        sdf_dir = pathlib.Path(self.raw_paths[0])
        data_list = []
        for sdf_file in sdf_dir.glob("**/*.sdf"):
            alchemy_data = self.sdf_graph_reader(sdf_file)
            if alchemy_data is not None:
                data_list.append(alchemy_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


