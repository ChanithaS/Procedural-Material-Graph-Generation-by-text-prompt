import os
import pathlib

import torch
from torch.utils.data import random_split
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset, download_url

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

class ShaderGraphDataset(InMemoryDataset):
    def __init__(self, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter)
        # Load dataset before splitting
        self.dataset_path = '/Users/chanithas/Desktop/fyp/Procedural-Material-Graph-Generation-by-text-prompt/data/shader/my_graph_data.pt'  
        self.all_data = torch.load(self.dataset_path)

        self.splitting()  # Perform splitting
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    @property
    def processed_file_names(self):
        # Assuming the processed files are named according to their split.
        return [f'{self.split}.pt']

    def splitting(self):
        # # Path to your dataset file
        # dataset_path = '/Users/chanithas/Desktop/fyp/Procedural-Material-Graph-Generation-by-text-prompt/data/shader/raw/my_graph_data.pt'
        
        # # Loading the dataset from the .pt file
        # all_data = torch.load(dataset_path)
        
        # Splitting the data (assuming all_data is a list of Data objects or a similar structure)
        # You might need to adjust the logic based on the actual structure of your dataset
        num_graphs = len(self.all_data)
        
        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        # Define split sizes (for example 70% train, 15% val, 15% test)
        test_len = int(round(num_graphs * 0.2))
        train_len = int(round((num_graphs - test_len) * 0.8))
        val_len = num_graphs - train_len - test_len
        indices = torch.randperm(num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        # Splitting indices for each dataset
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]
        
        train_data = []
        val_data = []
        test_data = []

        for i, adj in enumerate(self.all_data):
            if i in train_indices:
                train_data.append(adj)
            elif i in val_indices:
                val_data.append(adj)
            elif i in test_indices:
                test_data.append(adj)
            else:
                raise ValueError(f'Index {i} not in any split')

        torch.save(train_data, self.raw_paths[0])
        torch.save(val_data, self.raw_paths[1])
        torch.save(test_data, self.raw_paths[2])

    def process(self):
        # Assuming 'adj_matrix', 'node_features', etc. are in each 'graph_data'
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = torch.load(self.raw_paths[file_idx[self.split]])
        data_list = []
        for graph_data in raw_dataset:  # Access the correct split
            adj_matrix = graph_data['adj_matrix']
            node_features = graph_data['node_features']
            node_types = graph_data['node_types']
            edge_attr = graph_data.get('edge_attr', None)
            # edge_index = graph_data["edge_index"]
            num_nodes = graph_data['no_of_nodes']

            # num_nodes = node_features.size(0)

            edge_index, _ = torch_geometric.utils.dense_to_sparse(adj_matrix)
            data = torch_geometric.data.Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr, n_nodes=num_nodes)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])



class ShaderGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=200):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        self.datasets = {
            'train': ShaderGraphDataset(split='train', root=root_path),
            'val': ShaderGraphDataset(split='val', root=root_path),
            'test': ShaderGraphDataset(split='test', root=root_path)
        }
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, self.datasets)
        self.inner = self.datasets['train'] 

    def __getitem__(self, item):
        return self.inner[item]


class ShaderDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'shader_graphs'
        # Extract maximum node count
        self.n_nodes = max(d.n_nodes for d in datamodule.train_dataset) 

        # Collect unique node types
        self.node_types = torch.unique(
            torch.cat([d.x[:, 0] for d in datamodule.train_dataset])
        ).long()  # Assuming first feature in 'x' indicates node type

        # Consider edge types if relevant
        if datamodule.train_dataset[0].edge_attr is not None:
            self.edge_types = torch.unique(
                torch.cat([d.edge_attr for d in datamodule.train_dataset])
            ).long()
        else:
            self.edge_types = torch.tensor([1])
        super().complete_infos(self.n_nodes, self.node_types)

# self.n_nodes: The maximum number of nodes found in any single graph within the whole dataset.

# self.node_types:  A collection of all unique node types that exist across all graphs in the dataset.

# self.edge_types:  Similar to node types, this represents all unique edge types found across graphs in the dataset (if edge attributes are present).