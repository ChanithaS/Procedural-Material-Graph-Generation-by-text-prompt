import os
import pathlib

import torch
from torch.utils.data import random_split
import torch.nn.functional as F
import torch_geometric.utils
from torch_geometric.data import InMemoryDataset
import pandas as pd
import ast

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos

class ShaderGraphDataset(InMemoryDataset):
    def __init__(self, dataset_name, split, root, transform=None, pre_transform=None, pre_filter=None):
        self.dataset_name = dataset_name
        self.split = split
        self.num_graphs = 95
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def processed_file_names(self):
            return [self.split + '.csv']

    def download(self):
        file_path = os.path.join(self.raw_dir, 'shader_dataset.csv')
        df = pd.read_csv(file_path)

        g_cpu = torch.Generator()
        g_cpu.manual_seed(0)

        test_len = int(round(self.num_graphs * 0.2))
        train_len = int(round((self.num_graphs - test_len) * 0.8))
        val_len = self.num_graphs - train_len - test_len
        indices = torch.randperm(self.num_graphs, generator=g_cpu)
        print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')
        train_indices = indices[:train_len]
        val_indices = indices[train_len:train_len + val_len]
        test_indices = indices[train_len + val_len:]

        # Splitting data into train, validation, and test sets
        train_data = df.iloc[train_indices]
        val_data = df.iloc[val_indices]
        test_data = df.iloc[test_indices]

    
        train_data.to_csv(os.path.join(self.raw_paths[0]), index=False)
        val_data.to_csv(os.path.join(self.raw_paths[1]), index=False)
        test_data.to_csv(os.path.join(self.raw_paths[2]), index=False)


    def process(self):
        file_idx = {'train': 0, 'val': 1, 'test': 2}
        raw_dataset = pd.read_csv(self.raw_paths[file_idx[self.split]])

        edge_types_all = [[0, 1], [3, 0], [0, 0], [0, 6], [1, 0], [2, 6], [2, 7], [2, 0], [0, 2], [0, 7], [2, 2], [0, 22], [2, 17], [2, 9], [2, 1], [4, 1], [0, 16], [0, 4], [4, 7], [0, 3], [1, 6], [5, 1], [0, 9], [1, 7], [0, 17], [1, 4], [1, 1], [12, 0], [6, 1], [7, 0], [0, 12], [0, 21], [0, 18], [6, 0], [5, 0], [2, 4], [1, 2], [3, 10], [3, 1], [0, 15], [0, 23], [2, 18], [0, 5], [0, 10], [8, 0], [2, 5], [0, 8], [0, 19], [0, 20], [4, 0], [3, 7], [2, 19], [11, 0]]
        node_types_all = ['ShaderNodeValToRGB', 'ShaderNodeTexNoise', 'ShaderNodeMix', 'ShaderNodeDisplacement', 'ShaderNodeBsdfRefraction', 'ShaderNodeBsdfTransparent', 'ShaderNodeMixShader', 'ShaderNodeTexCoord', 'ShaderNodeOutputMaterial', 'ShaderNodeTexMusgrave', 'ShaderNodeTexVoronoi', 'ShaderNodeMapping', 'ShaderNodeTexWave', 'ShaderNodeBump', 'ShaderNodeInvert', 'ShaderNodeBsdfPrincipled', 'ShaderNodeRGB', 'ShaderNodeValue', 'ShaderNodeNewGeometry', 'ShaderNodeVectorMath', 'ShaderNodeGamma', 'ShaderNodeLightPath', 'ShaderNodeBsdfDiffuse', 'ShaderNodeMath', 'ShaderNodeBsdfAnisotropic', 'ShaderNodeFresnel', 'ShaderNodeLayerWeight', 'ShaderNodeRGBCurve', 'ShaderNodeBsdfGlass', 'ShaderNodeSeparateXYZ', 'ShaderNodeMapRange', 'ShaderNodeHueSaturation', 'ShaderNodeAddShader', 'ShaderNodeEmission', 'ShaderNodeVolumeAbsorption', 'ShaderNodeBsdfGlossy', 'ShaderNodeRGBToBW', 'ShaderNodeTexGradient', 'ShaderNodeBrightContrast', 'ShaderNodeSeparateColor', 'ShaderNodeAmbientOcclusion', 'ShaderNodeObjectInfo', 'ShaderNodeVectorRotate', 'ShaderNodeVolumePrincipled', 'ShaderNodeVolumeInfo', 'ShaderNodeTexBrick', 'ShaderNodeVectorCurve', 'ShaderNodeCombineXYZ', 'ShaderNodeEeveeSpecular', 'ShaderNodeWireframe', 'ShaderNodeBsdfTranslucent', 'ShaderNodeBevel', 'ShaderNodeShaderToRGB', 'ShaderNodeCameraData', 'ShaderNodeNormalMap', 'ShaderNodeTexMagic', 'ShaderNodeTexChecker', 'ShaderNodeBsdfToon', 'ShaderNodeVertexColor']
        data_list = []
        for idx, row in raw_dataset.iterrows():
            # Assuming the data has columns 'adjacency', 'n_nodes'
            type_idx_str = row['Nodes']
            type_idx = ast.literal_eval(type_idx_str)
            N = len(type_idx)

            edge_index_csv = row['Edges']
            edge_index_list = ast.literal_eval(edge_index_csv)
            edge_index = torch.tensor(edge_index_list, dtype=torch.long)

            edge_type_csv = row['Edge_types']
            edge_type_list = ast.literal_eval(edge_type_csv)
            edge_type_list_doubled = []
            for num in edge_type_list:
                edge_type_list_doubled.extend([num, num])
            edge_type = torch.tensor(edge_type_list_doubled, dtype=torch.long)

            edge_attr = F.one_hot(edge_type, num_classes=len(edge_types_all)+1).to(torch.float)

            print("Size of the bonds using .size() method:", len(edge_types_all))
            print("Size of the edge_type using .size() method:", edge_type.size())
            print("Size of the edge_index using .size() method:", edge_index.size())
            print("Size of the edge_attr using .size() method:", edge_attr.size())

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(node_types_all)).float()
            y = torch.zeros((1, 0), dtype=torch.float)

            data = torch_geometric.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=idx)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)
        
        # Save processed data
        torch.save(self.collate(data_list), self.processed_paths[0])



class ShaderGraphDataModule(AbstractDataModule):
    def __init__(self, cfg, n_graphs=95):
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)


        datasets = {'train': ShaderGraphDataset(dataset_name=self.cfg.dataset.name,
                                                 split='train', root=root_path),
                    'val': ShaderGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='val', root=root_path),
                    'test': ShaderGraphDataset(dataset_name=self.cfg.dataset.name,
                                        split='test', root=root_path)}
        # print(f'Dataset sizes: train {train_len}, val {val_len}, test {test_len}')

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]


class ShaderDatasetInfos(AbstractDatasetInfos):
    def __init__(self, datamodule, dataset_config):
        self.datamodule = datamodule
        self.name = 'nx_graphs'
        self.n_nodes = self.datamodule.node_counts()
        self.node_types = self.datamodule.node_types()              # There are no node types
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)