# this file contains all the class in the same file
# Datast File
import os
import os.path as osp
import pathlib
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from rdkit import Chem, RDLogger
from rdkit.Chem.rdchem import BondType as BT
from tqdm import tqdm
import numpy as np
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from torch_geometric.utils import subgraph

import src.utils as utils
from src.datasets.abstract_dataset import MolecularDataModule, AbstractDatasetInfos
from src.analysis.rdkit_functions import  mol2smiles, build_molecule_with_partial_charges
from src.analysis.rdkit_functions import compute_molecular_metrics


HAR2EV = 27.211386246
KCALMOL2EV = 0.04336414

conversion = torch.tensor([1., 1., HAR2EV, HAR2EV, HAR2EV, 1., HAR2EV, HAR2EV, HAR2EV, HAR2EV, HAR2EV,
                           1., KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, KCALMOL2EV, 1., 1., 1.])


class RemoveYTransform:
    def __call__(self, data):
        data.y = torch.zeros((1, 0), dtype=torch.float)
        return data


class SelectMuTransform:
    def __call__(self, data):
        data.y = data.y[..., :1]
        return data


class SelectHOMOTransform:
    def __call__(self, data):
        data.y = data.y[..., 1:]
        return data


def files_exist(files) -> bool:
    # NOTE: We return `False` in case `files` is empty, leading to a
    # re-processing of files on every instantiation.
    return len(files) != 0 and all([osp.exists(f) for f in files])


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]


class QM9Dataset(InMemoryDataset):
    raw_url = ('https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/'
               'molnet_publish/qm9.zip')
    raw_url2 = 'https://ndownloader.figshare.com/files/3195404'
    processed_url = 'https://data.pyg.org/datasets/qm9_v3.zip'

    def __init__(self, stage, root, remove_h: bool, transform=None, pre_transform=None, pre_filter=None):
        """ stage: train, val, test
            root: data directory
            remove_h: remove hydrogens
            target_prop: property to predict (for guidance only).
        """
        self.stage = stage
        if self.stage == 'train':
            self.file_idx = 0
        elif self.stage == 'val':
            self.file_idx = 1
        else:
            self.file_idx = 2
        self.remove_h = remove_h
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

    @property
    def raw_file_names(self):
        return ['gdb9.sdf', 'gdb9.sdf.csv', 'uncharacterized.txt']

    @property
    def split_file_name(self):
        return ['train.csv', 'val.csv', 'test.csv']

    @property
    def split_paths(self):
        r"""The absolute filepaths that must be present in order to skip
        splitting."""
        files = to_list(self.split_file_name)
        return [osp.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        if self.remove_h:
            return ['proc_tr_no_h.pt', 'proc_val_no_h.pt', 'proc_test_no_h.pt']
        else:
            return ['proc_tr_h.pt', 'proc_val_h.pt', 'proc_test_h.pt']

    def download(self):
        """
        Download raw qm9 files. Taken from PyG QM9 class
        """
        try:
            import rdkit  # noqa
            file_path = download_url(self.raw_url, self.raw_dir)
            extract_zip(file_path, self.raw_dir)
            os.unlink(file_path)

            file_path = download_url(self.raw_url2, self.raw_dir)
            os.rename(osp.join(self.raw_dir, '3195404'),
                      osp.join(self.raw_dir, 'uncharacterized.txt'))
        except ImportError:
            path = download_url(self.processed_url, self.raw_dir)
            extract_zip(path, self.raw_dir)
            os.unlink(path)

        if files_exist(self.split_paths):
            return

        dataset = pd.read_csv(self.raw_paths[1])

        n_samples = len(dataset)
        n_train = 100000
        n_test = int(0.1 * n_samples)
        n_val = n_samples - (n_train + n_test)

        # Shuffle dataset with df.sample, then split
        train, val, test = np.split(dataset.sample(frac=1, random_state=42), [n_train, n_val + n_train])

        train.to_csv(os.path.join(self.raw_dir, 'train.csv'))
        val.to_csv(os.path.join(self.raw_dir, 'val.csv'))
        test.to_csv(os.path.join(self.raw_dir, 'test.csv'))

    def process(self):
        RDLogger.DisableLog('rdApp.*')

        types = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
        bonds = {BT.SINGLE: 0, BT.DOUBLE: 1, BT.TRIPLE: 2, BT.AROMATIC: 3}

        target_df = pd.read_csv(self.split_paths[self.file_idx], index_col=0)
        target_df.drop(columns=['mol_id'], inplace=True)
        target = torch.tensor(target_df.values, dtype=torch.float)
        target = torch.cat([target[:, 3:], target[:, :3]], dim=-1)
        target = target * conversion.view(1, -1)

        with open(self.raw_paths[-1], 'r') as f:
            skip = [int(x.split()[0]) - 1 for x in f.read().split('\n')[9:-2]]

        suppl = Chem.SDMolSupplier(self.raw_paths[0], removeHs=False, sanitize=False)

        data_list = []
        for i, mol in enumerate(tqdm(suppl)):
            if i in skip or i not in target_df.index:
                continue

            N = mol.GetNumAtoms()

            type_idx = []
            for atom in mol.GetAtoms():
                type_idx.append(types[atom.GetSymbol()])

            row, col, edge_type = [], [], []
            for bond in mol.GetBonds():
                start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                row += [start, end]
                col += [end, start]
                edge_type += 2 * [bonds[bond.GetBondType()] + 1]

            edge_index = torch.tensor([row, col], dtype=torch.long)
            edge_type = torch.tensor(edge_type, dtype=torch.long)
            edge_attr = F.one_hot(edge_type, num_classes=len(bonds)+1).to(torch.float)

            perm = (edge_index[0] * N + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

            x = F.one_hot(torch.tensor(type_idx), num_classes=len(types)).float()

            y = target[target_df.index.get_loc(i)].unsqueeze(0)
            y = torch.hstack((y[..., :1], y[..., 2:3]))         # mu, homo

            if self.remove_h:
                type_idx = torch.Tensor(type_idx).long()
                to_keep = type_idx > 0
                edge_index, edge_attr = subgraph(to_keep, edge_index, edge_attr, relabel_nodes=True,
                                                 num_nodes=len(to_keep))
                x = x[to_keep]
                # Shift onehot encoding to match atom decoder
                x = x[:, 1:]

            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=i)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])


class QM9DataModule(MolecularDataModule):
    def __init__(self, cfg, regressor: bool = False):
        self.datadir = cfg.dataset.datadir
        self.regressor = regressor
        super().__init__(cfg)
        self.remove_h = cfg.dataset.remove_h

    def prepare_data(self) -> None:
        target = self.cfg.general.guidance_target
        if self.regressor and target == 'mu':
            transform = SelectMuTransform()
        elif self.regressor and target == 'homo':
            transform = SelectHOMOTransform()
        elif self.regressor and target == 'both':
            transform = None
        else:
            transform = RemoveYTransform()

        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)
        datasets = {'train': QM9Dataset(stage='train', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                        transform=transform if self.regressor else RemoveYTransform()),
                    'val': QM9Dataset(stage='val', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                      transform=transform if self.regressor else RemoveYTransform()),
                    'test': QM9Dataset(stage='test', root=root_path, remove_h=self.cfg.dataset.remove_h,
                                       transform=transform)}
        super().prepare_data(datasets)


class QM9infos(AbstractDatasetInfos):
    def __init__(self, datamodule, cfg, recompute_statistics=False):
        self.remove_h = cfg.dataset.remove_h
        self.need_to_strip = False        # to indicate whether we need to ignore one output from the model

        self.name = 'qm9'
        if self.remove_h:
            self.atom_encoder = {'C': 0, 'N': 1, 'O': 2, 'F': 3}
            self.atom_decoder = ['C', 'N', 'O', 'F']
            self.num_atom_types = 4
            self.valencies = [4, 3, 2, 1]
            self.atom_weights = {0: 12, 1: 14, 2: 16, 3: 19}
            self.max_n_nodes = 9
            self.max_weight = 150
            self.n_nodes = torch.Tensor([0, 2.2930e-05, 3.8217e-05, 6.8791e-05, 2.3695e-04, 9.7072e-04,
                                         0.0046472, 0.023985, 0.13666, 0.83337])
            self.node_types = torch.Tensor([0.7230, 0.1151, 0.1593, 0.0026])
            self.edge_types = torch.Tensor([0.7261, 0.2384, 0.0274, 0.0081, 0.0])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0: 6] = torch.Tensor([2.6071e-06, 0.163, 0.352, 0.320, 0.16313, 0.00073])
        else:
            self.atom_encoder = {'H': 0, 'C': 1, 'N': 2, 'O': 3, 'F': 4}
            self.atom_decoder = ['H', 'C', 'N', 'O', 'F']
            self.valencies = [1, 4, 3, 2, 1]
            self.num_atom_types = 5
            self.max_n_nodes = 29
            self.max_weight = 390
            self.atom_weights = {0: 1, 1: 12, 2: 14, 3: 16, 4: 19}
            self.n_nodes = torch.Tensor([0, 0, 0, 1.5287e-05, 3.0574e-05, 3.8217e-05,
                                         9.1721e-05, 1.5287e-04, 4.9682e-04, 1.3147e-03, 3.6918e-03, 8.0486e-03,
                                         1.6732e-02, 3.0780e-02, 5.1654e-02, 7.8085e-02, 1.0566e-01, 1.2970e-01,
                                         1.3332e-01, 1.3870e-01, 9.4802e-02, 1.0063e-01, 3.3845e-02, 4.8628e-02,
                                         5.4421e-03, 1.4698e-02, 4.5096e-04, 2.7211e-03, 0.0000e+00, 2.6752e-04])

            self.node_types = torch.Tensor([0.5122, 0.3526, 0.0562, 0.0777, 0.0013])
            self.edge_types = torch.Tensor([0.88162,  0.11062,  5.9875e-03,  1.7758e-03, 0])

            super().complete_infos(n_nodes=self.n_nodes, node_types=self.node_types)
            self.valency_distribution = torch.zeros(3 * self.max_n_nodes - 2)
            self.valency_distribution[0:6] = torch.Tensor([0, 0.5136, 0.0840, 0.0554, 0.3456, 0.0012])

        if recompute_statistics:
            np.set_printoptions(suppress=True, precision=5)
            self.n_nodes = datamodule.node_counts()
            print("Distribution of number of nodes", self.n_nodes)
            np.savetxt('n_counts.txt', self.n_nodes.numpy())
            self.node_types = datamodule.node_types()                                     # There are no node types
            print("Distribution of node types", self.node_types)
            np.savetxt('atom_types.txt', self.node_types.numpy())

            self.edge_types = datamodule.edge_counts()
            print("Distribution of edge types", self.edge_types)
            np.savetxt('edge_types.txt', self.edge_types.numpy())

            valencies = datamodule.valency_count(self.max_n_nodes)
            print("Distribution of the valencies", valencies)
            np.savetxt('valencies.txt', valencies.numpy())
            self.valency_distribution = valencies
            assert False


def get_train_smiles(cfg, train_dataloader, dataset_infos, evaluate_dataset=False, ):
    if evaluate_dataset:
        assert dataset_infos is not None, "If wanting to evaluate dataset, need to pass dataset_infos"
    datadir = cfg.dataset.datadir
    remove_h = cfg.dataset.remove_h
    atom_decoder = dataset_infos.atom_decoder
    root_dir = pathlib.Path(os.path.realpath(__file__)).parents[2]
    smiles_file_name = 'train_smiles_no_h.npy' if remove_h else 'train_smiles_h.npy'
    smiles_path = os.path.join(root_dir, datadir, smiles_file_name)
    if os.path.exists(smiles_path):
        print("Dataset smiles were found.")
        train_smiles = np.load(smiles_path)
    else:
        print("Computing dataset smiles...")
        train_smiles = compute_qm9_smiles(atom_decoder, train_dataloader, remove_h)
        np.save(smiles_path, np.array(train_smiles))

    if evaluate_dataset:
        train_dataloader = train_dataloader
        all_molecules = []
        for i, data in enumerate(train_dataloader):
            dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
            dense_data = dense_data.mask(node_mask, collapse=True)
            X, E = dense_data.X, dense_data.E

            for k in range(X.size(0)):
                n = int(torch.sum((X != -1)[k, :]))
                atom_types = X[k, :n].cpu()
                edge_types = E[k, :n, :n].cpu()
                all_molecules.append([atom_types, edge_types])

        print("Evaluating the dataset -- number of molecules to evaluate", len(all_molecules))
        metrics = compute_molecular_metrics(molecule_list=all_molecules, train_smiles=train_smiles,
                                            dataset_info=dataset_infos)
        print(metrics[0])

    return train_smiles


def compute_qm9_smiles(atom_decoder, train_dataloader, remove_h):
    '''

    :param dataset_name: qm9 or qm9_second_half
    :return:
    '''
    print(f"\tConverting QM9 dataset to SMILES for remove_h={remove_h}...")

    mols_smiles = []
    len_train = len(train_dataloader)
    invalid = 0
    disconnected = 0
    for i, data in enumerate(train_dataloader):
        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask, collapse=True)
        X, E = dense_data.X, dense_data.E

        n_nodes = [int(torch.sum((X != -1)[j, :])) for j in range(X.size(0))]

        molecule_list = []
        for k in range(X.size(0)):
            n = n_nodes[k]
            atom_types = X[k, :n].cpu()
            edge_types = E[k, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        for l, molecule in enumerate(molecule_list):
            mol = build_molecule_with_partial_charges(molecule[0], molecule[1], atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                mols_smiles.append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                if len(mol_frags) > 1:
                    print(f"Disconnected molecule {len(mol_frags)} fragments")
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        if i % 1000 == 0:
            print("\tConverting QM9 dataset to SMILES {0:.2%}".format(float(i) / len_train))
    print("Number of invalid molecules", invalid)
    print("Number of disconnected molecules", disconnected)
    return mols_smiles

# Main Guidance file
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import psi4
from rdkit import Chem
import torch
import wandb
import hydra
import os
import omegaconf
from omegaconf import DictConfig, OmegaConf, open_dict
from pytorch_lightning import Trainer
from pytorch_lightning.utilities.warnings import PossibleUserWarning
import warnings


import src.utils as utils
from src.guidance.guidance_diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.datasets import qm9_dataset
from src.metrics.molecular_metrics import SamplingMolecularMetrics, TrainMolecularMetricsDiscrete
from src.analysis.visualization import MolecularVisualization
from src.diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from src.diffusion.extra_features_molecular import ExtraMolecularFeatures
from src.utils import update_config_with_new_keys
from src.guidance.qm9_regressor_discrete import Qm9RegressorDiscrete


warnings.filterwarnings("ignore", category=PossibleUserWarning)


def get_resume(cfg, model_kwargs):
    saved_cfg = cfg.copy()

    name = cfg.general.name + '_resume'
    resume = cfg.general.test_only
    final_samples_to_generate = cfg.general.final_model_samples_to_generate
    final_chains_to_save = cfg.general.final_model_chains_to_save
    batch_size = cfg.train.batch_size
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(resume, **model_kwargs)
    cfg = model.cfg
    cfg.general.test_only = resume
    cfg.general.name = name
    cfg.general.final_model_samples_to_generate = final_samples_to_generate
    cfg.general.final_model_chains_to_save = final_chains_to_save
    cfg.train.batch_size = batch_size
    cfg = update_config_with_new_keys(cfg, saved_cfg)
    return cfg, model


def setup_wandb(cfg):
    config_dict = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    kwargs = {'name': cfg.general.name, 'project': 'guidance', 'config': config_dict,
              'settings': wandb.Settings(_disable_stats=True),
              'reinit': True, 'mode': cfg.general.wandb}
    wandb.init(**kwargs)
    wandb.save('*.txt')
    return cfg


@hydra.main(config_path='../../configs', config_name='config')
def main(cfg: DictConfig):
    dataset_config = cfg["dataset"]
    assert dataset_config.name == "qm9", "Only QM9 dataset is supported for now"
    datamodule = qm9_dataset.QM9DataModule(cfg, regressor=True)
    dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
    datamodule.prepare_data()
    train_smiles = qm9_dataset.get_train_smiles(cfg, datamodule, dataset_infos)

    if cfg.model.extra_features is not None:
        extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)
    else:
        extra_features = DummyExtraFeatures()
        domain_features = DummyExtraFeatures()

    dataset_infos.compute_input_output_dims(datamodule=datamodule, extra_features=extra_features,
                                            domain_features=domain_features)

    train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
    sampling_metrics = SamplingMolecularMetrics(dataset_infos, train_smiles)
    visualization_tools = MolecularVisualization(cfg.dataset.remove_h, dataset_infos=dataset_infos)

    model_kwargs = {'dataset_infos': dataset_infos, 'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics, 'visualization_tools': visualization_tools,
                    'extra_features': extra_features, 'domain_features': domain_features, 'load_model': True}

    # When testing, previous configuration is fully loaded
    cfg_pretrained, guidance_sampling_model = get_resume(cfg, model_kwargs)

    OmegaConf.set_struct(cfg, True)
    with open_dict(cfg):
        cfg.model = cfg_pretrained.model
    model_kwargs['load_model'] = False

    utils.create_folders(cfg)
    cfg = setup_wandb(cfg)

    # load pretrained regressor
    # Fetch path to this file to get base path
    current_path = os.path.dirname(os.path.realpath(__file__))
    root_dir = current_path.split('outputs')[0]

    guidance_model = Qm9RegressorDiscrete.load_from_checkpoint(os.path.join(cfg.general.trained_regressor_path))

    model_kwargs['guidance_model'] = guidance_model

    if cfg.general.name == 'debug':
        print("[WARNING]: Run is called 'debug' -- it will run with fast_dev_run. ")
    trainer = Trainer(gradient_clip_val=cfg.train.clip_grad,
                      gpus=cfg.general.gpus if torch.cuda.is_available() else 0,
                      limit_test_batches=100,
                      max_epochs=cfg.train.n_epochs,
                      check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
                      fast_dev_run=cfg.general.name == 'debug',
                      strategy='ddp' if cfg.general.gpus > 1 else None,        # TODO CHANGE with ray
                      enable_progress_bar=False,
                      logger=[],
                      )

    # add for conditional sampling
    model = guidance_sampling_model
    model.args = cfg
    model.guidance_model = guidance_model
    trainer.test(model, datamodule=datamodule, ckpt_path=None)


if __name__ == '__main__':
    main()

#guidance diffusion model discrete.py
    import numpy as np
import torch
import pytorch_lightning as pl
import time
import wandb
import os
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import DiscreteUniformTransition, PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
import networkx as nx
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete
import src.utils as utils

# packages for conditional generation with guidance
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFHasAllMoleculeParams, MMFFOptimizeMolecule
from rdkit import Chem
import math
try:
    import psi4
except ModuleNotFoundError:
    print("PSI4 not found")
from src.analysis.rdkit_functions import build_molecule, mol2smiles, build_molecule_with_partial_charges
import pickle
import pandas as pd


class DiscreteDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features, guidance_model=None, load_model=False):
        super().__init__()

        # add for test
        if load_model:
            OmegaConf.set_struct(cfg, True)
            with open_dict(cfg):
                cfg.guidance = {'use_guidance': True, 'lambda_guidance': 0.5}

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.train_loss = TrainLossDiscrete(self.cfg.model.lambda_train)

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)
        # Marginal noise schedule
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_classes=self.ydim_output)
        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        # specific properties to generate molecules
        self.cond_val = MeanAbsoluteError()
        self.num_valid_molecules = 0
        self.num_total = 0

        self.guidance_model = guidance_model

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.args.train.lr, amsgrad=True,
                                 weight_decay=self.args.train.weight_decay)

    @torch.enable_grad()
    @torch.inference_mode(False)
    def test_step(self, data, i):
        print(f'Select No.{i+1} test molecule')
        # Extract properties
        target_properties = data.y.clone()

        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)
        print("TARGET PROPERTIES", target_properties)

        start = time.time()

        ident = 0
        samples = self.sample_batch(batch_id=ident, batch_size=10, num_nodes=None,
                                    save_final=10,
                                    keep_chain=1,
                                    number_chain_steps=self.number_chain_steps,
                                    input_properties=target_properties)
        print(f'Sampling took {time.time() - start:.2f} seconds\n')

        self.save_cond_samples(samples, target_properties, file_path=os.path.join(os.getcwd(), f'cond_smiles{i}.pkl'))
        # save conditional generated samples
        mae = self.cond_sample_metric(samples, target_properties)
        return {'mae': mae}

    def test_epoch_end(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        final_mae = self.cond_val.compute()
        final_validity = self.num_valid_molecules / self.num_total
        print("Final MAE", final_mae)
        print("Final validity", final_validity * 100)

        wandb.run.summary['final_MAE'] = final_mae
        wandb.run.summary['final_validity'] = final_validity
        wandb.log({'final mae': final_mae,
                   'final validity': final_validity})

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim_output)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim_output)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data


    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, number_chain_steps: int,
                     save_final: int, num_nodes=None, input_properties=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()
        # Build the masks
        arange = torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        # TODO: how to move node_mask on the right device in the multi-gpu case?
        # TODO: everything else depends on its device
        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = diffusion_utils.sample_discrete_feature_noise(limit_dist=self.limit_dist, node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            t_norm = t_array / self.T
            s_norm = s_array / self.T

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(s_norm, t_norm, X, E, y, node_mask, input_properties)
            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            write_index = (s_int * number_chain_steps) // self.T
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        print("Examples of generated graphs:")
        for i in range(min(5, X.shape[0])):
            print("E: ", E[i])
            print("X: ", X[i])

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]

            chain_X[0] = final_X_chain                  # Overwrite last frame with the resulting X, E
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            print('Visualizing chains starts!')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.args.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)
            print('\nVisualizing chains Ends!')

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)

        return molecule_list


    def cond_sample_metric(self, samples, input_properties):
        mols_dipoles = []
        mols_homo = []

        # Hardware side settings (CPU thread number and memory settings used for calculation)
        psi4.set_num_threads(nthread=4)
        psi4.set_memory("5GB")
        psi4.core.set_output_file('psi4_output.dat', False)

        for sample in samples:
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)

            try:
                Chem.SanitizeMol(mol)
            except:
                print('invalid chemistry')
                continue

            # Coarse 3D structure optimization by generating 3D structure from SMILES
            mol = Chem.AddHs(mol)
            params = ETKDGv3()
            params.randomSeed = 1
            try:
                EmbedMolecule(mol, params)
            except Chem.rdchem.AtomValenceException:
                print('invalid chemistry')
                continue

            # Structural optimization with MMFF (Merck Molecular Force Field)
            try:
                s = MMFFOptimizeMolecule(mol)
                print(s)
            except:
                print('Bad conformer ID')
                continue

            conf = mol.GetConformer()

            # Convert to a format that can be input to Psi4.
            # Set charge and spin multiplicity (below is charge 0, spin multiplicity 1)

            # Get the formal charge
            fc = 'FormalCharge'
            mol_FormalCharge = int(mol.GetProp(fc)) if mol.HasProp(fc) else Chem.GetFormalCharge(mol)

            sm = 'SpinMultiplicity'
            if mol.HasProp(sm):
                mol_spin_multiplicity = int(mol.GetProp(sm))
            else:
                # Calculate spin multiplicity using Hund's rule of maximum multiplicity...
                NumRadicalElectrons = 0
                for Atom in mol.GetAtoms():
                    NumRadicalElectrons += Atom.GetNumRadicalElectrons()
                TotalElectronicSpin = NumRadicalElectrons / 2
                SpinMultiplicity = 2 * TotalElectronicSpin + 1
                mol_spin_multiplicity = int(SpinMultiplicity)

            mol_input = "%s %s" % (mol_FormalCharge, mol_spin_multiplicity)
            print(mol_input)
            #mol_input = "0 1"

            # Describe the coordinates of each atom in XYZ format
            for atom in mol.GetAtoms():
                mol_input += "\n " + atom.GetSymbol() + " " + str(conf.GetAtomPosition(atom.GetIdx()).x) \
                             + " " + str(conf.GetAtomPosition(atom.GetIdx()).y) \
                             + " " + str(conf.GetAtomPosition(atom.GetIdx()).z)

            try:
                molecule = psi4.geometry(mol_input)
            except:
                print('Can not calculate psi4 geometry')
                continue

            # Convert to a format that can be input to pyscf
            # Set calculation method (functional) and basis set
            level = "b3lyp/6-31G*"

            # Calculation method (functional), example of basis set
            # theory = ['hf', 'b3lyp']
            # basis_set = ['sto-3g', '3-21G', '6-31G(d)', '6-31+G(d,p)', '6-311++G(2d,p)']

            # Perform structural optimization calculations
            print('Psi4 calculation starts!!!')
            #energy, wave_function = psi4.optimize(level, molecule=molecule, return_wfn=True)
            try:
                energy, wave_function = psi4.energy(level, molecule=molecule, return_wfn=True)
            except psi4.driver.SCFConvergenceError:
                print("Psi4 did not converge")
                continue

            print('Chemistry information check!!!')

            if self.args.general.guidance_target in ['mu', 'both']:
                dip_x, dip_y, dip_z = wave_function.variable('SCF DIPOLE')[0],\
                                      wave_function.variable('SCF DIPOLE')[1],\
                                      wave_function.variable('SCF DIPOLE')[2]
                dipole_moment = math.sqrt(dip_x**2 + dip_y**2 + dip_z**2) * 2.5417464519
                print("Dipole moment", dipole_moment)
                mols_dipoles.append(dipole_moment)

            if self.args.general.guidance_target in ['homo', 'both']:
                # Compute HOMO (Unit: au= Hartree）
                LUMO_idx = wave_function.nalpha()
                HOMO_idx = LUMO_idx - 1
                homo = wave_function.epsilon_a_subset("AO", "ALL").np[HOMO_idx]

                # convert unit from a.u. to ev
                homo = homo * 27.211324570273
                print("HOMO", homo)
                mols_homo.append(homo)

        num_valid_molecules = max(len(mols_dipoles), len(mols_homo))
        print("Number of valid samples", num_valid_molecules)
        self.num_valid_molecules += num_valid_molecules
        self.num_total += len(samples)

        mols_dipoles = torch.FloatTensor(mols_dipoles)
        mols_homo = torch.FloatTensor(mols_homo)

        if self.args.general.guidance_target == 'mu':
            mae = self.cond_val(mols_dipoles.unsqueeze(1),
                                input_properties.repeat(len(mols_dipoles), 1).cpu())

        elif self.args.general.guidance_target == 'homo':
            mae = self.cond_val(mols_homo.unsqueeze(1),
                                input_properties.repeat(len(mols_homo), 1).cpu())

        elif self.args.general.guidance_target == 'both':
            properties = torch.hstack((mols_dipoles.unsqueeze(1), mols_homo.unsqueeze(1)))
            mae = self.cond_val(properties,
                                input_properties.repeat(len(mols_dipoles), 1).cpu())

        print('Conditional generation metric:')
        print(f'Epoch {self.current_epoch}: MAE: {mae}')
        wandb.log({"val_epoch/conditional generation mae": mae,
                   'Valid molecules': num_valid_molecules})
        return mae

    def cond_fn(self, noisy_data, node_mask, target=None):
        #self.guidance_model.eval()
        loss = nn.MSELoss()

        t = noisy_data['t']

        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['t']

        with torch.enable_grad():
            x_in = X.float().detach().requires_grad_(True)
            e_in = E.float().detach().requires_grad_(True)

            pred = self.guidance_model.model(x_in, e_in, y, node_mask)

            # normalize target
            target = target.type_as(x_in)

            mse = loss(pred.y, target.repeat(pred.y.size(0), 1))

            t_int = int(t[0].item() * 500)
            if t_int % 10 == 0:
                print(f'Regressor MSE at step {t_int}: {mse.item()}')
            wandb.log({'Guidance MSE': mse})

            # calculate gradient of mse with respect to x and e
            grad_x = torch.autograd.grad(mse, x_in, retain_graph=True)[0]
            grad_e = torch.autograd.grad(mse, e_in)[0]

            x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
            bs, n = x_mask.shape[0], x_mask.shape[1]

            e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
            e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1
            diag_mask = torch.eye(n)
            diag_mask = ~diag_mask.type_as(e_mask1).bool()
            diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

            mask_grad_x = grad_x * x_mask
            mask_grad_e = grad_e * e_mask1 * e_mask2 * diag_mask

            mask_grad_e = 1 / 2 * (mask_grad_e + torch.transpose(mask_grad_e, 1, 2))
            return mask_grad_x, mask_grad_e

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask, input_properties):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        bs, n, dxs = X_t.shape
        beta_t = self.noise_schedule(t_normalized=t)  # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t)

        # Retrieve transitions matrix
        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, self.device)
        Qsb = self.transition_model.get_Qt_bar(alpha_s_bar, self.device)
        Qt = self.transition_model.get_Qt(beta_t, self.device)

        # Neural net predictions
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t, 'node_mask': node_mask}
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0

        p_s_and_t_given_0_X = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=X_t,
                                                                                           Qt=Qt.X,
                                                                                           Qsb=Qsb.X,
                                                                                           Qtb=Qtb.X)

        p_s_and_t_given_0_E = diffusion_utils.compute_batched_over0_posterior_distribution(X_t=E_t,
                                                                                           Qt=Qt.E,
                                                                                           Qsb=Qsb.E,
                                                                                           Qtb=Qtb.E)
        # Dim of these two tensors: bs, N, d0, d_t-1
        weighted_X = pred_X.unsqueeze(-1) * p_s_and_t_given_0_X         # bs, n, d0, d_t-1
        unnormalized_prob_X = weighted_X.sum(dim=2)                     # bs, n, d_t-1
        unnormalized_prob_X[torch.sum(unnormalized_prob_X, dim=-1) == 0] = 1e-5
        prob_X = unnormalized_prob_X / torch.sum(unnormalized_prob_X, dim=-1, keepdim=True)  # bs, n, d_t-1

        pred_E = pred_E.reshape((bs, -1, pred_E.shape[-1]))
        weighted_E = pred_E.unsqueeze(-1) * p_s_and_t_given_0_E        # bs, N, d0, d_t-1
        unnormalized_prob_E = weighted_E.sum(dim=-2)
        unnormalized_prob_E[torch.sum(unnormalized_prob_E, dim=-1) == 0] = 1e-5
        prob_E = unnormalized_prob_E / torch.sum(unnormalized_prob_E, dim=-1, keepdim=True)
        prob_E = prob_E.reshape(bs, n, n, pred_E.shape[-1])

        # # Guidance
        lamb = self.args.guidance.lambda_guidance

        grad_x, grad_e = self.cond_fn(noisy_data, node_mask, input_properties)

        p_eta_x = torch.softmax(- lamb * grad_x, dim=-1)
        p_eta_e = torch.softmax(- lamb * grad_e, dim=-1)

        prob_X_unnormalized = p_eta_x * prob_X
        prob_X_unnormalized[torch.sum(prob_X_unnormalized, dim=-1) == 0] = 1e-7
        prob_X = prob_X_unnormalized / torch.sum(prob_X_unnormalized, dim=-1, keepdim=True)

        prob_E_unnormalized = p_eta_e * prob_E
        prob_E_unnormalized[torch.sum(prob_E_unnormalized, dim=-1) == 0] = 1e-7
        prob_E = prob_E_unnormalized / torch.sum(prob_E_unnormalized, dim=-1, keepdim=True)

        assert ((prob_X.sum(dim=-1) - 1).abs() < 1e-4).all()
        assert ((prob_E.sum(dim=-1) - 1).abs() < 1e-4).all()

        sampled_s = diffusion_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

        X_s = F.one_hot(sampled_s.X, num_classes=self.Xdim_output).float()
        E_s = F.one_hot(sampled_s.E, num_classes=self.Edim_output).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=torch.zeros(y_t.shape[0], 0))

        return out_one_hot.mask(node_mask).type_as(y_t), out_discrete.mask(node_mask, collapse=True).type_as(y_t)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data['t']
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def save_cond_samples(self, samples, target, file_path):
        cond_results = {'smiles': [], 'input_targets': target}
        invalid = 0
        disconnected = 0

        print("\tConverting conditionally generated molecules to SMILES ...")
        for sample in samples:
            mol = build_molecule_with_partial_charges(sample[0], sample[1], self.dataset_info.atom_decoder)
            smile = mol2smiles(mol)
            if smile is not None:
                cond_results['smiles'].append(smile)
                mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
                if len(mol_frags) > 1:
                    print("Disconnected molecule", mol, mol_frags)
                    disconnected += 1
            else:
                print("Invalid molecule obtained.")
                invalid += 1

        print("Number of invalid molecules", invalid)
        print("Number of disconnected molecules", disconnected)

        # save samples
        with open(file_path, 'wb') as f:
            pickle.dump(cond_results, f)

        return cond_results

#qm9_regressor_decrete.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import time
import wandb
from torchmetrics import MeanSquaredError, MeanAbsoluteError

from src.models.transformer_model import GraphTransformer
from src.diffusion.noise_schedule import PredefinedNoiseScheduleDiscrete, MarginalUniformTransition
from src.diffusion import diffusion_utils
from src.metrics.abstract_metrics import NLL, SumExceptBatchKL, SumExceptBatchMetric
from src.metrics.train_metrics import TrainLossDiscrete
import src.utils as utils


def reset_metrics(metrics):
    for metric in metrics:
        metric.reset()


class Qm9RegressorDiscrete(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features,
                 domain_features):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.args = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.num_classes = dataset_infos.num_classes
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.val_nll = NLL()
        self.val_X_kl = SumExceptBatchKL()
        self.val_E_kl = SumExceptBatchKL()
        self.val_y_kl = SumExceptBatchKL()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMetric()

        self.test_nll = NLL()
        self.test_X_kl = SumExceptBatchKL()
        self.test_E_kl = SumExceptBatchKL()
        self.test_y_kl = SumExceptBatchKL()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMetric()

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.noise_schedule = PredefinedNoiseScheduleDiscrete(cfg.model.diffusion_noise_schedule,
                                                              timesteps=cfg.model.diffusion_steps)

        # Marginal transition model
        node_types = self.dataset_info.node_types.float()
        x_marginals = node_types / torch.sum(node_types)

        edge_types = self.dataset_info.edge_types.float()
        e_marginals = edge_types / torch.sum(edge_types)
        print(f"Marginal distribution of the classes: {x_marginals} for nodes, {e_marginals} for edges")
        self.transition_model = MarginalUniformTransition(x_marginals=x_marginals, e_marginals=e_marginals,
                                                          y_classes=self.ydim_output)

        self.limit_dist = utils.PlaceHolder(X=x_marginals, E=e_marginals,
                                            y=torch.ones(self.ydim_output) / self.ydim_output)

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

        self.train_loss = MeanSquaredError(squared=True)
        self.val_loss = MeanAbsoluteError()
        self.test_loss = MeanAbsoluteError()
        self.best_val_mae = 1e8

        self.val_loss_each = [MeanAbsoluteError().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for i
                              in range(2)]
        self.test_loss_each = [MeanAbsoluteError().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) for
                               i in range(2)]
        self.target_dict = {0: "mu", 1: "homo"}
        self.lr_scheduler = None

    def training_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        mse = self.compute_train_loss(pred, target, log=i % self.log_every_steps == 0)
        return {'loss': mse}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.train.lr, amsgrad=True, weight_decay=1e-12)
        self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "frequency": 1,
                # "monitor": "val_loss",
            },
        }
    def lr_scheduler_step(self, epoch, *args, **kwargs):
        # Override this method for custom LR scheduler step logic
        if self.lr_scheduler is not None:
            # Instead of calling self.lr_scheduler.step(epoch), use the following:
            self.lr_scheduler.step()


    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        print("Size of the input features", self.Xdim, self.Edim, self.ydim)

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        train_mse = self.train_loss.compute()

        to_log = {"train_epoch/mse": train_mse}
        print(f"Epoch {self.current_epoch}: train_mse: {train_mse :.3f} -- {time.time() - self.start_epoch_time:.1f}s ")

        wandb.log(to_log)
        self.train_loss.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_loss.reset()
        reset_metrics(self.val_loss_each)

    def validation_step(self, data, i):
        # input zero y to generate noised graphs
        target = data.y.clone()
        data.y = torch.zeros(data.y.shape[0], 0).type_as(data.y)

        dense_data, node_mask = utils.to_dense(data.x, data.edge_index, data.edge_attr, data.batch)
        dense_data = dense_data.mask(node_mask)
        noisy_data = self.apply_noise(dense_data.X, dense_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        mae = self.compute_val_loss(pred, target)
        # self.log('val_loss', mae, prog_bar=True, on_step=False, on_epoch=True)
        return {'val_loss': mae}

    def validation_epoch_end(self, outs) -> None:
        val_mae = self.val_loss.compute()
        to_log = {"val/epoch_mae": val_mae}
        print(f"Epoch {self.current_epoch}: val_mae: {val_mae :.3f}")
        wandb.log(to_log)
        self.log('val/epoch_mae', val_mae, on_epoch=True, on_step=False)

        if val_mae < self.best_val_mae:
            self.best_val_mae = val_mae
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_mae, self.best_val_mae))

        if self.args.general.guidance_target == 'both':
            print('Val loss each target:')
            for i in range(2):
                mae_each = self.val_loss_each[i].compute()
                print(f"Target {self.target_dict[i]}: val_mae: {mae_each :.3f}")
                to_log_each = {f"val_epoch/{self.target_dict[i]}_mae": mae_each}
                wandb.log(to_log_each)

        self.val_loss.reset()
        reset_metrics(self.val_loss_each)

    def on_test_epoch_start(self) -> None:
        self.test_loss.reset()
        reset_metrics(self.test_loss_each)

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """

        # Sample a timestep t.
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1), device=X.device).float()  # (bs, 1)
        s_int = t_int - 1

        t_float = t_int / self.T
        s_float = s_int / self.T

        # beta_t and alpha_s_bar are used for denoising/loss computation
        beta_t = self.noise_schedule(t_normalized=t_float)                         # (bs, 1)
        alpha_s_bar = self.noise_schedule.get_alpha_bar(t_normalized=s_float)      # (bs, 1)
        alpha_t_bar = self.noise_schedule.get_alpha_bar(t_normalized=t_float)      # (bs, 1)

        Qtb = self.transition_model.get_Qt_bar(alpha_t_bar, device=self.device)  # (bs, dx_in, dx_out), (bs, de_in, de_out)
        assert (abs(Qtb.X.sum(dim=2) - 1.) < 1e-4).all(), Qtb.X.sum(dim=2) - 1
        assert (abs(Qtb.E.sum(dim=2) - 1.) < 1e-4).all()

        # Compute transition probabilities
        probX = X @ Qtb.X  # (bs, n, dx_out)
        probE = E @ Qtb.E.unsqueeze(1)  # (bs, n, n, de_out)

        sampled_t = diffusion_utils.sample_discrete_features(probX=probX, probE=probE, node_mask=node_mask)

        X_t = F.one_hot(sampled_t.X, num_classes=self.Xdim)
        E_t = F.one_hot(sampled_t.E, num_classes=self.Edim)
        assert (X.shape == X_t.shape) and (E.shape == E_t.shape)

        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {'t_int': t_int, 't': t_float, 'beta_t': beta_t, 'alpha_s_bar': alpha_s_bar,
                      'alpha_t_bar': alpha_t_bar, 'X_t': z_t.X, 'E_t': z_t.E, 'y_t': z_t.y, 'node_mask': node_mask}
        return noisy_data

    def compute_val_loss(self, pred, target):
        """Computes MAE.
           pred: (batch_size, n, total_features)
           target: (batch_size, total_features)
           noisy_data: dict
           X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
           node_mask : (bs, n)
           Output: nll (size 1)
       """

        for i in range(pred.y.shape[1]):
            mae_each = self.val_loss_each[i](pred.y[:, i], target[:, i])

        mae = self.val_loss(pred.y, target)
        return mae

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data['y_t'], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """

        extra_features = self.extra_features(noisy_data)
        extra_molecular_features = self.domain_features(noisy_data)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)

        t = noisy_data['t']

        assert extra_X.shape[-1] == 0, 'The regressor model should not be used with extra features'
        assert extra_E.shape[-1] == 0, 'The regressor model should not be used with extra features'
        return utils.PlaceHolder(X=extra_X, E=extra_E, y=t)

    def compute_train_loss(self, pred, target, log: bool):
        """
           pred: (batch_size, n, total_features)
               pred_epsX: bs, n, dx
               pred_epsy: bs, n, n, dy
               pred_eps_z: bs, dz
           data: dict
           noisy_data: dict
           Output: mse (size 1)
       """
        mse = self.train_loss(pred.y, target)

        if log:
            wandb.log({"train_loss/batch_mse": mse.item()}, commit=True)
        return mse

#extra_features.py
import torch
# from torch_geometric.transforms import largest_connected_components
from src import utils


class DummyExtraFeatures:
    def __init__(self):
        """ This class does not compute anything, just returns empty tensors."""

    def __call__(self, noisy_data):
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        y = noisy_data['y_t']
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.PlaceHolder(X=empty_x, E=empty_e, y=empty_y)


class ExtraFeatures:
    def __init__(self, extra_features_type, dataset_info):
        self.max_n_nodes = dataset_info.max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ['eigenvalues', 'all']:
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)

    def __call__(self, noisy_data):
        n = noisy_data['node_mask'].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        x_cycles, y_cycles = self.ncycles(noisy_data)       # (bs, n_cycles)

        if self.features_type == 'cycles':
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles)))

        elif self.features_type == 'eigenvalues':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues = eigenfeatures   # (bs, 1), (bs, 10)
            return utils.PlaceHolder(X=x_cycles, E=extra_edge_attr, y=torch.hstack((n, y_cycles, n_components,
                                                                                    batched_eigenvalues)))
        elif self.features_type == 'all':
            eigenfeatures = self.eigenfeatures(noisy_data)
            E = noisy_data['E_t']
            extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
            n_components, batched_eigenvalues, nonlcc_indicator, k_lowest_eigvec = eigenfeatures   # (bs, 1), (bs, 10),
                                                                                                # (bs, n, 1), (bs, n, 2)

            return utils.PlaceHolder(X=torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                                     E=extra_edge_attr,
                                     y=torch.hstack((n, y_cycles, n_components, batched_eigenvalues)))
        else:
            raise ValueError(f"Features type {self.features_type} not implemented")


class NodeCycleFeatures:
    def __init__(self):
        self.kcycles = KNodeCycles()

    def __call__(self, noisy_data):
        adj_matrix = noisy_data['E_t'][..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)   # (bs, n_cycles)
        x_cycles = x_cycles.type_as(adj_matrix) * noisy_data['node_mask'].unsqueeze(-1)
        # Avoid large values when the graph is dense
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles[x_cycles > 1] = 1
        y_cycles[y_cycles > 1] = 1
        return x_cycles, y_cycles


class EigenFeatures:
    """
    Code taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py
    """
    def __init__(self, mode):
        """ mode: 'eigenvalues' or 'all' """
        self.mode = mode

    def __call__(self, noisy_data):
        E_t = noisy_data['E_t']
        mask = noisy_data['node_mask']
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        if self.mode == 'eigenvalues':
            eigvals = torch.linalg.eigvalsh(L)        # bs, n
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        elif self.mode == 'all':
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
            # Retrieve eigenvalues features
            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigenvalues=eigvals)

            # Retrieve eigenvectors features
            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(vectors=eigvectors,
                                                                               node_mask=noisy_data['node_mask'],
                                                                               n_connected=n_connected_comp)
            return n_connected_comp, batch_eigenvalues, nonlcc_indicator, k_lowest_eigenvector
        else:
            raise NotImplementedError(f"Mode {self.mode} is not implemented")


def compute_laplacian(adjacency, normalize: bool):
    """
    adjacency : batched adjacency matrix (bs, n, n)
    normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
    Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    diag = torch.sum(adjacency, dim=-1)     # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)      # Degree matrix      # (bs, n, n)
    combinatorial = D - adjacency                        # (bs, n, n)

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)            # (bs, n)
    D_norm = torch.diag_embed(diag_norm)        # (bs, n, n)
    L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues, k=5):
    """
    values : eigenvalues -- (bs, n)
    node_mask: (bs, n)
    k: num of non zero eigenvalues to keep
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = max(n_connected_components) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack((eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues)))
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(0) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(vectors, node_mask, n_connected, k=2):
    """
    vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
    returns:
        not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
        k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
    Warning: this function does not exactly return what is desired, the lcc might not be exactly the returned vector.
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Create an indicator for the nodes outside the largest connected components
    k0 = min(n, 5)
    first_evs = vectors[:, :, :k0]                         # bs, n, k0
    quantized = torch.round(first_evs * 1000) / 1000       # bs, n, k0
    random_mask = (50 * torch.ones((bs, n, k0)).type_as(vectors)) * (~node_mask.unsqueeze(-1))         # bs, n, k0
    min_batched = torch.min(quantized + random_mask, dim=1).values.unsqueeze(1)       # bs, 1, k0
    max_batched = torch.max(quantized - random_mask, dim=1).values.unsqueeze(1)       # bs, 1, k0
    nonzero_mask = quantized.abs() >= 1e-5
    is_min = (quantized == min_batched) * nonzero_mask * node_mask.unsqueeze(2)                      # bs, n, k0
    is_max = (quantized == max_batched) * nonzero_mask * node_mask.unsqueeze(2)                      # bs, n, k0
    is_other = (quantized != min_batched) * (quantized != max_batched) * nonzero_mask * node_mask.unsqueeze(2)

    all_masks = torch.cat((is_min.unsqueeze(-1), is_max.unsqueeze(-1), is_other.unsqueeze(-1)), dim=3)    # bs, n, k0, 3
    all_masks = all_masks.flatten(start_dim=-2)      # bs, n, k0 x 3
    counts = torch.sum(all_masks, dim=1)      # bs, k0 x 3

    argmax_counts = torch.argmax(counts, dim=1)       # bs
    lcc_indicator = all_masks[torch.arange(bs), :, argmax_counts]                   # bs, n
    not_lcc_indicator = ((~lcc_indicator).float() * node_mask).unsqueeze(2)

    # Get the eigenvectors corresponding to the first nonzero eigenvalues
    to_extend = max(n_connected) + k - n
    if to_extend > 0:
        vectors = torch.cat((vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2)   # bs, n , n + to_extend
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(0) + n_connected.unsqueeze(2)    # bs, 1, k
    indices = indices.expand(-1, n, -1)                                               # bs, n, k
    first_k_ev = torch.gather(vectors, dim=2, index=indices)       # bs, n, k
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class KNodeCycles:
    """ Builds cycle counts for each node in a graph.
    """

    def __init__(self):
        super().__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """ tr(A ** 3). """
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(-1).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = diag_a4 - self.d * (self.d - 1) - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(-1).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)

        c5 = diag_a5 - 2 * triangles * self.d - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1) + triangles
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(-1).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix ** 2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (term_1_t - 3 * term_2_t + 9 * term3_t - 6 * term_4_t + 6 * term_5_t - 4 * term_6_t + 4 * term_7_t +
                3 * term8_t - 12 * term9_t + 4 * term10_t)
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy
    
#extra features molecular.py
import torch
from src import utils


class ExtraMolecularFeatures:
    def __init__(self, dataset_infos):
        self.charge = ChargeFeature(remove_h=dataset_infos.remove_h, valencies=dataset_infos.valencies)
        self.valency = ValencyFeature()
        self.weight = WeightFeature(max_weight=dataset_infos.max_weight, atom_weights=dataset_infos.atom_weights)

    def __call__(self, noisy_data):
        charge = self.charge(noisy_data).unsqueeze(-1)      # (bs, n, 1)
        valency = self.valency(noisy_data).unsqueeze(-1)    # (bs, n, 1)
        weight = self.weight(noisy_data)                    # (bs, 1)

        extra_edge_attr = torch.zeros((*noisy_data['E_t'].shape[:-1], 0)).type_as(noisy_data['E_t'])

        return utils.PlaceHolder(X=torch.cat((charge, valency), dim=-1), E=extra_edge_attr, y=weight)


class ChargeFeature:
    def __init__(self, remove_h, valencies):
        self.remove_h = remove_h
        self.valencies = valencies

    def __call__(self, noisy_data):
        bond_orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        weighted_E = noisy_data['E_t'] * bond_orders      # (bs, n, n, de)
        current_valencies = weighted_E.argmax(dim=-1).sum(dim=-1)   # (bs, n)

        valencies = torch.tensor(self.valencies, device=noisy_data['X_t'].device).reshape(1, 1, -1)
        X = noisy_data['X_t'] * valencies  # (bs, n, dx)
        normal_valencies = torch.argmax(X, dim=-1)               # (bs, n)

        return (normal_valencies - current_valencies).type_as(noisy_data['X_t'])


class ValencyFeature:
    def __init__(self):
        pass

    def __call__(self, noisy_data):
        orders = torch.tensor([0, 1, 2, 3, 1.5], device=noisy_data['E_t'].device).reshape(1, 1, 1, -1)
        E = noisy_data['E_t'] * orders      # (bs, n, n, de)
        valencies = E.argmax(dim=-1).sum(dim=-1)    # (bs, n)
        return valencies.type_as(noisy_data['X_t'])


class WeightFeature:
    def __init__(self, max_weight, atom_weights):
        self.max_weight = max_weight
        self.atom_weight_list = torch.Tensor(list(atom_weights.values()))

    def __call__(self, noisy_data):
        X = torch.argmax(noisy_data['X_t'], dim=-1)     # (bs, n)
        X_weights = self.atom_weight_list[X]            # (bs, n)
        return X_weights.sum(dim=-1).unsqueeze(-1).type_as(noisy_data['X_t']) / self.max_weight     # (bs, 1)

#diffusion model
import os
import time

import torch
import torch.nn as nn
import numpy as np
import pytorch_lightning as pl
import wandb

from src.models.transformer_model import GraphTransformer
from diffusion.noise_schedule import PredefinedNoiseSchedule
from src.diffusion import diffusion_utils
from src.metrics.train_metrics import TrainLoss
from src.metrics.abstract_metrics import SumExceptBatchMetric, SumExceptBatchMSE, NLL
from src import utils


class LiftedDenoisingDiffusion(pl.LightningModule):
    def __init__(self, cfg, dataset_infos, train_metrics, sampling_metrics, visualization_tools, extra_features=None,
                 domain_features=None):
        super().__init__()

        input_dims = dataset_infos.input_dims
        output_dims = dataset_infos.output_dims
        nodes_dist = dataset_infos.nodes_dist

        self.norm_values = cfg.model.normalize_factors
        self.norm_biases = cfg.model.norm_biases
        self.gamma = PredefinedNoiseSchedule(cfg.model.diffusion_noise_schedule, timesteps=cfg.model.diffusion_steps)
        diffusion_utils.check_issues_norm_values(self.gamma, self.norm_values[1], self.norm_values[2])

        self.cfg = cfg
        self.name = cfg.general.name
        self.model_dtype = torch.float32
        self.T = cfg.model.diffusion_steps

        self.Xdim = input_dims['X']
        self.Edim = input_dims['E']
        self.ydim = input_dims['y']
        self.Xdim_output = output_dims['X']
        self.Edim_output = output_dims['E']
        self.ydim_output = output_dims['y']
        self.node_dist = nodes_dist

        self.dataset_info = dataset_infos

        self.val_nll = NLL()
        self.val_X_mse = SumExceptBatchMSE()
        self.val_E_mse = SumExceptBatchMSE()
        self.val_y_mse = SumExceptBatchMSE()
        self.val_X_logp = SumExceptBatchMetric()
        self.val_E_logp = SumExceptBatchMetric()
        self.val_y_logp = SumExceptBatchMSE()

        self.test_nll = NLL()
        self.test_X_mse = SumExceptBatchMSE()
        self.test_E_mse = SumExceptBatchMSE()
        self.test_y_mse = SumExceptBatchMSE()
        self.test_X_logp = SumExceptBatchMetric()
        self.test_E_logp = SumExceptBatchMetric()
        self.test_y_logp = SumExceptBatchMSE()

        self.train_loss = TrainLoss()
        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics
        self.visualization_tools = visualization_tools

        self.save_hyperparameters(ignore=[train_metrics, sampling_metrics])
        self.visualization_tools = visualization_tools

        self.model = GraphTransformer(n_layers=cfg.model.n_layers,
                                      input_dims=input_dims,
                                      hidden_mlp_dims=cfg.model.hidden_mlp_dims,
                                      hidden_dims=cfg.model.hidden_dims,
                                      output_dims=output_dims,
                                      act_fn_in=nn.ReLU(),
                                      act_fn_out=nn.ReLU())

        self.save_hyperparameters()

        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.best_val_nll = 1e8
        self.val_counter = 0

    def training_step(self, data, i):
        dense_data, node_mask = utils.to_dense(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, normalized_data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # TODO: change noisy data
        mse = self.train_loss(masked_pred_epsX=pred.X,
                              masked_pred_epsE=pred.E,
                              pred_y=pred.y,
                              true_epsX=noisy_data['epsX'],
                              true_epsE=noisy_data['epsE'],
                              true_y=noisy_data['epsy'],
                              log=i % self.log_every_steps == 0)

        self.train_metrics(masked_pred_epsX=pred.X,
                           masked_pred_epsE=pred.E,
                           pred_y=pred.y,
                           true_epsX=noisy_data['epsX'],
                           true_epsE=noisy_data['epsE'],
                           true_y=noisy_data['epsy'], log=i % self.log_every_steps == 0)

        return {'loss': mse}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.cfg.train.lr, amsgrad=True,
                                 weight_decay=self.cfg.train.weight_decay)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())

    def on_train_epoch_start(self) -> None:
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        self.train_loss.log_epoch_metrics(self.current_epoch, self.start_epoch_time)
        self.train_metrics.log_epoch_metrics(self.current_epoch)

    def on_validation_epoch_start(self) -> None:
        self.val_nll.reset()
        self.val_X_mse.reset()
        self.val_E_mse.reset()
        self.val_y_mse.reset()
        self.val_X_logp.reset()
        self.val_E_logp.reset()
        self.val_y_logp.reset()

    def validation_step(self, data, i):
        dense_data, node_mask = utils.to_dense(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        # TODO: check if compute val loss should be called on the normalized data or not
        nll = self.compute_val_loss(pred, noisy_data, normalized_data.X, normalized_data.E, normalized_data.y,
                                    node_mask, test=False)
        return {'loss': nll}

    def validation_epoch_end(self, outs) -> None:
        metrics = [self.val_nll.compute(), self.val_X_mse.compute(), self.val_E_mse.compute(),
                   self.val_y_mse.compute(), self.val_X_logp.compute(), self.val_E_logp.compute(),
                   self.val_y_logp.compute()]
        wandb.log({"val/epoch_NLL": metrics[0],
                   "val/X_mse": metrics[1],
                   "val/E_mse": metrics[2],
                   "val/y_mse": metrics[3],
                   "val/X_logp": metrics[4],
                   "val/E_logp": metrics[5],
                   "val/y_logp": metrics[6]}, commit=False)

        print(f"Epoch {self.current_epoch}: Val NLL {metrics[0] :.2f} -- Val Atom type MSE {metrics[1] :.2f} -- ",
              f"Val Edge type MSE: {metrics[2] :.2f} -- Val Global feat. MSE {metrics[3] :.2f}",
              f"-- Val X Reconstruction loss {metrics[4] :.2f} -- Val E Reconstruction loss {metrics[5] :.2f}",
              f"-- Val y Reconstruction loss {metrics[6] : .2f}\n")

        # Log val nll with default Lightning logger, so it can be monitored by checkpoint callback
        val_nll = metrics[0]
        self.log("val/epoch_NLL", val_nll, sync_dist=True)
        wandb.log(self.log_info(), commit=False)

        if val_nll < self.best_val_nll:
            self.best_val_nll = val_nll
        print('Val loss: %.4f \t Best val loss:  %.4f\n' % (val_nll, self.best_val_nll))

        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            start = time.time()
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

            samples = []

            ident = 0
            while samples_left_to_generate > 0:
                bs = 2 * self.cfg.train.batch_size
                to_generate = min(samples_left_to_generate, bs)
                to_save = min(samples_left_to_save, bs)
                chains_save = min(chains_left_to_save, bs)
                samples.extend(self.sample_batch(batch_id=ident,
                                                 batch_size=to_generate,
                                                 num_nodes=None, save_final=to_save,
                                                 keep_chain=chains_save,
                                                 number_chain_steps=self.number_chain_steps))
                ident += to_generate

                samples_left_to_save -= to_save
                samples_left_to_generate -= to_generate
                chains_left_to_save -= chains_save

            self.sampling_metrics(samples, self.name, self.current_epoch, val_counter=-1, test=False)
            print(f'Sampling took {time.time() - start:.2f} seconds\n')
            self.sampling_metrics.reset()

    def on_test_epoch_start(self) -> None:
        self.test_nll.reset()
        self.test_X_mse.reset()
        self.test_E_mse.reset()
        self.test_y_mse.reset()
        self.test_X_logp.reset()
        self.test_E_logp.reset()
        self.test_y_logp.reset()

    def test_step(self, data, i):
        dense_data, node_mask = utils.to_dense(x=data.x, edge_index=data.edge_index, edge_attr=data.edge_attr,
                                               batch=data.batch)
        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        normalized_data = utils.normalize(X, E, data.y, self.norm_values, self.norm_biases, node_mask)
        noisy_data = self.apply_noise(normalized_data.X, normalized_data.E, normalized_data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        nll = self.compute_val_loss(pred, noisy_data, normalized_data.X, normalized_data.E,
                                    normalized_data.y, node_mask, test=True)
        return {'loss': nll}

    def test_epoch_end(self, outs) -> None:
        """ Measure likelihood on a test set and compute stability metrics. """
        metrics = [self.test_nll.compute(), self.test_X_mse.compute(), self.test_E_mse.compute(),
                   self.test_y_mse.compute(), self.test_X_logp.compute(), self.test_E_logp.compute(),
                   self.test_y_logp.compute()]
        log_dict={"test/epoch_NLL": metrics[0],
         "test/X_mse": metrics[1],
         "test/E_mse": metrics[2],
         "test/y_mse": metrics[3],
         "test/X_logp": metrics[4],
         "test/E_logp": metrics[5],
         "test/y_logp": metrics[6]}
        wandb.log(log_dict, commit = False)

        print(f"Epoch {self.current_epoch}: Test NLL {metrics[0] :.2f} -- Test Atom type MSE {metrics[1] :.2f} -- ",
              f"Test Edge type MSE: {metrics[2] :.2f} -- Test Global feat. MSE {metrics[3] :.2f}",
              f"-- Test X Reconstruction loss {metrics[4] :.2f} -- Test E Reconstruction loss {metrics[5] :.2f}",
              f"-- Test y Reconstruction loss {metrics[6] : .2f}\n")

        test_nll = metrics[0]
        wandb.log({"test/epoch_NLL": test_nll}, commit=False)
        wandb.log(self.log_info(), commit=False)

        print(f'Test loss: {test_nll :.4f}')

        samples_left_to_generate = self.cfg.general.final_model_samples_to_generate
        samples_left_to_save = self.cfg.general.final_model_samples_to_save
        chains_left_to_save = self.cfg.general.final_model_chains_to_save

        samples = []
        id = 0
        while samples_left_to_generate > 0:
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            samples.extend(self.sample_batch(id, to_generate, num_nodes=None, save_final=to_save,
                                             keep_chain=chains_save, number_chain_steps=self.number_chain_steps))
            id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        self.sampling_metrics.reset()
        self.sampling_metrics(samples, self.name, self.current_epoch, self.val_counter, test=True)
        self.sampling_metrics.reset()

    def kl_prior(self, X, E, y, node_mask):
        """Computes the KL between q(z1 | x) and the prior p(z1) = Normal(0, 1).

        This is essentially a lot of work for something that is in practice negligible in the loss. However, you
        compute it so that you see it when you've made a mistake in your noise schedule.
        """
        # Compute the last alpha value, alpha_T.
        ones = torch.ones((X.size(0), 1))
        ones = ones.type_as(X)
        gamma_T = self.gamma(ones)
        alpha_T = diffusion_utils.alpha(gamma_T, X.size())

        # Compute means.
        mu_T_X = alpha_T * X
        mu_T_E = alpha_T.unsqueeze(1) * E
        mu_T_y = alpha_T.squeeze(1) * y

        # Compute standard deviations (only batch axis for x-part, inflated for h-part).
        sigma_T_X = diffusion_utils.sigma(gamma_T, mu_T_X.size())
        sigma_T_E = diffusion_utils.sigma(gamma_T, mu_T_E.size())
        sigma_T_y = diffusion_utils.sigma(gamma_T, mu_T_y.size())

        # Compute KL for h-part.
        kl_distance_X = diffusion_utils.gaussian_KL(mu_T_X, sigma_T_X)
        kl_distance_E = diffusion_utils.gaussian_KL(mu_T_E, sigma_T_E)
        kl_distance_y = diffusion_utils.gaussian_KL(mu_T_y, sigma_T_y)

        return kl_distance_X + kl_distance_E + kl_distance_y

    def log_constants_p_y_given_z0(self, batch_size):
        """ Computes p(y|z0)= -0.5 ydim (log(2pi) + gamma(0)).
            sigma_y = sqrt(sigma_0^2 / alpha_0^2) = SNR(-0.5 gamma_0).
            output size: (batch_size)
        """
        if self.ydim_output == 0:
            return 0.0

        zeros = torch.zeros((batch_size, 1))
        gamma_0 = self.gamma(zeros).squeeze(1)
        # Recall that
        return -0.5 * self.ydim * (gamma_0 + np.log(2 * np.pi))

    def reconstruction_logp(self, data, data_0, gamma_0, eps, pred_0, node_mask, epsilon=1e-10, test=False):
        """ Reconstruction loss.
            output size: (1).
        """
        X, E, y = data.values()
        X_0, E_0, y_0 = data_0.values()

        # TODO: why don't we need the values of X and E?
        _, _, eps_y0 = eps.values()
        predy = pred_0.y

        # 1. Compute reconstruction loss for global, continuous features
        if test:
            error_y = -0.5 * self.test_y_logp(predy, eps_y0)
        else:
            error_y = -0.5 * self.val_y_logp(predy, eps_y0)
        # The _constants_ depending on sigma_0 from the cross entropy term E_q(z0 | y) [log p(y | z0)].
        neg_log_constants = - self.log_constants_p_y_given_z0(y.shape[0])
        log_py = error_y + neg_log_constants

        # 2. Compute reconstruction loss for integer/categorical features on nodes and edges

        # Compute sigma_0 and rescale to the integer scale of the data_utils.
        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=X_0.size())
        sigma_0_X = sigma_0 * self.norm_values[0]
        sigma_0_E = (sigma_0 * self.norm_values[1]).unsqueeze(-1)

        # Unnormalize features
        unnormalized_data = utils.unnormalize(X, E, y, self.norm_values, self.norm_biases, node_mask, collapse=False)
        unnormalized_0 = utils.unnormalize(X_0, E_0, y_0, self.norm_values, self.norm_biases, node_mask, collapse=False)
        X_0, E_0, _ = unnormalized_0.X, unnormalized_0.E, unnormalized_0.y
        assert unnormalized_data.X.size() == X_0.size()

        # Centered cat features around 1, since onehot encoded.
        E_0_centered = E_0 - 1
        X_0_centered = X_0 - 1

        # Compute integrals from 0.5 to 1.5 of the normal distribution
        log_pE_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((E_0_centered + 0.5) / sigma_0_E)
            - diffusion_utils.cdf_std_gaussian((E_0_centered - 0.5) / sigma_0_E)
            + epsilon)

        log_pX_proportional = torch.log(
            diffusion_utils.cdf_std_gaussian((X_0_centered + 0.5) / sigma_0_X)
            - diffusion_utils.cdf_std_gaussian((X_0_centered - 0.5) / sigma_0_X)
            + epsilon)

        # Normalize the distributions over the categories.
        norm_cst_E = torch.logsumexp(log_pE_proportional, dim=-1, keepdim=True)
        norm_cst_X = torch.logsumexp(log_pX_proportional, dim=-1, keepdim=True)

        log_probabilities_E = log_pE_proportional - norm_cst_E
        log_probabilities_X = log_pX_proportional - norm_cst_X

        # Select the log_prob of the current category using the one-hot representation.
        logps = utils.PlaceHolder(X=log_probabilities_X * unnormalized_data.X,
                                  E=log_probabilities_E * unnormalized_data.E,
                                  y=None).mask(node_mask)

        if test:
            log_pE = - self.test_E_logp(-logps.E)
            log_pX = - self.test_X_logp(-logps.X)
        else:
            log_pE = - self.val_E_logp(-logps.E)
            log_pX = - self.val_X_logp(-logps.X)
        return log_pE + log_pX + log_py

    def apply_noise(self, X, E, y, node_mask):
        """ Sample noise and apply it to the data. """
        # When evaluating, the loss for t=0 is computed separately
        lowest_t = 0 if self.training else 1

        # Sample a timestep t.
        t_int = torch.randint(lowest_t, self.T + 1, size=(X.size(0), 1))
        t_int = t_int.type_as(X).float()  # (bs, 1)
        s_int = t_int - 1

        # Normalize t to [0, 1]. Note that the negative
        # step of s will never be used, since then p(x | z0) is computed.
        s_normalized = s_int / self.T
        t_normalized = t_int / self.T

        # Compute gamma_s and gamma_t via the network.
        gamma_s = diffusion_utils.inflate_batch_array(self.gamma(s_normalized), X.size())    # (bs, 1, 1),
        gamma_t = diffusion_utils.inflate_batch_array(self.gamma(t_normalized), X.size())    # (bs, 1, 1)

        # Compute alpha_t and sigma_t from gamma, with correct size for X, E and z
        alpha_t = diffusion_utils.alpha(gamma_t, X.size())                        # (bs, 1, ..., 1), same n_dims than X
        sigma_t = diffusion_utils.sigma(gamma_t, X.size())                        # (bs, 1, ..., 1), same n_dims than X

        # Sample zt ~ Normal(alpha_t x, sigma_t)
        eps = diffusion_utils.sample_feature_noise(X.size(), E.size(), y.size(), node_mask).type_as(X)

        # Sample z_t given x, h for timestep t, from q(z_t | x, h)
        X_t = alpha_t * X + sigma_t * eps.X
        E_t = alpha_t.unsqueeze(1) * E + sigma_t.unsqueeze(1) * eps.E
        y_t = alpha_t.squeeze(1) * y + sigma_t.squeeze(1) * eps.y

        noisy_data = {'t': t_normalized, 's': s_normalized, 'gamma_t': gamma_t, 'gamma_s': gamma_s,
                      'epsX': eps.X, 'epsE': eps.E, 'epsy': eps.y,
                      'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't_int': t_int}

        return noisy_data

    def compute_val_loss(self, pred, noisy_data, X, E, y, node_mask, test=False):
        """ Computes an estimator for the variational lower bound, or the simple loss (MSE).
               pred: (batch_size, n, total_features)
               noisy_data: dict
               X, E, y : (bs, n, dx),  (bs, n, n, de), (bs, dy)
               node_mask : (bs, n)
           Output: nll (size 1). """
           
        s = noisy_data['s']
        gamma_s = noisy_data['gamma_s']     # gamma_s.size() == X.size()
        gamma_t = noisy_data['gamma_t']
        epsX = noisy_data['epsX']
        epsE = noisy_data['epsE']
        epsy = noisy_data['epsy']
        X_t = noisy_data['X_t']
        E_t = noisy_data['E_t']
        y_t = noisy_data['y_t']

        # 1.
        N = node_mask.sum(1).long()
        log_pN = self.node_dist.log_prob(N)

        # 2. The KL between q(z_T | x) and p(z_T) = Normal(0, 1). Should be close to zero. Do not forget the prefactor
        kl_prior_without_prefactor = self.kl_prior(X, E, y, node_mask)

        delta_log_py = -self.ydim_output * np.log(self.norm_values[2])
        delta_log_px = -self.Xdim_output * N * np.log(self.norm_values[0])
        delta_log_pE = -self.Edim_output * 0.5 * N * (N-1) * np.log(self.norm_values[1])
        kl_prior = kl_prior_without_prefactor - delta_log_px - delta_log_py - delta_log_pE

        # 3. Diffusion loss

        # Compute weighting with SNR: (1 - SNR(s-t)) for epsilon parametrization.
        SNR_weight = - (1 - diffusion_utils.SNR(gamma_s - gamma_t))
        sqrt_SNR_weight = torch.sqrt(SNR_weight)            # same n_dims than X
        # Compute the error.
        weighted_predX_diffusion = sqrt_SNR_weight * pred.X
        weighted_epsX_diffusion = sqrt_SNR_weight * epsX

        weighted_predE_diffusion = sqrt_SNR_weight.unsqueeze(1) * pred.E
        weighted_epsE_diffusion = sqrt_SNR_weight.unsqueeze(1) * epsE

        weighted_predy_diffusion = sqrt_SNR_weight.squeeze(1) * pred.y
        weighted_epsy_diffusion = sqrt_SNR_weight.squeeze(1) * epsy

        # Compute the MSE summed over channels
        if test:
            diffusion_error = (self.test_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion) +
                               self.test_E_mse(weighted_predE_diffusion, weighted_epsE_diffusion) +
                               self.test_y_mse(weighted_predy_diffusion, weighted_epsy_diffusion))
        else:
            diffusion_error = (self.val_X_mse(weighted_predX_diffusion, weighted_epsX_diffusion) +
                               self.val_E_mse(weighted_predE_diffusion, weighted_epsE_diffusion) +
                               self.val_y_mse(weighted_predy_diffusion, weighted_epsy_diffusion))
        loss_all_t = 0.5 * self.T * diffusion_error           # t=0 is not included here.

        # 4. Compute L0 term : -log p (X, E, y | z_0) = reconstruction loss
        # Compute noise values for t = 0.
        t_zeros = torch.zeros_like(s)                                                       # bs, 1
        gamma_0 = diffusion_utils.inflate_batch_array(self.gamma(t_zeros), X_t.size())      # bs, 1, 1
        alpha_0 = diffusion_utils.alpha(gamma_0, X_t.size())                                # bs, 1, 1
        sigma_0 = diffusion_utils.sigma(gamma_0, X_t.size())                                # bs, 1, 1

        # Sample z_0 given X, E, y for timestep t, from q(z_t | X, E, y)
        eps0 = diffusion_utils.sample_feature_noise(X_t.size(), E_t.size(), y_t.size(), node_mask).type_as(X_t)

        X_0 = alpha_0 * X_t + sigma_0 * eps0.X
        E_0 = alpha_0.unsqueeze(1) * E_t + sigma_0.unsqueeze(1) * eps0.E
        y_0 = alpha_0.squeeze(1) * y_t + sigma_0.squeeze(1) * eps0.y

        noisy_data0 = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': t_zeros}
        extra_data = self.compute_extra_data(noisy_data)
        pred_0 = self.forward(noisy_data0, extra_data, node_mask)

        loss_term_0 = - self.reconstruction_logp(data={'X': X, 'E': E, 'y': y},
                                                 data_0={'X_0': X_0, 'E_0': E_0, 'y_0': y_0},
                                                 gamma_0=gamma_0,
                                                 eps={'eps_X0': eps0.X, 'eps_E0': eps0.E, 'eps_y0': eps0.y},
                                                 pred_0=pred_0,
                                                 node_mask=node_mask,
                                                 test=test)

        # Combine terms
        nlls = - log_pN + kl_prior + loss_all_t + loss_term_0
        assert len(nlls.shape) == 1, f'{nlls.shape} has more than only batch dim.'

        # Update NLL metric object and return batch nll

        nll = self.test_nll(nlls) if test else self.val_nll(nlls)  # Average over the batch

        wandb.log({"kl prior": kl_prior.mean(),
                   "Estimator loss terms": loss_all_t.mean(),
                   "Loss term 0": loss_term_0,
                   "log_pn": log_pN.mean(),
                   'test_nll' if test else 'val_nll': nll},
                  commit=False)
        return nll

    def forward(self, noisy_data, extra_data, node_mask):
        """ Concatenates extra data to the noisy data, then calls the network. """
        X = torch.cat((noisy_data['X_t'], extra_data.X), dim=2)
        E = torch.cat((noisy_data['E_t'], extra_data.E), dim=3)
        y = torch.hstack((noisy_data['y_t'], extra_data.y))
        return self.model(X, E, y, node_mask)

    def log_info(self):
        """
        Some info logging of the model.
        """
        gamma_0 = self.gamma(torch.zeros(1, device=self.device))
        gamma_1 = self.gamma(torch.ones(1, device=self.device))

        log_SNR_max = -gamma_0
        log_SNR_min = -gamma_1

        info = {'log_SNR_max': log_SNR_max.item(), 'log_SNR_min': log_SNR_min.item()}
        print("", info, "\n")

        return info

    @torch.no_grad()
    def sample_batch(self, batch_id: int, batch_size: int, keep_chain: int, save_final: int, number_chain_steps: int,
                      num_nodes=None):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param number_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(batch_size, device=self.device, dtype=torch.int)
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_nodes_max = torch.max(n_nodes).item()

        # Build the masks
        arange = torch.arange(n_nodes_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        node_mask = arange < n_nodes.unsqueeze(1)
        node_mask = node_mask.float()

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        # TODO: how to move on the right device in the multi-gpu case?
        z_T = diffusion_utils.sample_feature_noise(X_size=(batch_size, n_nodes_max, self.Xdim_output),
                                                   E_size=(batch_size, n_nodes_max, n_nodes_max, self.Edim_output),
                                                   y_size=(batch_size, self.ydim_output),
                                                   node_mask=node_mask)
        X, E, y = z_T.X, z_T.E, z_T.y

        assert (E == torch.transpose(E, 1, 2)).all()
        assert number_chain_steps < self.T
        chain_X_size = torch.Size((number_chain_steps, keep_chain, X.size(1)))
        chain_E_size = torch.Size((number_chain_steps, keep_chain, E.size(1), E.size(2)))

        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)

        # Iteratively sample p(z_s | z_t) for t = 1, ..., T, with s = t - 1.
        average_X_coord = []
        average_E_coord = []
        for s_int in reversed(range(0, self.T)):
            s_array = s_int * torch.ones((batch_size, 1)).type_as(y)
            t_array = s_array + 1
            s_norm = s_array / self.T
            t_norm = t_array / self.T

            z_s = self.sample_p_zs_given_zt(s=s_norm, t=t_norm, X_t=X, E_t=E, y_t=y, node_mask=node_mask)
            X, E, y = z_s.X, z_s.E, z_s.y
            write_index = (s_int * number_chain_steps) // self.T
            unnormalized = utils.unnormalize(X=X[:keep_chain], E=E[:keep_chain], y=y[:keep_chain],
                                             norm_values=self.norm_values,
                                             norm_biases=self.norm_biases,
                                             node_mask=node_mask[:keep_chain],
                                             collapse=True)

            chain_X[write_index] = unnormalized.X
            chain_E[write_index] = unnormalized.E
            average_X_coord.append(X.abs().mean().item())
            average_E_coord.append(E.abs().mean().item())

        print(f"Average X coordinate at each step {[int(c) for i, c in enumerate(average_X_coord) if i % 10 == 0]}")
        print(f"Average E coordinate at each step {[int(c) for i, c in enumerate(average_E_coord) if i % 10 == 0]}")

        # Finally sample the discrete data given the last latent code z0
        final_graph = self.sample_discrete_graph_given_z0(X, E, y, node_mask)
        X, E, y = final_graph.X, final_graph.E, final_graph.y
        assert (E == torch.transpose(E, 1, 2)).all()

        print("Examples of generated graphs:")
        for i in range(min(5, X.shape[0])):
            print("E", E[i])
            print("X: ", X[i])

        # Prepare the chain for saving
        if keep_chain > 0:
            final_X_chain = X[:keep_chain]
            final_E_chain = E[:keep_chain]
            chain_X[0] = final_X_chain
            chain_E[0] = final_E_chain

            chain_X = diffusion_utils.reverse_tensor(chain_X)
            chain_E = diffusion_utils.reverse_tensor(chain_E)

            # Repeat last frame to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            assert chain_X.size(0) == (number_chain_steps + 10)

        # Split the generated molecules
        molecule_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])

        # Visualize chains
        if self.visualization_tools is not None:
            print('Visualizing chains...')
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)       # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(current_path, f'chains/{self.cfg.general.name}/'
                                                         f'epoch{self.current_epoch}/'
                                                         f'chains/molecule_{batch_id + i}')
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(result_path,
                                                                 chain_X[:, i, :].numpy(),
                                                                 chain_E[:, i, :].numpy())
                print('\r{}/{} complete'.format(i+1, num_molecules), end='', flush=True)

            # Visualize the final molecules
            print("Visualizing molecules...")
            current_path = os.getcwd()
            result_path = os.path.join(current_path,
                                       f'graphs/{self.name}/epoch{self.current_epoch}_b{batch_id}/')
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            print("Done.")

        return molecule_list

    def sample_discrete_graph_given_z0(self, X_0, E_0, y_0, node_mask):
        """ Samples X, E, y ~ p(X, E, y|z0): once the diffusion is done, we need to map the result
        to categorical values.
        """
        zeros = torch.zeros(size=(X_0.size(0), 1), device=X_0.device)
        gamma_0 = self.gamma(zeros)
        # Computes sqrt(sigma_0^2 / alpha_0^2)
        sigma = diffusion_utils.SNR(-0.5 * gamma_0).unsqueeze(1)
        noisy_data = {'X_t': X_0, 'E_t': E_0, 'y_t': y_0, 't': torch.zeros(y_0.shape[0], 1).type_as(y_0)}
        extra_data = self.compute_extra_data(noisy_data)
        eps0 = self.forward(noisy_data, extra_data, node_mask)

        # Compute mu for p(zs | zt).
        sigma_0 = diffusion_utils.sigma(gamma_0, target_shape=eps0.X.size())
        alpha_0 = diffusion_utils.alpha(gamma_0, target_shape=eps0.X.size())

        pred_X = 1. / alpha_0 * (X_0 - sigma_0 * eps0.X)
        pred_E = 1. / alpha_0.unsqueeze(1) * (E_0 - sigma_0.unsqueeze(1) * eps0.E)
        pred_y = 1. / alpha_0.squeeze(1) * (y_0 - sigma_0.squeeze(1) * eps0.y)
        assert (pred_E == torch.transpose(pred_E, 1, 2)).all()

        sampled = diffusion_utils.sample_normal(pred_X, pred_E, pred_y, sigma, node_mask).type_as(pred_X)
        assert (sampled.E == torch.transpose(sampled.E, 1, 2)).all()

        sampled = utils.unnormalize(sampled.X, sampled.E, sampled.y, self.norm_values,
                                    self.norm_biases, node_mask, collapse=True)
        return sampled

    def sample_p_zs_given_zt(self, s, t, X_t, E_t, y_t, node_mask):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        gamma_s = self.gamma(s)
        gamma_t = self.gamma(t)

        sigma2_t_given_s, sigma_t_given_s, alpha_t_given_s = diffusion_utils.sigma_and_alpha_t_given_s(gamma_t,
                                                                                                       gamma_s,
                                                                                                       X_t.size())
        sigma_s = diffusion_utils.sigma(gamma_s, target_shape=X_t.size())
        sigma_t = diffusion_utils.sigma(gamma_t, target_shape=X_t.size())

        E_t = (E_t + E_t.transpose(1, 2)) / 2
        noisy_data = {'X_t': X_t, 'E_t': E_t, 'y_t': y_t, 't': t}
        extra_data = self.compute_extra_data(noisy_data)
        eps = self.forward(noisy_data, extra_data, node_mask)

        # Compute mu for p(zs | zt).
        mu_X = X_t / alpha_t_given_s - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)) * eps.X
        mu_E = E_t / alpha_t_given_s.unsqueeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).unsqueeze(1) * eps.E
        mu_y = y_t / alpha_t_given_s.squeeze(1) - (sigma2_t_given_s / (alpha_t_given_s * sigma_t)).squeeze(1) * eps.y

        # Compute sigma for p(zs | zt).
        sigma = sigma_t_given_s * sigma_s / sigma_t

        # Sample zs given the parameters derived from zt.
        z_s = diffusion_utils.sample_normal(mu_X, mu_E,  mu_y, sigma, node_mask).type_as(mu_X)

        return z_s

    def compute_extra_data(self, noisy_data):
        """ At every training step (after adding noise) and step in sampling, compute extra information and append to
            the network input. """
        X = noisy_data['X_t']
        E = noisy_data['E_t']
        extra_x = torch.zeros((*X.shape[:-1], 0)).type_as(X)
        extra_edge_attr = torch.zeros((*E.shape[:-1], 0)).type_as(E)
        t = noisy_data['t']
        return utils.PlaceHolder(X=extra_x, E=extra_edge_attr, y=t)

#transformer model
import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from src import utils
from src.diffusion import diffusion_utils
from src.models.layers import Xtoy, Etoy


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations.
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx)
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx)
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx)
        self.e_out = Linear(dx, de)
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        attn = F.softmax(Y, dim=2)

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]

        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]

        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)
