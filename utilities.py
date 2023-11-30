import os
import random
from itertools import tee
from typing import List

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from scipy import sparse
import tensorflow as tf


def scaffold_split(data, seed=42, test_size=0.2, n_splits=10, n_cpds_tolerance=5):
    from collections import defaultdict

    scaffolds = defaultdict(list)
    for idx, core in enumerate(data.core):
        scaffolds[core].append(idx)

    n_total_test = int(np.floor(test_size * len(data)))
    rng = np.random.RandomState(seed)
    for i in range(n_splits):

        scaffold_sets = rng.permutation(list(scaffolds.values()))
        scaffold_sets = np.array(scaffold_sets, dtype=object)

        train_index = []
        test_index = []

        for scaffold_set in scaffold_sets:
            if len(test_index) + len(scaffold_set) <= n_total_test:
                test_index.extend(scaffold_set)
            else:
                train_index.extend(scaffold_set)
        assert np.abs(len(test_index) - n_total_test) <= n_cpds_tolerance, (f'There are {len(test_index)} CPDs in the '
                                                                            f'test set, but {n_total_test} are '
                                                                            f'expected')
        yield train_index, test_index


def nn_removal(fp_matrix, cid_list, sim_cutoff=None):
    sim = []
    for fp in fp_matrix:
        sim.append(DataStructs.BulkTanimotoSimilarity(fp, fp_matrix))
    sim = np.array(sim)
    np.fill_diagonal(sim, 0)
    df_sim = pd.DataFrame(sim, columns=cid_list, index=cid_list)

    idx_nn = df_sim.idxmax().reset_index().apply(tuple, axis=1).to_list()

    df_sim = df_sim.stack()[idx_nn].sort_values(ascending=False).reset_index().rename(
        columns={'level_0': 'reference_cpd', 'level_1': 'nn_cpd', 0: 'similarity'})

    if sim_cutoff:
        df_sim = df_sim.loc[df_sim.similarity < sim_cutoff]

    return df_sim


class TanimotoKernel:
    def __init__(self, sparse_features=False):
        self.sparse_features = sparse_features

    @staticmethod
    def similarity_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
        intersection = matrix_a.dot(matrix_b.transpose()).toarray()
        norm_1 = np.array(matrix_a.multiply(matrix_a).sum(axis=1))
        norm_2 = np.array(matrix_b.multiply(matrix_b).sum(axis=1))
        union = norm_1 + norm_2.T - intersection
        return intersection / union

    @staticmethod
    def similarity_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
        intersection = matrix_a.dot(matrix_b.transpose())
        norm_1 = np.multiply(matrix_a, matrix_a).sum(axis=1)
        norm_2 = np.multiply(matrix_b, matrix_b).sum(axis=1)
        union = np.add.outer(norm_1, norm_2.T) - intersection

        return intersection / union

    def __call__(self, matrix_a, matrix_b):
        if self.sparse_features:
            return self.similarity_from_sparse(matrix_a, matrix_b)
        else:
            raise self.similarity_from_dense(matrix_a, matrix_b)


def tanimoto_from_sparse(matrix_a: sparse.csr_matrix, matrix_b: sparse.csr_matrix):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_sparse(matrix_a, matrix_b)


def tanimoto_from_dense(matrix_a: np.ndarray, matrix_b: np.ndarray):
    DeprecationWarning("Please use TanimotoKernel.sparse_similarity")
    return TanimotoKernel.similarity_from_dense(matrix_a, matrix_b)


def create_directory(path: str, verbose: bool = True):
    if not os.path.exists(path):

        if len(path.split("/")) <= 2:
            os.mkdir(path)
        else:
            os.makedirs(path)
        if verbose:
            print(f"Created new directory '{path}'")
    return path


def flatten(t):
    return [item for sublist in t for item in sublist]


def construct_check_mol_list(smiles_list: List[str]) -> List[Chem.Mol]:
    mol_obj_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    if None in mol_obj_list:
        invalid_smiles = []
        for smiles, mol_obj in zip(smiles_list, mol_obj_list):
            if not mol_obj:
                invalid_smiles.append(smiles)
        invalid_smiles = "\n".join(invalid_smiles)
        raise ValueError(f"Following smiles are not valid:\n {invalid_smiles}")
    return mol_obj_list


def construct_check_mol(smiles: str) -> Chem.Mol:
    mol_obj = Chem.MolFromSmiles(smiles)
    if not mol_obj:
        raise ValueError(f"Following smiles are not valid: {smiles}")
    return mol_obj


def ECFP4(smiles_list: List[str], n_bits=2048, radius=2) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    """
    Converts array of SMILES to ECFP bitvectors.
        AllChem.GetMorganFingerprintAsBitVect(mol, radius, length)
        n_bits: number of bits
        radius: ECFP fingerprint radius

    Returns: RDKit mol objects [List]
    """
    mols = construct_check_mol_list(smiles_list)
    return [AllChem.GetMorganFingerprintAsBitVect(m, radius, n_bits) for m in mols]


def set_seeds(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)


def set_global_determinism(seed):
    set_seeds(seed=seed)

    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def potency_classes(potency_values: list, potency_bins=None):
    if potency_bins is None:
        potency_bins = [5, 6, 7, 8, 9, 10, 11]

    pot_bin = []
    for pot in potency_values:
        pot_idx = pairwise(potency_bins)
        for idx in list(pot_idx):
            if idx[1] == max(potency_bins):
                if idx[0] <= pot <= idx[1]:
                    pot_bin.append(f'{idx[0]} - {idx[1]}')

            elif idx[0] <= pot < idx[1]:
                pot_bin.append(f'{idx[0]} - {idx[1]}')

    return pot_bin


def get_target_id(target_df: pd.DataFrame, extract_mode, n_cpds=None, t_id_list=None, random=False,
                      n_tids=None) -> pd.DataFrame:
    """

    :param target_df: compound db
    :param t_id_list: target to extract
    :param extract_mode: SS: similarity searching / ML: machine learning
    :return: Final Dataframe
    """

    if random:
        t_id_list = list(target_df['chembl_tid'].sample(n=n_tids, random_state=42))
    else:
        t_id_list = t_id_list

    if extract_mode == 'SS':
        df_join = pd.DataFrame()
        for t_id in t_id_list:
            result_df = target_df.query('chembl_tid in @t_id')
            result_df_sample = result_df.sample(n=n_cpds, random_state=1)
            df_all = pd.concat([df_join, result_df_sample], axis=0)
            df_join = df_all

        df_join.reset_index(drop=True, inplace=True)
        return df_join

    elif extract_mode == 'ML':

        result_df = target_df.query('chembl_tid in @t_id_list')

        return result_df


def stratify_dataset_tr_te(data, data_size=0.1, potency_bins=None, ):
    if potency_bins is None:
        potency_bins = [5, 6, 7, 8, 9, 10, 11]

    training_idx = []

    pot_idx = pairwise(potency_bins)

    for idx in list(pot_idx):
        if idx[1] == 11:
            pot_idx = [i for i, v in enumerate(data.labels) if idx[0] <= v <= idx[1]]
        else:
            pot_idx = [i for i, v in enumerate(data.labels) if idx[0] <= v < idx[1]]

        train_bin_size = int(round(data_size * len(data), 0))

        if len(pot_idx) >= train_bin_size:
            strat_idx = random.sample(pot_idx, train_bin_size)
            training_idx.append(strat_idx)
        else:
            strat_idx = random.sample(pot_idx, len(pot_idx))
            training_idx.append(strat_idx)

    training_set_idx = sorted([item for sublist in training_idx for item in sublist])
    test_set_idx = [x for x in range(len(data)) if x not in training_set_idx]

    assert training_set_idx != test_set_idx

    return training_set_idx, test_set_idx


def potency_class_size(data, potency_bins=None, size_factor=1.25, statistic='median'):
    if potency_bins is None:
        potency_bins = [5, 6, 7, 8, 9, 10, 11]

    pot_bins_pair = pairwise(potency_bins)

    dataset_bin_size = []
    for idx in list(pot_bins_pair):
        if idx[1] == 11:
            pot_idx = [i for i, v in enumerate(data.labels) if idx[0] <= v <= idx[1]]
            dataset_bin_size.append(len(pot_idx))
        else:
            pot_idx = [i for i, v in enumerate(data.labels) if idx[0] <= v < idx[1]]
            dataset_bin_size.append(len(pot_idx))

    from statistics import median, mean, mode
    if statistic == 'median':
        data_bin_size = int(round(median(dataset_bin_size) / size_factor))
    elif statistic == 'mean':
        data_bin_size = int(round(mean(dataset_bin_size) / size_factor))
    elif statistic == 'mode':
        data_bin_size = int(round(mode(dataset_bin_size) / size_factor))
    elif statistic == 'max':
        data_bin_size = int(round(max(dataset_bin_size) / size_factor))

    return data_bin_size


def balance_dataset(data, data_bin_size=None, potency_bins=None):
    if potency_bins is None:
        potency_bins = [5, 6, 7, 8, 9, 10, 11]

    dataset_idx = []
    for idx in list(pairwise(potency_bins)):

        if idx[1] == 11:
            pot_idx = [i for i, v in enumerate(data.labels) if idx[0] <= v <= idx[1]]
        else:
            pot_idx = [i for i, v in enumerate(data.labels) if idx[0] <= v < idx[1]]

        if len(pot_idx) >= data_bin_size:
            random.seed(1)
            strat_idx = random.sample(pot_idx, data_bin_size)
            dataset_idx.append(strat_idx)
        else:
            random.seed(1)
            strat_idx = random.sample(pot_idx, len(pot_idx))
            dataset_idx.append(strat_idx)

    dataset_set_idx = sorted([item for sublist in dataset_idx for item in sublist])

    return dataset_set_idx


def get_uniformly_distributed_sample(data, n_sample, bins, seed, verbose: bool = False):
    df = pd.DataFrame({'value': data}).reset_index().rename({'index': 'ID'}, axis=1)
    assert df.shape[0] >= n_sample
    df['bin'] = df['value'].apply(lambda x: np.digitize(x, bins))

    n_bin_sample = int(np.floor(n_sample / df['bin'].nunique()))

    n_missing = n_sample
    df_sample = pd.DataFrame()
    df_remaining = df
    n_available_bins = df['bin'].nunique()
    while n_missing >= n_available_bins:
        if verbose:
            print(f"Available pool size: {df_remaining.shape[0]}")
            print(f"Number of datapoints to select per bin: {n_bin_sample}")

        sizes_bins = df_remaining.groupby('bin').size()

        if verbose:
            print("Bin sizes:", sizes_bins.to_dict())

        bins_enough_data = sizes_bins[sizes_bins >= n_bin_sample].index.to_list()

        if verbose:
            print("Bins with enough data", bins_enough_data)

        df_sample_enough = pd.DataFrame()
        if len(bins_enough_data) != 0:
            df_sample_enough = df_remaining[df_remaining['bin'].isin(bins_enough_data)].groupby(
                'bin').sample(n_bin_sample, random_state=seed)
        df_sample_not_enough = df_remaining[~df_remaining['bin'].isin(bins_enough_data)]
        df_sample = pd.concat([df_sample, df_sample_enough, df_sample_not_enough])
        del df_sample_enough

        if verbose:
            print(f"Sample size: {df_sample.shape[0]}")

        n_missing = n_sample - df_sample.shape[0]

        df_remaining = df_remaining[~df_remaining['ID'].isin(df_sample['ID'].to_numpy())]
        n_bin_sample = int(np.floor(n_missing / df_remaining['bin'].nunique()))

        if verbose:
            print(f"Missing samples: {n_missing}\n")

    if n_missing > 0:
        if verbose:
            print(f"Select {n_missing} samples randomly")

        df_sample = pd.concat([df_sample, df_remaining.sample(n_missing, random_state=seed)])

    if verbose:
        print(f"Selected {df_sample.shape[0]} samples")
    assert df_sample.shape[0] == n_sample, f"Something went wrong."
    return sorted(df_sample['ID'].to_list())