# Utils
from collections import namedtuple
from itertools import tee
from os.path import dirname
import pandas as pd
import numpy as np
from typing import *
from scipy import sparse
# Rdkit
from rdkit import Chem
from rdkit import DataStructs
import rdkit.Chem.FilterCatalog as FilterCatalog
from rdkit.Chem import Draw

FilterMatch = namedtuple('FilterMatch', 'SubstructureMatches')


def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


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


def np_to_bv(array: np.array) -> DataStructs.ExplicitBitVect:
    """
    To calculate similarity by RDkit library, bitvector is converted to DataStructs.ExplicitBitVect object
    :param array: e.g. [0,1,0,0,1,0,0]
    :return: ExplicitBitVect
    """
    bv = DataStructs.ExplicitBitVect(len(array))
    bv.SetBitsFromList(np.where(array)[0].tolist())
    return bv


def csr_to_bv_list(csr_matrix: sparse.csr_matrix) -> List[DataStructs.cDataStructs.ExplicitBitVect]:
    return [np_to_bv(arr) for arr in csr_matrix.toarray()]


class AtomEnvironment(NamedTuple):
    """"A Class to store environment-information for morgan-fingerprint features"""
    central_atom: int  # Atom index of central atom
    radius: int  # bond-radius of environment
    environment_atoms: Set[int]  # set of all atoms within radius


class CoreSubstituentFingerprint:

    def __init__(self, file_path=f'{dirname(__file__)}/data/SR_FR_RFSFrc.csv'):
        self._filter = FilterCatalog.FilterCatalog()

        # Extracting pattern from Files,
        SR_FR_RFSFrc_df = pd.read_csv(file_path)
        SR_FR_RFSFrc_df["smarts_list"] = SR_FR_RFSFrc_df["sep_smarts"].map(lambda x: x.split("_"))
        smarts_list_arr: List[list] = SR_FR_RFSFrc_df["smarts_list"].to_list()
        self._n_bits = len(smarts_list_arr)

    @property
    def n_bits(self) -> int:
        return self._n_bits

    def fit(self, mol_obj_list: List[Chem.Mol]) -> None:
        pass

    def fit_transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        return self.transform(mol_obj_list)

    def _gen_features(self, mol_obj: Chem.Mol):
        return sorted(set([int(match.GetDescription()) for match in self._filter.GetMatches(mol_obj)]))

    def feature_to_bit(self, mol_obj: Chem.Mol):

        filter_matches = list(self._filter.GetMatches(mol_obj))
        bit_matches = [int(match.GetDescription()) for match in self._filter.GetMatches(mol_obj)]
        result_dict = {k: v for k, v in zip(bit_matches, filter_matches)}
        return result_dict

    def _transform(self, mol_fp_list: Union[Iterator[Dict[int, int]], List[Dict[int, int]]]) -> sparse.csr_matrix:
        data = []
        rows = []
        cols = []
        n_col = 0
        for i, mol_fp in enumerate(mol_fp_list):
            data.extend([1] * len(mol_fp))
            rows.extend(mol_fp)
            cols.extend([i] * len(mol_fp))
            n_col += 1
        return sparse.csr_matrix((data, (cols, rows)), shape=(n_col, self.n_bits))

    def transform(self, mol_obj_list: List[Chem.Mol]) -> sparse.csr_matrix:
        mol_iterator = (self._gen_features(mol_obj) for mol_obj in mol_obj_list)
        return self._transform(mol_iterator)

    def fit_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        self.fit(mol_obj_list)

    def fit_transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.fit_transform(mol_obj_list)

    def transform_smiles(self, smiles_list: List[str]):
        mol_obj_list = construct_check_mol_list(smiles_list)
        return self.transform(mol_obj_list)

    def feature_to_draw(self, mol_obj: Chem.Mol):

        matches = self._filter.GetFilterMatches(mol_obj)

        legends = []
        mols = []
        highlightAtomLists = []
        print(f'{len(matches)} matched')
        for idx in range(len(matches)):
            patterns_name = matches[idx].filterMatch.GetName()
            legends.append(f'{patterns_name}')
            patterns = matches[idx].filterMatch.GetPattern()
            matched_atoms = np.array(mol_obj.GetSubstructMatch(patterns)).flatten().tolist()
            highlightAtomLists.append(list(matched_atoms))
            mols.append(mol_obj)
        im = Draw.MolsToGridImage(mols, legends=legends, highlightAtomLists=highlightAtomLists, subImgSize=[400, 400])
        return im

    def feature_to_atoms(self, mol_obj: Chem.Mol, bits: List[int] = None):

        filter_matches = self._filter.GetFilterMatches(mol_obj)

        present_bits = self._gen_features(mol_obj)
        print(present_bits)
        result_dict = {k: v for k, v in zip(present_bits, filter_matches)}

        # select features to bits provided
        if bits:
            match_dict = {k: result_dict[k] for k in bits if k in result_dict}
            assert list(match_dict.keys()) == bits
        else:
            match_dict = result_dict

        match_atoms_list = []
        match_bonds_list = []

        for idx in match_dict.keys():
            pattern = match_dict[idx].filterMatch.GetPattern()
            match_atoms = np.array(mol_obj.GetSubstructMatch(pattern)).flatten().tolist()
            match_atoms_list.append(match_atoms)
            match_bonds = self.atom_bonds(mol_obj, match_atoms, pattern)
            match_bonds_list.append(match_bonds)

        if bits:
            highlight_atoms = match_atoms_list
            highlight_bonds = match_bonds_list
            return highlight_atoms, highlight_bonds
        else:
            highlight_atoms = [y for x in match_atoms_list for y in x]
            return list(set(highlight_atoms))

    def atom_bonds(self, mol_obj: Chem.Mol, hit_ats, pattern):
        bond_list = []
        for bond in pattern.GetBonds():
            a1 = hit_ats[bond.GetBeginAtomIdx()]
            a2 = hit_ats[bond.GetEndAtomIdx()]
            bond_list.append(mol_obj.GetBondBetweenAtoms(a1, a2).GetIdx())
        return bond_list


def CSFP(smiles_list: List[str], bit_vec=True):
    """
    Generates CSFP fingerprint
    :param smiles_list: compound smile list
    :param bit_vec: True if output is a RDKit object, False - CSFP as a sparse matrix
    :return: CSFP
    """
    CSFP = CoreSubstituentFingerprint()
    sparse_fp_matrix: sparse.csr_matrix = CSFP.transform_smiles(smiles_list)
    if bit_vec:
        # Dense matrix
        return csr_to_bv_list(sparse_fp_matrix)
    else:
        # Sparse matrix
        return sparse_fp_matrix


if __name__ == "__main__":

    test_smiles_list = ["c1ccccc1",
                        "C=C(C)C(CC=C(C)C)Cc1c(O)cc(O)c2c1OC(c1ccccc1O)CC2=O",
                        "C=Cc1cnc(C(=O)Nc2cccc(C3(C)CCSC(N)=N3)c2)cn1",
                        ]
    test_mol_obj_list = construct_check_mol_list(test_smiles_list)

    csfp = CSFP(test_smiles_list)
    print(csfp)

