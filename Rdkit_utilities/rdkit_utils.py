# Imports
import re
from typing import Iterable, List
import pandas as pd
from tqdm import tqdm
# RDKit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MolStandardize


def substituent_conversion(subst_strings: str) -> str:
    """
    Replaces the cutting sites with Hidrogen atoms, to be later removed
    :param subst_strings: smile str
    :return: str: modified smile string
    """
    subst = re.sub("\[\*:\d+\]", "[H]", subst_strings)
    return subst


def remove_H(mol_smarts: str) -> list[str]:
    """
    Removes the H atoms from smiles
    :param mol_smarts: smarts str
    :return: List of smarts without H
    """
    smart_no_H_list = []
    for smart in mol_smarts:
        reg = re.sub('[H]', '', smart)
        smart_no_H_list.append(reg)
    return smart_no_H_list


def remove_charge(core):
    return re.sub('[+]', '', core)


def mol_weight(smiles):
    molweight_list = []
    for smi in smiles:
        try:
            mw = Chem.Descriptors.MolWt(Chem.MolFromSmiles(smi))
            molweight_list.append(mw)
        except:
            print(smi)
    return molweight_list


def gen_canon_smiles(smarts):
    """
    Generates canonical smiles from smarts
    :param smarts: str
    :return: Canonical smiles
    """
    return Chem.MolToSmiles(Chem.MolFromSmarts(smarts))


def gen_canon_smiles_dict(smarts_list):
    """
    Generates a dictionary with the canonical smiles as values
    :param smarts_list: List of smarts
    :return: Dictionary with the canonical smiles
    """
    canon_smiles_list = [Chem.MolToSmiles(Chem.MolFromSmarts(smarts)) for smarts in smarts_list]
    smarts_dict = dict(zip(smarts_list, canon_smiles_list))
    return smarts_dict


def remove_dup_smarts(ringlist):
    # Get Cano smiles to remove dup smarts
    canon_smiles_dict = gen_canon_smiles_dict(ringlist)
    # Remove duplicate values in dictionary
    # Using dictionary comprehension
    temp_dict = {val: key for key, val in canon_smiles_dict.items()}
    smarts_dict = {val: key for key, val in temp_dict.items()}
    return [smarts for smarts in smarts_dict.keys()]


def GetNumRingsize(mol):
    """
    In order to calculate ring size easily, once converting into (CSK) cyclic skeletons.
    :param mol: mol object
    :return: int: max ring size
    """
    mol = Chem.Scaffolds.MurckoScaffold.MakeScaffoldGeneric(mol)
    Chem.FastFindRings(mol)
    ringinfo = mol.GetRingInfo()
    sizes = [len(tpl) for tpl in ringinfo.AtomRings()]
    return max(sizes)


def ring_count(sma: str) -> int:
    """
    Counts the number of rings in the molecule
    :param sma: smarts str
    :param smi: smile str
    :return: int: Number of rings
    """
    mol = Chem.MolFromSmarts(sma)
    Chem.FastFindRings(mol)
    ri = mol.GetRingInfo().NumRings()
    return ri


def fused_ring_count(smi: str) -> list:
    """
    Generates a list of atoms indices that are inside fused ring systems
    :param smi: smiles
    :return: list of atom indices
    """
    ringfusedatom = Chem.MolFromSmarts('[*R2]')
    matches = AllChem.MolFromSmiles(smi).GetSubstructMatches(ringfusedatom)
    return [x[0] for x in matches]


def ring_info(mol):
    """
    Get the ring information from the molecule
    :param mol: mol object
    :return: RingInfo
    """
    try:
        ring = mol.GetRingInfo()
        return ring
    except RuntimeError:
        Chem.GetSymmSSSR(mol)
        ring = mol.GetRingInfo()
        return ring


def ring_atoms(mol):
    """
    Get the ring atoms from the molecule
    :param mol: mol object
    :return: Ring atoms
    """
    rings = ring_info(mol).AtomRings()
    ring_list = [list(ri) for ri in rings]
    return ring_list


def GetRingSystems(mol, includeSpiro=True):
    """
    From https://www.rdkit.org/docs/Cookbook.html
    :param mol: mol boject
    :param includeSpiro: bool
    :return: List
    """
    ri = mol.GetRingInfo()
    systems = []
    for ring in ri.AtomRings():
        ringAts = set(ring)
        nSystems = []
        for system in systems:
            nInCommon = len(ringAts.intersection(system))
            if nInCommon and (includeSpiro or nInCommon > 1):
                ringAts = ringAts.union(system)
            else:
                nSystems.append(system)
        nSystems.append(ringAts)
        systems = nSystems

    return systems


def mol_with_atom_index(smi: str) -> Iterable[Chem.Mol]:
    """
    Display the index number in the mol 2D draw
    :param smi: mol smile
    :return: RO.Mol
    """
    mol = Chem.MolFromSmiles(smi)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return mol


def reorder_smart(original_smart) -> List[str]:
    """
    Reorganize SMART list
    :param original_smart: SMART list
    :return: str: Common SMART
    """
    sma_list = []
    sma_list_unique = list(dict.fromkeys(sma_list))
    for sma in original_smart:
        try:
            mol_obj = Chem.MolFromSmiles(Chem.CanonSmiles(sma))
            Chem.Kekulize(mol_obj)
            sma_list.append(Chem.MolToSmarts(mol_obj))
        except:
            sma_list.append(sma)
    return sma_list_unique


def canon_smarts(smarts: Iterable[str]) -> List:
    """
    Generates Unique Smarts
    :param smarts: SMARTS str
    :return: List of unique SMARTS
    """
    sma_list = []
    for sma in smarts:
        canon_sma = Chem.MolToSmarts(Chem.MolFromSmiles(MolStandardize.standardize_smiles(sma)))
        sma_list.append(canon_sma)
    return sma_list


# correct nH smiles MS
def my_mol_from_smiles(smiles):
    """
    Corrects the number of H atoms in the molecule
    :param smiles: smiles str
    :return: corrected smiles
    """
    m = Chem.MolFromSmiles(smiles)
    if m is not None:
        return Chem.MolToSmiles(m)
    m = Chem.MolFromSmiles(smiles, sanitize=False)
    for a in m.GetAtoms():
        if a.GetAtomicNum() == 7:
            v = sum([int(bond.GetBondTypeAsDouble()) for bond in a.GetBonds()])
            if v < 3:
                a.SetNumExplicitHs(3 - v)
            a.UpdatePropertyCache()
            break
    return Chem.MolToSmiles(m)


def my_mol_from_smiles_list(smi_list: Iterable[str]):
    """
    Corrects the number of H atoms in the molecule
    :param smi_list: smiles list
    :return: Corrected smiles list
    """
    smi_list_nH = []
    for smiles in smi_list:
        m = Chem.MolFromSmiles(smiles)
        if m is not None:
            smi_list_nH.append(Chem.MolToSmiles(m))
        m = Chem.MolFromSmiles(smiles, sanitize=False)
        for a in m.GetAtoms():
            if a.GetAtomicNum() == 7:
                v = sum([int(bond.GetBondTypeAsDouble()) for bond in a.GetBonds()])
                if v < 3:
                    a.SetNumExplicitHs(3 - v)
                a.UpdatePropertyCache()
                smi_list_nH.append(Chem.MolToSmiles(m))
                break
    return smi_list_nH


def pairwise_similarity(fps: list, idx) -> pd.DataFrame:
    """
    Generates the similarity Tc values for every cpds vs all other cpds
    :param fps: fingerprint vector
    :param idx: database chembl cid
    :return: Pandas dataframe with similarity
    """
    # the list for the dataframe
    qu_m, ta_m, sim = [], [], []
    # compare all fp pairwise without duplicates
    for n in tqdm(range(len(fps) - 1)):
        s = DataStructs.BulkTanimotoSimilarity(fps[n], fps[n + 1:])
        # collect the CIDs and values
        qu_m.extend([idx[n]] * len(s))
        sim.extend(s)
        ta_m.extend(idx[n + 1: n + 1 + len(s)])

    d = {'Similarity': sim}
    df_sim_matrix = pd.DataFrame(data=d)

    return df_sim_matrix
