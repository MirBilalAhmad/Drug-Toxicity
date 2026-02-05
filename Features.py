#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:00:56 2024

@author: bilal
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
# =======================
# LOAD DATA
# =======================
nr_ahr = pd.read_csv(Data_Path)

# =======================
# DESCRIPTOR FUNCTION
# =======================
def calculate_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [
        Descriptors.MolWt(mol),
        Descriptors.NumValenceElectrons(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol)
    ]

# =======================
# ECFP FINGERPRINT FUNCTION
# =======================



morgan_gen = GetMorganGenerator(radius=2, fpSize=1024)

def calculate_ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = morgan_gen.GetFingerprint(mol)
    return np.array(fp)


# def calculate_ecfp(smiles, radius=2, n_bits=1024):
#     mol = Chem.MolFromSmiles(smiles)
#     if mol is None:
#         return None
#     fp = AllChem.GetMorganFingerprintAsBitVect(
#         mol, radius=radius, nBits=n_bits
#     )
#     return np.array(fp)

# =======================
# APPLY DESCRIPTORS
# =======================
nr_ahr["Descriptors"] = nr_ahr["smiles"].apply(calculate_descriptors)
nr_ahr["ECFP"] = nr_ahr["smiles"].apply(calculate_ecfp)

# Remove failed molecules
nr_ahr = nr_ahr.dropna(subset=["Descriptors", "ECFP"])

# =======================
# EXPAND DESCRIPTORS
# =======================
descriptor_columns = [
    "MolecularWeight", "NumValenceElectrons", "MolLogP",
    "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "RingCount"
]

nr_ahr[descriptor_columns] = pd.DataFrame(
    nr_ahr["Descriptors"].tolist(), index=nr_ahr.index
)

# =======================
# EXPAND FINGERPRINTS
# =======================
fp_bits = 1024
fp_columns = [f"ECFP_{i}" for i in range(fp_bits)]

nr_ahr[fp_columns] = pd.DataFrame(
    nr_ahr["ECFP"].tolist(), index=nr_ahr.index
)

# =======================
# CLEAN DATAFRAME
# =======================
nr_ahr = nr_ahr.drop(columns=["smiles", "Descriptors", "ECFP"])

# =======================
# SAVE FINAL DATASET
# =======================
nr_ahr.to_csv("/",index=False)

print("✅ Descriptors + ECFP fingerprints generated and saved to '/....'")
