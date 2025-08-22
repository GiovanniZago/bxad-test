import numpy as np
import uproot
import awkward as ak
import matplotlib.pyplot as plt
import pickle

import torch
from torch.utils.data import Dataset
import torch.nn as nn
import torch.functional as F

FEATURES_TO_DISCARD = [
    "nL1BMTFStub", 
    "orbitNumber"
]

CONTINUOUS_FEATURES = [
    "L1BMTFStub_Phi", 
    "L1BMTFStub_PhiB"
]

DISCRETE_FEATURES = [
    "L1BMTFStub_hwEtaD", # discretized variable
    "L1BMTFStub_hwQEtaD", # discretized variable 
    "L1BMTFStub_hwQual",
    "L1BMTFStub_wheel", 
    "L1BMTFStub_sector", 
    "L1BMTFStub_station",
    "L1BMTFStub_bxpos"
]

HWETA_KMEANS = "hweta_kmeans.pkl"
HWQETA_KMEANS = "hwqeta_kmeans.pkl"

def removeFeatures(features):
    global FEATURES_TO_DISCARD
    newfeatures = features.copy()
    
    for feat in FEATURES_TO_DISCARD:
        newfeatures.remove(feat)

    return newfeatures

class SequenceDataset(Dataset):
    def __init__(self, sequenceTree, max_stubs_per_bx=3):
        super().__init__()
        self.max_stubs_per_bx = max_stubs_per_bx

        meta = sequenceTree["L1BMTFStubSequencesMeta"].arrays()
        self.sequence_length = meta["sequenceLength"].to_numpy().item()
        
        # create sequence dataframe and prune useless columns
        df = ak.to_dataframe(sequenceTree["L1BMTFStubSequences"].arrays())
        df = df.drop(columns=FEATURES_TO_DISCARD)

        # discretize hwEta
        with open(HWETA_KMEANS, "rb") as f:
            hweta_kmeans = pickle.load(f)

        hweta = df["L1BMTFStub_hwEta"].values.reshape((-1, 1))
        hweta_dis = hweta_kmeans.predict(hweta)
        df["L1BMTFStub_hwEtaD"] = hweta_dis

        # discretize hwQEta
        with open(HWQETA_KMEANS, "rb") as f:
            hwqeta_kmeans = pickle.load(f)

        hwqeta = df["L1BMTFStub_hwQEta"].values.reshape((-1, 1))
        hwqeta_dis = hwqeta_kmeans.predict(hwqeta)
        df["L1BMTFStub_hwQEtaD"] = hwqeta_dis

        # define Phi
        df["L1BMTFStub_Phi"] = df["L1BMTFStub_hwPhi"].values * ((np.pi / 6) / 2048)

        # define PhiB
        df["L1BMTFStub_PhiB"] = df["L1BMTFStub_hwPhiB"].values * (np.pi / 512)

        # cast Station between 0 and 3 instead of 1 and 4
        df["L1BMTFStub_station"] = df["L1BMTFStub_station"] - 1

        # cast Wheel between 0 and 4
        df["L1BMTFStub_wheel"] = df["L1BMTFStub_wheel"] + 2
        
        # define sequences
        self.sequences = [g for _, g in df.groupby("sequenceIndex")]
        
    def __getitem__(self, index):
        sequence = self.sequences[index]
        
        continuous_features = torch.ones((len(CONTINUOUS_FEATURES), self.sequence_length, self.max_stubs_per_bx), dtype=torch.float32) * -1.0
        discrete_features = torch.ones((len(DISCRETE_FEATURES), self.sequence_length, self.max_stubs_per_bx), dtype=torch.int32) * -1

        for ii, (_, bxgroup) in enumerate(sequence.groupby("bunchCrossing")):
            bxgroup["L1BMTFStub_bxpos"] = ii

            for jj, feature in enumerate(CONTINUOUS_FEATURES):
                ncols = bxgroup[feature].values.shape[0] if bxgroup[feature].values.shape[0] <= self.max_stubs_per_bx else self.max_stubs_per_bx
                continuous_features[jj, ii, :ncols] = torch.from_numpy(bxgroup[feature].values[:ncols])

            for jj, feature in enumerate(DISCRETE_FEATURES):
                ncols = bxgroup[feature].values.shape[0] if bxgroup[feature].values.shape[0] <= self.max_stubs_per_bx else self.max_stubs_per_bx
                discrete_features[jj, ii, :ncols] = torch.from_numpy(bxgroup[feature].values[:ncols])

        return torch.unbind(continuous_features, dim=0) + torch.unbind(discrete_features, dim=0)

    def __len__(self):
        return len(self.sequences)
    

class FeatureEmbedding(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer_norm = nn.LayerNorm()

if __name__ == "__main__":
    filepath = "data/output_1000_seq3s.root"
    data_tree = uproot.open(filepath)
    dataset = SequenceDataset(data_tree)

    print(len(dataset))