import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from tqdm.notebook import tqdm
import seaborn as sns
from collections import Counter


def getSameBxTargetCollections(bxs):
    target_pairs = [
        [(bx, station) for station in range(1, 4)] for bx in bxs
    ]

    target_pairs += [
        [(bx, station) for station in range(2, 5)] for bx in bxs
    ]

    return target_pairs

def getMultiBxTargetCollections(bxs):
    target_pairs = [
        [(bx, station) for bx, station in zip(bxs, range(1, 4))]
    ]

    target_pairs += [
        [(bx, station) for bx, station in zip(bxs, range(2, 5))]
    ]

    return target_pairs

if __name__ == "__main__":
    filepath = "data/output_1000_seq3.root"
    data = uproot.open(filepath + ":L1BMTFStubSequences").arrays()
    df = ak.to_dataframe(data)


    same_bx_selection = {}
    multi_bx_selection = {}

    for idx_seq, df_seq in tqdm(df.groupby("sequenceIndex")):
        # df_seq_grouped = df_seq.groupby(["bunchCrossing", "L1BMTFStub_station"])
        pairs = [
            tuple(el.item() for el in key) 
            for key in df_seq.groupby(["bunchCrossing", "L1BMTFStub_station"]).indices
        ]
        
        same_bx_collections = getSameBxTargetCollections(df_seq["bunchCrossing"].unique().tolist())

        for tps in same_bx_collections:
            if np.all([target_pair in pairs for target_pair in tps]):
                """
                Here we keep only the entries of the sequence sub-dataframe that 
                correspond to the identified (bx, station) pairs.
                This allows us then to groupby wheel and keep only the wheels that are
                at most distant 1 to one another.
                """
                mi = pd.MultiIndex.from_tuples(tps, names=["bunchCrossing", "L1BMTFStub_station"])
                df_temp = df_seq[
                    df_seq.set_index(["bunchCrossing", "L1BMTFStub_station"]).index.isin(mi)
                ]

                print(df_temp[["bunchCrossing", "L1BMTFStub_station", "L1BMTFStub_wheel"]])
                break