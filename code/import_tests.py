from sklearn.utils import shuffle
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import gc

# Global variable for dataset label (can be updated as needed)
datalabel = "B_CELL"

def data_label():
    return datalabel

# [MOD] Feature files per feature type (train = B-*, independent test = Ind-*)
DATASET_DIR = "/mnt/D/jupyter/globe/B_CELL/dataset"
FEATURE_PATHS = {
    "ESM2": {
        "neg": {"train": f"{DATASET_DIR}/ESM2/B-negative.npy", "test": f"{DATASET_DIR}/ESM2/Ind-negative.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESM2/B-positive.npy", "test": f"{DATASET_DIR}/ESM2/Ind-positive.npy"},
    },
    # [MOD] ESM C 600M embeddings from code/get_ESMCEmbedding.py
    "ESMC": {
        "neg": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_neg_fused_1_1.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_neg_concat_1_1.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_pos_fused_1_1.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_pos_concat_1_1.npy"},
    },
    # "ESMC": {
    #     "neg": {"train": f"{DATASET_DIR}/ESMC/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC/Ind_neg.npy"},
    #     "pos": {"train": f"{DATASET_DIR}/ESMC/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC/Ind_pos.npy"},
    # },
    # [MOD] RAG fused features at three query ratios r, built by
    # analysis/make_ratio.py from the stored neighbour indices:
    #     fused = r * query + (1 - r) * mean(top-5 neighbours)
    # R05 reuses the existing *_fused_1_1.npy files (r = 0.5). Note that
    # *_fused_9_1.npy / *_fused_3_7.npy are NOT other ratios - they are
    # byte-identical copies of the r = 0.5 data and are not referenced here.
    "ESMC_R09": {
        "neg": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_neg_fused_r09.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_neg_fused_r09.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_pos_fused_r09.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_pos_fused_r09.npy"},
    },
    "ESMC_R05": {
        "neg": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_neg_fused_1_1.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_neg_fused_1_1.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_pos_fused_1_1.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_pos_fused_1_1.npy"},
    },
    "ESMC_R03": {
        "neg": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_neg_fused_r03.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_neg_fused_r03.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_pos_fused_r03.npy", "test": f"{DATASET_DIR}/ESMC/RAG/Ind_pos_fused_r03.npy"},
    },
    # [MOD] Raw ESM C embeddings re-split by code/make_protein_disjoint_split.py.
    # Same 12,443 peptides and the same embeddings as ESMC/{Train,Ind}_*.npy -
    # only the train/test cut differs, so these two isolate the provenance
    # shortcut from the effect of re-splitting at all:
    #   ESMC_PDIS    no source protein shared between train and test;
    #                per-protein positive rate scores AUC 0.500 on its test set
    #   ESMC_RSPLIT  random split of the same pool, matched sizes;
    #                the same control scores AUC 0.876
    "ESMC_PDIS": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_PDIS/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC_PDIS/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_PDIS/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC_PDIS/Ind_pos.npy"},
    },
    "ESMC_RSPLIT": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_RSPLIT/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC_RSPLIT/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_RSPLIT/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC_RSPLIT/Ind_pos.npy"},
    },
    # [MOD] RAG + KG context at r = 0.3, built by code/make_ragkg.py: every
    # retrieved database item carries its IEDB provenance (source protein,
    # antigen, source organism) projected to 1152 dims and added to its ESM C
    # embedding before the top-5 are averaged into the query. Retrieval is
    # label-partitioned (positive queries -> kg_pos_clean, negative ->
    # kg_neg_clean). Two directions, same training data:
    #   ESMC_RAGKG_R03          independent test fused the same way as training
    #   ESMC_RAGKG_R03_RAWIND   independent test left as raw ESM C
    # and the no-context control for the second direction:
    #   ESMC_R03_RAWIND         train on RAG r = 0.3 without context, test raw
    "ESMC_RAGKG_R03": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC_RAGKG/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC_RAGKG/Ind_pos.npy"},
    },
    "ESMC_RAGKG_R03_RAWIND": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC/Ind_pos.npy"},
    },
    "ESMC_R03_RAWIND": {
        "neg": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_neg_fused_r03.npy", "test": f"{DATASET_DIR}/ESMC/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC/RAG/Train_pos_fused_r03.npy", "test": f"{DATASET_DIR}/ESMC/Ind_pos.npy"},
    },
    # [MOD] Held-out breast-cancer antigen validation set (code/make_breast_cancer_set.py,
    # code/retrieve_bc.py). Training is the benchmark training split; only the test side
    # changes, so these three entries evaluate the same models on 107 + 162 peptides from
    # CEA, MUC1, BRCA1/2, p53, ER/PR, mammaglobin and Ki-67 that appear nowhere in
    # Train_*/Ind_*.
    #   BC_ESMC_RAW        raw ESM C in, raw ESM C out - needs no label at test time
    #   BC_RAGKG_RAWIND    trained on RAG+KG r=0.3, tested on raw ESM C - also label-free
    #   BC_RAGKG_FUSED     tested on label-partitioned fusion, so the test label is used
    #                      to choose the partition; not a deployable configuration
    "BC_ESMC_RAW": {
        "neg": {"train": f"{DATASET_DIR}/ESMC/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC/BC_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC/BC_pos.npy"},
    },
    "BC_RAGKG_RAWIND": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC/BC_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC/BC_pos.npy"},
    },
    "BC_RAGKG_FUSED": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_neg.npy", "test": f"{DATASET_DIR}/BC/BC_neg_ragkg_r03.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_RAGKG/Train_pos.npy", "test": f"{DATASET_DIR}/BC/BC_pos_ragkg_r03.npy"},
    },
    # [MOD] CUNG embedding ESM C nhu "ESMC", nhung stack lai o MAXSEQ = 49 (chuoi dai
    # nhat cua toan bo dataset) thay vi 25, sinh boi esmc49/build_esmc49.py.
    # Dung voi -maxseq 49. Luu y: chi 34/12443 chuoi dai hon 25 residue va TAT CA
    # deu nam trong Ind-negative, nen vi tri 26-49 cua tap train toan bo la zero.
    "ESMC49": {
        "neg": {"train": f"{DATASET_DIR}/ESMC49/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC49/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC49/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC49/Ind_pos.npy"},
    },
    # [MOD] AMPLIFY 350M embeddings from code/get_AMPLIFYEmbedding.py
    "AMPLIFY": {
        "neg": {"train": f"{DATASET_DIR}/AMPLIFY/B-negative.npy", "test": f"{DATASET_DIR}/AMPLIFY/Ind-negative.npy"},
        "pos": {"train": f"{DATASET_DIR}/AMPLIFY/B-positive.npy", "test": f"{DATASET_DIR}/AMPLIFY/Ind-positive.npy"},
    },
    # [MOD] ProtT5-XL (1024-d) and Ankh3-large (1536-d) embeddings from code/get_T5Embedding.py
    "ProtT5": {
        "neg": {"train": f"{DATASET_DIR}/ProtT5/B-negative.npy", "test": f"{DATASET_DIR}/ProtT5/Ind-negative.npy"},
        "pos": {"train": f"{DATASET_DIR}/ProtT5/B-positive.npy", "test": f"{DATASET_DIR}/ProtT5/Ind-positive.npy"},
    },
    "Ankh3": {
        "neg": {"train": f"{DATASET_DIR}/Ankh3/B-negative.npy", "test": f"{DATASET_DIR}/Ankh3/Ind-negative.npy"},
        "pos": {"train": f"{DATASET_DIR}/Ankh3/B-positive.npy", "test": f"{DATASET_DIR}/Ankh3/Ind-positive.npy"},
    },
    # [MOD] ESM C đã fuse với KG context theo luật IFNg_DeepKG (code/kg_ifng_fusion.py).
    # Nội dung thay đổi theo -ratio của lần sinh gần nhất, xem dataset/ESMC_IFNG/ratio.json
    "ESMC_IFNG": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_IFNG/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC_IFNG/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_IFNG/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC_IFNG/Ind_pos.npy"},
    },
    # [MOD] ESM C fuse với top-5 RAG theo RAG_MCNNIL6 (code/rag_mcnnil6_fusion.py).
    # Nội dung thay đổi theo --ratio/--store của lần sinh gần nhất, xem dataset/ESMC_RAG/ratio.json
    "ESMC_RAG": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_RAG/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC_RAG/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_RAG/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC_RAG/Ind_pos.npy"},
    },
    # [MOD] ESM C fuse với top-5 RAG trên store đã lọc CD-HIT-2d 40%
    # (code/rag_clean40_fusion.py). Store này không chứa query peptide nào.
    # Nội dung thay đổi theo --ratio của lần sinh gần nhất, xem dataset/ESMC_RAG40/ratio.json
    "ESMC_RAG40": {
        "neg": {"train": f"{DATASET_DIR}/ESMC_RAG40/Train_neg.npy", "test": f"{DATASET_DIR}/ESMC_RAG40/Ind_neg.npy"},
        "pos": {"train": f"{DATASET_DIR}/ESMC_RAG40/Train_pos.npy", "test": f"{DATASET_DIR}/ESMC_RAG40/Ind_pos.npy"},
    },
    "TrimoPGLM": {
        "neg": {"train": f"{DATASET_DIR}/concatenate/all_b_neg_embeds.npy", "test": f"{DATASET_DIR}/concatenate/all_ind_neg_embeds.npy"},
        "pos": {"train": f"{DATASET_DIR}/concatenate/all_b_pos_embeds.npy", "test": f"{DATASET_DIR}/concatenate/all_ind_pos_embeds.npy"},
    },
}

def MCNN_data_load(feature_type, class_0_types=["neg"], class_1_types=["pos"]):
    """
    Load data for MCNN model, assigning class 0 and class 1 to the specified protein types.

    Parameters:
    - feature_type (str): Type of feature (e.g., ProtTrans embeddings).
    - class_0_types (list of str): Protein types to assign as class 0. 
    - class_1_types (list of str): Protein types to assign as class 1.

    Returns:
    - x_train, y_train: Training data and labels.
    - x_test, y_test: Test data and labels.
    """
    # Validate class_0_types and class_1_types
    all_types = ["pos", "neg"]
    if not isinstance(class_0_types, list) or not isinstance(class_1_types, list):
        raise ValueError("class_0_types and class_1_types must be lists")
    if not class_0_types or not class_1_types:
        raise ValueError("class_0_types and class_1_types cannot be empty")
    if not all(t in all_types for t in class_0_types + class_1_types):
        raise ValueError(f"Types must be from {all_types}, got {class_0_types + class_1_types}")
    if set(class_0_types) & set(class_1_types):
        raise ValueError("class_0_types and class_1_types cannot overlap")
    # if set(all_types) != set(class_0_types + class_1_types):
    #     raise ValueError("class_0_types and class_1_types must cover all types: membrane, primary, secondary")

    # [MOD] Pick the files from feature_type. Before, the path was hard-coded to ESM2,
    # so every feature_type (including "TrimoPGLM") silently loaded the ESM2 features.
    if feature_type not in FEATURE_PATHS:
        raise ValueError(f"Unknown feature_type '{feature_type}', expected one of {list(FEATURE_PATHS)}")
    paths = FEATURE_PATHS[feature_type]

    # Check if all files exist
    for protein_type, splits in paths.items():
        for split, path in splits.items():
            if not os.path.exists(path):
                raise FileNotFoundError(f"File not found: {path} for {protein_type} {split}")

    # Load data for all protein types
    data = {
        "neg": {
            "train": np.load(paths["neg"]["train"]),
            "test": np.load(paths["neg"]["test"])
        },
        "pos": {
            "train": np.load(paths["pos"]["train"]),
            "test": np.load(paths["pos"]["test"])
        },
    }

    # Print shapes for debugging
    for protein_type, splits in data.items():
        for split, array in splits.items():
            print(f"Shape of {protein_type} {split}: {array.shape}")

    # Prepare training and test data by combining and labeling
    x_train, y_train = combine_and_label_data(
        class_0_types=class_0_types,
        class_1_types=class_1_types,
        pos_data=data["pos"]["train"],
        neg_data=data["neg"]["train"]
    )
    x_test, y_test = combine_and_label_data(
        class_0_types=class_0_types,
        class_1_types=class_1_types,
        pos_data=data["pos"]["test"],
        neg_data=data["neg"]["test"]
    )

    # Shuffle the data
    x_train, y_train = shuffle(x_train, y_train, random_state=42)
    x_test, y_test = shuffle(x_test, y_test, random_state=42)

    print(f"Training data shape: {x_train.shape}, Labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}, Labels shape: {y_test.shape}")

    return x_train, y_train, x_test, y_test

def combine_and_label_data(class_0_types, class_1_types, pos_data, neg_data):
    """
    Combine data from different protein types and assign labels based on class_0_types and class_1_types.

    Parameters:
    - class_0_types (list of str): Protein types to assign as class 0.
    - class_1_types (list of str): Protein types to assign as class 1.
    - pos_data (np.ndarray): Data for b cell epitopes.
    - neg_data (np.ndarray): Data for non b cell epitopes.

    Returns:
    - x (np.ndarray): Combined feature data.
    - y (np.ndarray): One-hot encoded labels.
    """
    # Map protein types to their data
    data_map = {
        "pos": pos_data,
        # "secondary": secondary_train,
        "neg": neg_data
    }

    # Combine data for class 0
    class_0_data_list = [data_map[t] for t in class_0_types if data_map[t].size > 0]
    if not class_0_data_list:
        raise ValueError("No data available for class 0 types")
    class_0_data = np.concatenate(class_0_data_list, axis=0)

    # Combine data for class 1
    class_1_data_list = [data_map[t] for t in class_1_types if data_map[t].size > 0]
    if not class_1_data_list:
        raise ValueError("No data available for class 1 types")
    class_1_data = np.concatenate(class_1_data_list, axis=0)

    # Create labels
    label_1 = np.ones(class_1_data.shape[0])  # Class 1 for specified types
    label_0 = np.zeros(class_0_data.shape[0])  # Class 0 for specified types

    print(f"Class 1 ({class_1_types}) shape: {class_1_data.shape}")
    print(f"Class 0 ({class_0_types}) shape: {class_0_data.shape}")
    print(f"Class 1 labels shape: {label_1.shape}")
    print(f"Class 0 labels shape: {label_0.shape}")

    # Combine features and labels
    x = np.concatenate([class_1_data, class_0_data], axis=0)
    y = np.concatenate([label_1, label_0], axis=0)

    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, 2)

    # Clean up memory
    gc.collect()
    return x, y

# Example usage
if __name__ == "__main__":

    # Example 3: Membrane as class 0, primary and secondary as class 1
    print("\nLoading data with membrane as class 0 and primary as class 1...")
    x_train_m_ps, y_train_m_ps, x_test_m_ps, y_test_m_ps = MCNN_data_load(
        feature_type="ESM2", class_0_types=["neg"], class_1_types=["pos"]
    )