"""
PyTorch DataLoaders for neural / symbolic baselines on risky-choice tabular data.

Moved from ``legacy/data_loader.py`` (previously imported by ``exp1_neural_training``).
Paths resolve to the repository root (parent of ``utils/``).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import LeaveOneOut, train_test_split
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
import torch
import torch.nn.functional as F
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class BehavioralDataset(Dataset):
    def __init__(self, df, transform=None, dataset="small"):
        self.df = df
        self.transform = transform
        # Assuming 'inputs' and 'targets' are columns in your DataFrame
        self.inputs = df[["p1", "v1", "p2", "v2"]]  # Example input features
        self.sub_id = df["sub_id_encoded"]
        self.targets = df["choice"]  # Example target variable
        self.num_of_subs = df["sub_id_encoded"].nunique()
        if dataset == "small":
            self.embeddings = None
        elif dataset == "large":
            embeddings_file = (
                PROJECT_ROOT
                / "results"
                / "model_result"
                / "large_dataset"
                / "LLaMA3_70B_embedding.npy"
            )
            if embeddings_file.exists():
                self.embeddings = np.memmap(
                    str(embeddings_file), dtype="float32", mode="r", shape=(44308, 8192)
                )
            else:
                self.embeddings = None
            # normalize the embeddings
            # self.embeddings = (self.embeddings - self.embeddings.min()) / (self.embeddings.max() - self.embeddings.min())
            # self.embeddings = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Assuming p1, v1, p2, v2 are each lists of numbers:
        p1, v1, p2, v2 = self.inputs.iloc[idx]  # This might need adjustment based on actual data structure

        # Convert each to a tensor; example assumes they are list-like and can be directly converted
        p1_tensor = torch.tensor(p1, dtype=torch.float)
        v1_tensor = torch.tensor(v1, dtype=torch.float)
        p2_tensor = torch.tensor(p2, dtype=torch.float)
        v2_tensor = torch.tensor(v2, dtype=torch.float)

        # rescaling them before enetering the model
        p1_tensor = p1_tensor / 100
        v1_tensor = v1_tensor / 1000
        p2_tensor = p2_tensor / 100
        v2_tensor = v2_tensor / 1000

        # Bundle them into a single tuple or dictionary as the 'inputs'
        inputs = (p1_tensor, v1_tensor, p2_tensor, v2_tensor)

        target = torch.tensor(self.targets.iloc[idx], dtype=torch.float)  # Adjust the dtype as needed

        individual_id = torch.tensor(self.sub_id.iloc[idx], dtype=torch.int)

        num_individual_id = torch.tensor(self.num_of_subs, dtype=torch.int)

        embeddings_tensor = (
            torch.tensor(self.embeddings[idx, :], dtype=torch.float)
            if self.embeddings is not None
            else None
        )

        return inputs, target, individual_id, num_individual_id, embeddings_tensor


def collate_fn(batch):
    # Custom collate_fn to pad sequences to a fixed length along their feature dimension.
    # Extract inputs, targets, and individual IDs from the batch
    inputs = [
        (item[0][0], item[0][1], item[0][2], item[0][3]) for item in batch
    ]  # Each item[0] is a tuple of tensors (p1, v1, p2, v2)
    targets = torch.tensor([item[1] for item in batch], dtype=torch.float)
    # Assuming individual IDs are also part of the batch data, similar to inputs and targets
    individual_ids = torch.tensor([item[2] for item in batch], dtype=torch.long)
    num_of_sub = torch.tensor([item[3] for item in batch], dtype=torch.long)

    # Apply padding to each sequence within the input tuples and stack them
    p1_padded = torch.stack([pad_tensor(inp[0]) for inp in inputs])
    v1_padded = torch.stack([pad_tensor(inp[1]) for inp in inputs])
    p2_padded = torch.stack([pad_tensor(inp[2]) for inp in inputs])
    v2_padded = torch.stack([pad_tensor(inp[3]) for inp in inputs])

    embeddings_padded = torch.stack(
        [torch.zeros(1,) if item[4] is None else item[4] for item in batch]
    )
    # For symbolic models, individual IDs may not be needed, so they can be ignored.
    # For NN models, individual IDs will be used alongside other inputs.
    return (
        p1_padded,
        v1_padded,
        p2_padded,
        v2_padded,
        individual_ids,
        num_of_sub,
        embeddings_padded,
    ), targets


def pad_tensor(tensor, max_length=9):
    """
    Pad the tensor to have a sequence length of `max_length` along the last dimension.
    """
    current_length = tensor.size(-1)
    pad_size = max_length - current_length
    if pad_size > 0:
        # Pad at the end of the last dimension
        padded_tensor = F.pad(tensor, (0, pad_size), "constant", 0)
        return padded_tensor
    return tensor


def prepare_dataloaders(
    dataset="small",
    test_size=0.1,
    batch_size=64,
    transform=None,
    random_state=2024,
    model_usage="NN",
    by_progress=False,
):
    # Choose the dataset file based on the 'dataset' argument
    if dataset == "small":
        csv_file = PROJECT_ROOT / "data" / "behavioral_text_data.csv"
    elif dataset == "large":
        csv_file = PROJECT_ROOT / "data" / "behavioral_text_data_expanded.csv"

    df = pd.read_csv(csv_file)
    df.drop(columns=["0"], errors="ignore", inplace=True)
    df.columns = [
        "sub_id",
        "choice",
        "p1",
        "v1",
        "p2",
        "v2",
        "problem_id",
        "rt",
        "think_aloud",
        "word_count",
    ]

    # Process lists in the DataFrame
    for column in ["p1", "v1", "p2", "v2"]:
        df[column] = df[column].apply(literal_eval)

    # Split trials based on the dataset type
    split_dict = split_trials(
        df,
        dataset=dataset,
        test_size=test_size,
        random_state=random_state,
        by_progress=by_progress,
    )

    # Prepare data loaders based on model usage
    if model_usage == "symbolic":
        loaders = prepare_symbolic_dataloaders(
            df, split_dict, dataset, by_progress, transform=transform
        )
    else:  # 'NN'
        loaders = prepare_nn_dataloaders(
            df,
            split_dict,
            dataset,
            by_progress,
            batch_size=batch_size,
            transform=transform,
        )

    return loaders


def prepare_symbolic_dataloaders(df, all_splits, dataset, by_progress, transform=None):
    symbolic_loaders = []

    # Handling for small dataset LOOCV or large dataset without progression
    if dataset == "small" or (dataset == "large" and not by_progress):
        for split_dict in all_splits:
            round_loader = {}
            for participant_id, (train_df, test_df) in split_dict.items():
                train_dataset = BehavioralDataset(train_df, transform=transform)
                test_dataset = BehavioralDataset(test_df, transform=transform)
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=len(train_df),
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=0,
                )
                test_loader = DataLoader(
                    test_dataset,
                    batch_size=len(test_df),
                    shuffle=False,
                    collate_fn=collate_fn,
                    num_workers=0,
                )
                round_loader[participant_id] = (train_loader, test_loader)
            symbolic_loaders.append(round_loader)

    # Handling for large dataset with progression
    elif dataset == "large" and by_progress:
        for progressive_split in all_splits:
            progressive_loader = []
            for step_splits in progressive_split:
                step_loader = {}
                for rep, rep_splits in step_splits.items():
                    rep_loader = {}
                    for participant_id, (train_df, test_df) in enumerate(rep_splits):
                        train_dataset = BehavioralDataset(
                            train_df, transform=transform, dataset=dataset
                        )
                        test_dataset = BehavioralDataset(
                            test_df, transform=transform, dataset=dataset
                        )
                        train_loader = DataLoader(
                            train_dataset,
                            batch_size=len(train_df),
                            shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=0,
                        )
                        test_loader = DataLoader(
                            test_dataset,
                            batch_size=len(test_df),
                            shuffle=False,
                            collate_fn=collate_fn,
                            num_workers=0,
                        )
                        rep_loader[participant_id] = (train_loader, test_loader)
                    step_loader[rep] = rep_loader
                progressive_loader.append(step_loader)
            symbolic_loaders.append(progressive_loader)

    return symbolic_loaders


def prepare_nn_dataloaders(df, all_splits, dataset, by_progress, batch_size=4, transform=None):
    nn_loaders = []

    if dataset == "small" or (dataset == "large" and not by_progress):
        for round_num, split_dict in enumerate(all_splits):
            round_loaders = []
            aggregated_train_df = pd.concat([train_df for _, (train_df, _) in split_dict.items()])
            aggregated_test_df = pd.concat([test_df for _, (_, test_df) in split_dict.items()])
            train_dataset = BehavioralDataset(aggregated_train_df, transform=transform, dataset=dataset)
            test_dataset = BehavioralDataset(aggregated_test_df, transform=transform, dataset=dataset)
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                num_workers=0,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            round_loaders.append((train_loader, test_loader))
            nn_loaders.append(round_loaders)  # Append the loaders for this round to the main list
    elif dataset == "large" and by_progress:
        for round_num, progressive_split in enumerate(all_splits):  # Ensure rounds are distinguished
            round_loaders = []  # Loaders for this round
            for step_num, step_splits in enumerate(progressive_split):  # Each step within the round
                step_loaders = []  # Loaders for this step
                for rep_num, rep_splits in enumerate(step_splits.values()):  # Each repetition within the step
                    aggregated_train_dfs = [train_df for train_df, _ in rep_splits]
                    aggregated_test_dfs = [test_df for _, test_df in rep_splits]
                    aggregated_train_df = pd.concat(aggregated_train_dfs)
                    aggregated_test_df = pd.concat(aggregated_test_dfs)
                    train_dataset = BehavioralDataset(
                        aggregated_train_df, transform=transform, dataset=dataset
                    )
                    test_dataset = BehavioralDataset(
                        aggregated_test_df, transform=transform, dataset=dataset
                    )
                    train_loader = DataLoader(
                        train_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )
                    test_loader = DataLoader(
                        test_dataset,
                        batch_size=batch_size,
                        shuffle=False,
                        collate_fn=collate_fn,
                        num_workers=0,
                    )
                    step_loaders.append((train_loader, test_loader))  # Append loaders for this repetition to the step
                round_loaders.append(step_loaders)  # Append step loaders to the round
            nn_loaders.append(round_loaders)  # Append round loaders to the main list

    return nn_loaders


def split_trials(
    df,
    dataset="small",
    test_size=0.1,
    random_state=42,
    by_progress=False,
    steps=20,
    repetitions=10,
    num_samples=10,
):
    df["problem_id_encoded"] = LabelEncoder().fit_transform(df["problem_id"])
    df["sub_id_encoded"] = LabelEncoder().fit_transform(df["sub_id"])

    all_splits = []
    if by_progress:
        num_samples = 1

    if dataset == "small":
        # Handle small dataset with LOOCV
        loo = LeaveOneOut()
        for train_index, test_index in loo.split(df["problem_id_encoded"].unique()):
            split_dict = {}
            for participant_id in df["sub_id_encoded"].unique():
                participant_df = df[df["sub_id_encoded"] == participant_id]
                train_trials = participant_df.iloc[train_index]["problem_id_encoded"].unique()
                test_trials = participant_df.iloc[test_index]["problem_id_encoded"].unique()
                train_df = participant_df[participant_df["problem_id_encoded"].isin(train_trials)]
                test_df = participant_df[participant_df["problem_id_encoded"].isin(test_trials)]
                split_dict[participant_id] = (train_df, test_df)
            all_splits.append(split_dict)
    else:
        # Handle large dataset
        for rep in range(num_samples):
            # Perform initial train-test split
            initial_splits = {}
            for participant_id, group_df in df.groupby("sub_id_encoded"):
                participant_train_df, participant_test_df = train_test_split(
                    group_df, test_size=test_size, random_state=random_state + rep
                )
                initial_splits[participant_id] = (participant_train_df, participant_test_df)

            if by_progress:
                # If by_progress is True, sample progressively from the training data
                progressive_splits = []
                for step in np.linspace(0.05, 1, steps):
                    step_splits = {rep: [] for rep in range(repetitions)}
                    for rep in range(repetitions):
                        for participant_id, (train_df, test_df) in initial_splits.items():
                            sampled_train_size = int(len(train_df) * step)
                            sampled_train_df = train_df.sample(
                                n=sampled_train_size, random_state=random_state + rep
                            )
                            step_splits[rep].append((sampled_train_df, test_df))
                    progressive_splits.append(step_splits)
                all_splits.append(progressive_splits)
            else:
                # For fixed split without progression, use the initial splits
                all_splits.append(initial_splits)

    return all_splits


__all__ = [
    "PROJECT_ROOT",
    "BehavioralDataset",
    "collate_fn",
    "pad_tensor",
    "prepare_dataloaders",
    "prepare_symbolic_dataloaders",
    "prepare_nn_dataloaders",
    "split_trials",
]
