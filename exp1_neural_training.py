"""
Neural network baseline training (value-based / context-dependent NN models).

Data loading lives in ``utils.neural_data_loader`` (moved from former ``legacy/data_loader``).
Saves results under ``results/model_result/...`` with per-sample predictions for SEM.
"""

import sys
import os
import json
from typing import Dict, List, Tuple, Any
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.neural_data_loader import prepare_dataloaders
from models.neural_model import ValueBasedModel, ContextDependentModel


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Train neural baseline models (NN)")
    parser.add_argument("--data_size", type=str, choices=["small", "large"], default="large")
    parser.add_argument("--model_type", type=str, choices=["value_based", "context_dependent", "all"], default="value_based")
    parser.add_argument("--by_progress", action="store_true", help="Use progressive sampling (large dataset only)")
    parser.add_argument("--override", action="store_true", help="Overwrite existing results for this model")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2024, help="Random seed for data splits")
    parser.add_argument("--step_index", type=int, default=None, help="Run only this step index (0-based, by_progress only)")
    parser.add_argument("--last_step", action="store_true", help="Run only the last step (by_progress only)")
    parser.add_argument("--verbose", action="store_true", help="Enable INFO logging")
    parser.add_argument("--no_tqdm", action="store_true", help="Disable tqdm progress bars")
    return parser.parse_args()


def _get_logger(verbose: bool) -> logging.Logger:
    logger = logging.getLogger("exp1_neural_training")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s", datefmt="%H:%M:%S")
        handler.setFormatter(fmt)
        logger.addHandler(handler)
    logger.setLevel(logging.INFO if verbose else logging.WARNING)
    return logger


def get_model_class_and_config(model_type: str, lr: float) -> Tuple[Any, Dict[str, Any]]:
    # Base config for neural baselines
    config: Dict[str, Any] = {
        "input_size": 9,
        "hidden_size": 32,  # will be overwritten below
        "n_subjects": None,  # filled later from dataloader
        "sub_embed_dim": 16,
        "model_type": "NN",
        "lr": lr,
        "dropout": 0.0,
    }
    if model_type == "context_dependent":
        model_cls = ContextDependentModel
        config["dropout"] = 0.3
        config["hidden_size"] = 64
    else:
        model_cls = ValueBasedModel
        config["dropout"] = 0.1
        config["hidden_size"] = 128
    return model_cls, config


def _results_root_and_file(data_size: str, by_progress: bool) -> Tuple[str, str]:
    base = os.path.join("results", "model_result", f"{data_size}_dataset")
    if data_size == "large":
        sub = "by_progress" if by_progress else "at_once"
        base = os.path.join(base, sub)
    os.makedirs(base, exist_ok=True)
    results_file = os.path.join(base, f"NN_model_detailed_results_{data_size}.json")
    return base, results_file


def _checkpoint_dir(results_root: str, model_name: str, round_idx: int, step_idx: int | None = None, sample_idx: int | None = None) -> str:
    ckpt_dir = os.path.join(results_root, "model_checkpoints", model_name, f"round_{round_idx}")
    if step_idx is not None:
        ckpt_dir = os.path.join(ckpt_dir, f"step_{step_idx}")
        if sample_idx is not None:
            ckpt_dir = os.path.join(ckpt_dir, f"sample_{sample_idx}")
    os.makedirs(ckpt_dir, exist_ok=True)
    return ckpt_dir


def _first_train_loader(dataloaders: List) -> Any:
    # small or large-at_once: [ [ (train_loader, test_loader) ] ]
    # large-by_progress: [ [ [ (train_loader, test_loader), ... ] , ... ] ]
    first = dataloaders[0]
    if isinstance(first, list) and len(first) > 0:
        if isinstance(first[0], list):
            # by_progress
            return first[0][0][0]
        else:
            # at_once/small
            return first[0][0]
    # Fallback
    return first


def _bce_loss_on_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    # Ensure 1D logit per sample
    if logits.dim() > 1 and logits.size(-1) > 1:
        logits = logits[..., -1]
    logits = logits.squeeze()
    criterion = nn.BCEWithLogitsLoss()
    return criterion(logits, targets)


def _sigmoid_prob(logits: torch.Tensor) -> torch.Tensor:
    if logits.dim() > 1 and logits.size(-1) > 1:
        logits = logits[..., -1]
    return torch.sigmoid(logits.squeeze())


def _evaluate_collect(model: nn.Module, test_loader: Any, device: torch.device, include_losses: bool = False) -> Tuple[float, float, List[Dict[str, Any]]]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    predictions: List[Dict[str, Any]] = []
    with torch.no_grad():
        # We rely on test_loader.dataset.df order (shuffle=False)
        df = getattr(test_loader.dataset, "df", None)
        if df is not None:
            df_ordered = df.reset_index(drop=True)
        offset = 0
        for inputs, targets in test_loader:
            inputs = [tensor.to(device) for tensor in inputs]
            targets = targets.to(device)
            outputs = model(inputs if len(inputs) > 1 else inputs[0])
            loss = _bce_loss_on_logits(outputs, targets)
            total_loss += loss.item()
            probs = _sigmoid_prob(outputs)
            preds = (probs > 0.5).to(targets.dtype)
            correct += (preds == targets).sum().item()
            batch_size = targets.size(0)
            total += batch_size

            # Collect per-sample metadata if df available
            if df is not None:
                # Optionally compute per-sample losses
                loss_vec = None
                if include_losses:
                    # Compute unreduced BCE per-sample
                    logits_vec = outputs
                    if logits_vec.dim() > 1 and logits_vec.size(-1) > 1:
                        logits_vec = logits_vec[..., -1]
                    logits_vec = logits_vec.squeeze()
                    targets_vec = targets
                    loss_vec = F.binary_cross_entropy_with_logits(logits_vec, targets_vec.float(), reduction='none').detach().cpu()

                for i in range(batch_size):
                    row_idx = offset + i
                    row = df_ordered.iloc[row_idx]
                    predictions.append({
                        "problem_id": row.get("problem_id"),
                        "sub_id": int(row.get("sub_id_encoded")) if "sub_id_encoded" in row else int(row.get("sub_id", -1)),
                        "predicted_choice": int(preds[i].item()),
                        "probability_option_b": float(probs[i].item()),
                        "actual_choice": int(targets[i].item()),
                        **({"loss": float(loss_vec[i].item())} if include_losses and loss_vec is not None else {}),
                    })
                offset += batch_size

    avg_loss = total_loss / max(1, len(test_loader))
    acc = correct / max(1, total)
    return avg_loss, acc, predictions


def _make_eval_loader_from(loader: Any) -> Any:
    try:
        bs = loader.batch_size if loader.batch_size is not None else 64
    except Exception:
        bs = 64
    cf = getattr(loader, 'collate_fn', None)
    ds = loader.dataset
    return DataLoader(ds, batch_size=bs, shuffle=False, collate_fn=cf, num_workers=0)


def _train_one(model: nn.Module, train_loader: Any, device: torch.device, epochs: int, lr: float, logger: logging.Logger, show_tqdm: bool) -> float:
    model.train()
    optimizer = Adam(model.parameters(), lr=lr)
    last_epoch_loss = 0.0
    epoch_iter = range(epochs)
    if show_tqdm and tqdm is not None:
        epoch_iter = tqdm(epoch_iter, desc="Epochs", leave=False)
    for _ in epoch_iter:
        running = 0.0
        num_batches = 0
        batch_iter = train_loader
        if show_tqdm and tqdm is not None:
            try:
                total_batches = len(train_loader)
            except Exception:
                total_batches = None
            batch_iter = tqdm(train_loader, total=total_batches, desc="Batches", leave=False)
        for inputs, targets in batch_iter:
            inputs = [tensor.to(device) for tensor in inputs]
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs if len(inputs) > 1 else inputs[0])
            loss = _bce_loss_on_logits(outputs, targets)
            loss.backward()
            optimizer.step()
            running += loss.item()
            num_batches += 1
        last_epoch_loss = running / max(1, num_batches)
    return last_epoch_loss


def run_training(data_size: str, model_type: str, by_progress: bool, override: bool, epochs: int, batch_size: int, lr: float, seed: int,
                 step_index: int | None = None, last_step: bool = False,
                 verbose: bool = False, no_tqdm: bool = False) -> None:
    logger = _get_logger(verbose)
    show_tqdm = (not no_tqdm)
    results_root, results_file = _results_root_and_file(data_size, by_progress)
    # Load or init results file
    if os.path.exists(results_file):
        with open(results_file, "r", encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}

    model_cls, model_config = get_model_class_and_config(model_type, lr)
    model_name = model_cls.__name__
    if (not override) and (model_name in all_results):
        logger.info(f"Model {model_name} results already exist and override is False. Skipping...")
        return

    # Prepare dataloaders
    logger.info(f"Preparing dataloaders (data_size={data_size}, by_progress={by_progress}, seed={seed})...")
    dataloaders = prepare_dataloaders(dataset=data_size, model_usage='NN', by_progress=by_progress, batch_size=batch_size, random_state=seed)
    logger.info("Dataloaders ready.")

    # Fill n_subjects from dataset
    first_train = _first_train_loader(dataloaders)
    try:
        n_subs = int(getattr(first_train.dataset, 'num_of_subs'))
    except Exception:
        # Fallback by peeking a batch
        inputs, _ = next(iter(first_train))
        # inputs structure: (p1, v1, p2, v2, individual_ids, num_of_sub, embeddings)
        n_subs = int(inputs[5][0].item())
    model_config["n_subjects"] = n_subs
    logger.info(f"Model={model_name} | n_subjects={n_subs} | hidden_size={model_config['hidden_size']} | dropout={model_config['dropout']}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    per_model_results: Dict[str, Any] = {}

    # Iterate rounds
    # Determine rounds to run
    total_rounds = len(dataloaders)
    logger.info(f"Total rounds available: {total_rounds}")
    rounds_to_run = list(range(total_rounds))
    logger.info(f"Rounds to run: {rounds_to_run}")

    for round_idx in rounds_to_run:
        round_data = dataloaders[round_idx]
        round_results: Dict[str, Any] = {}

        if data_size == 'large' and by_progress:
            # Multiple steps, each with multiple samples
            total_steps = len(round_data)
            logger.info(f"Round {round_idx}: total steps available: {total_steps}")
            if last_step:
                steps_to_run = [total_steps - 1]
            elif step_index is not None:
                if step_index < 0 or step_index >= total_steps:
                    raise IndexError(f"step_index {step_index} out of range [0, {total_steps-1}]")
                steps_to_run = [step_index]
            else:
                steps_to_run = list(range(total_steps))
            logger.info(f"Round {round_idx}: steps to run: {steps_to_run}")

            for step_idx in steps_to_run:
                step_loaders = round_data[step_idx]
                step_results: Dict[str, Any] = {}
                # Reduce to a single sample when only running the last step, since content is identical at 100%
                if last_step:
                    samples_to_run = [0]
                else:
                    samples_to_run = list(range(len(step_loaders)))
                for sample_idx in samples_to_run:
                    train_loader, test_loader = step_loaders[sample_idx]
                    try:
                        num_train_batches = len(train_loader)
                        num_test_batches = len(test_loader)
                    except Exception:
                        num_train_batches = num_test_batches = -1
                    logger.info(f"Round {round_idx} | Step {step_idx} | Sample {sample_idx}: train_batches={num_train_batches}, test_batches={num_test_batches}")
                    model = model_cls(model_config).to(device)
                    train_loss = _train_one(model, train_loader, device, epochs, lr, logger, show_tqdm)
                    # Evaluate on train (with shuffle disabled) and test
                    train_eval_loader = _make_eval_loader_from(train_loader)
                    train_loss_eval, train_acc_eval, train_preds = _evaluate_collect(model, train_eval_loader, device, include_losses=True)
                    test_loss, test_acc, test_preds = _evaluate_collect(model, test_loader, device, include_losses=True)
                    ckpt_dir = _checkpoint_dir(results_root, model_name, round_idx, step_idx, sample_idx)
                    ckpt_path = os.path.join(ckpt_dir, 'model.pth')
                    torch.save(model.state_dict(), ckpt_path)
                    logger.info(f"Saved checkpoint: {ckpt_path}")
                    step_results[sample_idx] = {
                        'train_loss': train_loss,
                        'train_eval_loss': train_loss_eval,
                        'train_eval_accuracy': train_acc_eval,
                        'test_loss': test_loss,
                        'test_accuracy': test_acc,
                        'checkpoint': ckpt_path,
                        'train_predictions': train_preds,
                        'test_predictions': test_preds,
                    }
                round_results[f'step_{step_idx}'] = step_results
        else:
            # Small or large-at_once: single pair per round
            train_loader, test_loader = round_data[0]
            try:
                num_train_batches = len(train_loader)
                num_test_batches = len(test_loader)
            except Exception:
                num_train_batches = num_test_batches = -1
            logger.info(f"Round {round_idx}: train_batches={num_train_batches}, test_batches={num_test_batches}")
            model = model_cls(model_config).to(device)
            train_loss = _train_one(model, train_loader, device, epochs, lr, logger, show_tqdm)
            train_eval_loader = _make_eval_loader_from(train_loader)
            train_loss_eval, train_acc_eval, train_preds = _evaluate_collect(model, train_eval_loader, device, include_losses=True)
            test_loss, test_acc, test_preds = _evaluate_collect(model, test_loader, device, include_losses=True)
            ckpt_dir = _checkpoint_dir(results_root, model_name, round_idx)
            ckpt_path = os.path.join(ckpt_dir, 'model.pth')
            torch.save(model.state_dict(), ckpt_path)
            logger.info(f"Saved checkpoint: {ckpt_path}")
            round_results = {
                'train_loss': train_loss,
                'train_eval_loss': train_loss_eval,
                'train_eval_accuracy': train_acc_eval,
                'test_loss': test_loss,
                'test_accuracy': test_acc,
                'checkpoint': ckpt_path,
                'train_predictions': train_preds,
                'test_predictions': test_preds,
            }

        per_model_results[f'round_{round_idx}'] = round_results

    # Merge and save
    all_results[model_name] = per_model_results
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4)
    logger.info(f"Saved results to {results_file}")


def main():
    args = parse_args()
    if args.model_type == "all":
        for mt in ["value_based", "context_dependent"]:
            run_training(
                data_size=args.data_size,
                model_type=mt,
                by_progress=args.by_progress,
                override=args.override,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                seed=args.seed,
                step_index=args.step_index,
                last_step=args.last_step,
                verbose=args.verbose,
                no_tqdm=args.no_tqdm,
            )
    else:
        run_training(
            data_size=args.data_size,
            model_type=args.model_type,
            by_progress=args.by_progress,
            override=args.override,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            seed=args.seed,
            step_index=args.step_index,
            last_step=args.last_step,
            verbose=args.verbose,
            no_tqdm=args.no_tqdm,
        )


if __name__ == "__main__":
    main()