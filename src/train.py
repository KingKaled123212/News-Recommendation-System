"""
train.py
========
Training script for the NRMS news recommendation model.

Usage (from project root):
    python src/train.py

Trains on MIND-small train split, evaluates after every epoch on the dev
split, saves the best checkpoint to models/best_model.pt, and writes a
training curve plot to results/training_curve.png.
"""

import os
import sys
import time
import pickle
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Allow imports from src/ when running as a script
sys.path.insert(0, os.path.dirname(__file__))

from data_loader import MINDTrainDataset, load_processed
from model import NRMSModel
from evaluate import evaluate

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CONFIG = {
    'data_dir'   : 'data/processed',
    'model_dir'  : 'models',
    'results_dir': 'results',
    # Model
    'num_heads'  : 16,
    'head_dim'   : 16,
    'dropout'    : 0.2,
    # Training
    'batch_size' : 64,
    'lr'         : 1e-4,
    'epochs'     : 5,
    'clip_grad'  : 1.0,
    # LR scheduler: multiply LR by gamma every step_size epochs
    'lr_step'    : 2,
    'lr_gamma'   : 0.5,
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(config: dict = CONFIG) -> dict:
    """
    Full training run.

    Args:
        config: Hyperparameter dict (see CONFIG above).

    Returns:
        history dict with keys 'train_loss' and 'val_metrics'.
    """
    os.makedirs(config['model_dir'],   exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[train] Device: {device}')
    if device.type == 'cuda':
        print(f'[train] GPU: {torch.cuda.get_device_name(0)}')

    # ── Load data ──────────────────────────────────────────────────────
    print('[train] Loading processed data...')
    train_samples, dev_samples, embedding_matrix, _, _ = \
        load_processed(config['data_dir'])

    loader = DataLoader(
        MINDTrainDataset(train_samples),
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == 'cuda'),
    )
    print(f'[train] Train batches per epoch: {len(loader):,}')

    # ── Model ──────────────────────────────────────────────────────────
    model = NRMSModel(
        embedding_matrix,
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        dropout=config['dropout'],
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[train] Trainable parameters: {total_params:,}')

    # ── Optimiser & scheduler ──────────────────────────────────────────
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=config['lr_step'], gamma=config['lr_gamma'])
    criterion = nn.CrossEntropyLoss()

    # ── Training loop ──────────────────────────────────────────────────
    history   = {'train_loss': [], 'val_metrics': []}
    best_auc  = 0.0
    ckpt_path = os.path.join(config['model_dir'], 'best_model.pt')

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0.0
        t0         = time.time()

        for batch in loader:
            hist = batch['history'].to(device)
            cand = batch['candidates'].to(device)
            mask = batch['hist_mask'].to(device)
            tgt  = batch['label'].to(device)

            optimizer.zero_grad()
            scores = model(hist, cand, mask)
            loss   = criterion(scores, tgt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), config['clip_grad'])
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history['train_loss'].append(avg_loss)
        scheduler.step()

        elapsed = time.time() - t0
        print(f'\n[Epoch {epoch+1}/{config["epochs"]}] '
              f'Loss: {avg_loss:.4f}  LR: {scheduler.get_last_lr()[0]:.2e}  '
              f'({elapsed:.0f}s)')

        # Evaluate
        print('[train] Evaluating on dev set...')
        metrics = evaluate(model, dev_samples, device)
        history['val_metrics'].append(metrics)

        if metrics['AUC'] > best_auc:
            best_auc = metrics['AUC']
            torch.save(model.state_dict(), ckpt_path)
            print(f'[train] ✓ Best AUC={best_auc:.4f} — saved to {ckpt_path}')

    # ── Plot training curve ────────────────────────────────────────────
    _plot_history(history, config['results_dir'])
    print(f'\n[train] Done. Best validation AUC: {best_auc:.4f}')
    return history


# ---------------------------------------------------------------------------
# Plotting helper
# ---------------------------------------------------------------------------
def _plot_history(history: dict, results_dir: str) -> None:
    epochs = range(1, len(history['train_loss']) + 1)
    aucs   = [m['AUC'] for m in history['val_metrics']]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history['train_loss'], 'o-', color='steelblue')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Cross-Entropy Loss')

    axes[1].plot(epochs, aucs, 'o-', color='coral')
    axes[1].set_title('Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('AUC')
    axes[1].set_ylim(0.5, 0.75)

    plt.tight_layout()
    out = os.path.join(results_dir, 'training_curve.png')
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    print(f'[train] Training curve saved to {out}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    train(CONFIG)
