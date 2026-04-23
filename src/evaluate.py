"""
evaluate.py
===========
Evaluation script for the NRMS model on the MIND-small dev set.

Metrics computed per impression, then averaged:
    AUC     — Area Under the ROC Curve
    MRR     — Mean Reciprocal Rank
    nDCG@5  — Normalised Discounted Cumulative Gain at rank 5
    nDCG@10 — Normalised Discounted Cumulative Gain at rank 10

Usage (from project root):
    python src/evaluate.py

Or import and call evaluate() directly from train.py or a notebook.
"""

import os
import sys
import json

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Metric functions
# ---------------------------------------------------------------------------

def dcg_score(y_true: list, y_score: list, k: int = 10) -> float:
    """
    Discounted Cumulative Gain at rank k.

    Ranks articles by y_score descending, then sums relevance gains
    weighted by 1/log2(rank+1).
    """
    order  = np.argsort(y_score)[::-1][:k]
    gains  = np.array(y_true)[order].astype(float)
    disc   = np.log2(np.arange(len(gains)) + 2)
    return float(np.sum(gains / disc))


def ndcg_score(y_true: list, y_score: list, k: int = 10) -> float:
    """
    Normalised DCG at rank k.  Divides DCG by the ideal (perfect) DCG.
    Returns 0.0 if the ideal DCG is zero.
    """
    best = dcg_score(y_true, y_true, k)
    return dcg_score(y_true, y_score, k) / best if best > 0 else 0.0


def mrr_score(y_true: list, y_score: list) -> float:
    """
    Mean Reciprocal Rank.

    Returns 1/(rank of first relevant article in the ranked list).
    Returns 0.0 if no relevant article is found.
    """
    order    = np.argsort(y_score)[::-1]
    y_sorted = np.array(y_true)[order]
    for i, val in enumerate(y_sorted):
        if val == 1:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(model: torch.nn.Module,
             dev_samples: list,
             device: torch.device) -> dict:
    """
    Evaluate a trained NRMSModel on a list of dev samples.

    Each dev sample contains the full candidate list for one impression
    (variable length, no negative sampling).  Impressions where all
    candidates are positive or all are negative are skipped.

    Args:
        model       : Trained NRMSModel in eval mode.
        dev_samples : Output of parse_behaviors_dev().
        device      : torch.device to run inference on.

    Returns:
        dict with keys 'AUC', 'MRR', 'nDCG@5', 'nDCG@10'.
    """
    model.eval()
    aucs, mrrs, ndcg5s, ndcg10s = [], [], [], []

    with torch.no_grad():
        for s in dev_samples:
            y_true = s['labels']

            # Skip degenerate impressions
            if sum(y_true) == 0 or sum(y_true) == len(y_true):
                continue

            hist = torch.tensor(
                s['history'],    dtype=torch.long).unsqueeze(0).to(device)
            mask = torch.tensor(
                s['hist_mask'],  dtype=torch.long).unsqueeze(0).to(device)
            cand = torch.tensor(
                np.array(s['candidates']), dtype=torch.long
            ).unsqueeze(0).to(device)

            scores  = model(hist, cand, mask).squeeze(0).cpu().numpy()
            y_score = scores[:len(y_true)]

            aucs.append(roc_auc_score(y_true, y_score))
            mrrs.append(mrr_score(y_true, y_score))
            ndcg5s.append(ndcg_score(y_true, y_score, 5))
            ndcg10s.append(ndcg_score(y_true, y_score, 10))

    results = {
        'AUC'    : float(np.mean(aucs)),
        'MRR'    : float(np.mean(mrrs)),
        'nDCG@5' : float(np.mean(ndcg5s)),
        'nDCG@10': float(np.mean(ndcg10s)),
    }

    print(f"  AUC     : {results['AUC']:.4f}")
    print(f"  MRR     : {results['MRR']:.4f}")
    print(f"  nDCG@5  : {results['nDCG@5']:.4f}")
    print(f"  nDCG@10 : {results['nDCG@10']:.4f}")

    return results


# ---------------------------------------------------------------------------
# Entry point — run standalone evaluation from saved checkpoint
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import pickle
    import numpy as np
    from data_loader import load_processed
    from model import NRMSModel

    DATA_DIR  = 'data/processed'
    MODEL_DIR = 'models'
    OUT_DIR   = 'results'

    NUM_HEADS = 16
    HEAD_DIM  = 16
    DROPOUT   = 0.2

    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[evaluate] Device: {device}')

    # Load data
    _, dev_samples, embedding_matrix, _, _ = load_processed(DATA_DIR)

    # Load model
    model = NRMSModel(embedding_matrix, NUM_HEADS, HEAD_DIM, DROPOUT).to(device)
    ckpt  = os.path.join(MODEL_DIR, 'best_model.pt')
    model.load_state_dict(torch.load(ckpt, map_location=device))
    print(f'[evaluate] Loaded checkpoint: {ckpt}')

    # Evaluate
    print('[evaluate] Running evaluation...')
    results = evaluate(model, dev_samples, device)

    # Save results
    out_path = os.path.join(OUT_DIR, 'eval_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[evaluate] Results saved to {out_path}')
