"""
data_loader.py
==============
Handles all data loading, tokenization, GloVe embedding loading,
and behavior parsing for the MIND-small news recommendation dataset.
"""

import os
import pickle
import random

import nltk
import numpy as np
import pandas as pd
from collections import Counter
from torch.utils.data import Dataset
import torch

nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NEWS_COLS = ['news_id', 'category', 'subcategory', 'title',
             'abstract', 'url', 'title_entities', 'abstract_entities']
BEH_COLS  = ['impression_id', 'user_id', 'time', 'history', 'impressions']


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class NewsTokenizer:
    """
    Builds a word vocabulary from news titles and encodes titles as
    fixed-length sequences of integer word indices.

    Args:
        max_title_len (int): Maximum number of tokens per title.
        min_word_freq (int): Minimum frequency for a word to enter the vocab.
    """

    def __init__(self, max_title_len: int = 30, min_word_freq: int = 2):
        self.max_title_len = max_title_len
        self.min_word_freq = min_word_freq
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}

    # ------------------------------------------------------------------
    def _tokenize(self, text: str) -> list:
        if not isinstance(text, str) or not text.strip():
            return []
        return nltk.word_tokenize(text.lower())

    # ------------------------------------------------------------------
    def build_vocab(self, titles: list) -> None:
        """Builds word2idx from a list of title strings (training set only)."""
        counts = Counter()
        for title in titles:
            counts.update(self._tokenize(title))
        for word, cnt in counts.items():
            if cnt >= self.min_word_freq:
                self.word2idx[word] = len(self.word2idx)
        print(f'[Tokenizer] Vocabulary size: {len(self.word2idx):,}')

    # ------------------------------------------------------------------
    def encode_title(self, title: str) -> list:
        """Converts a title string to a padded list of word indices."""
        tokens  = self._tokenize(title)
        indices = [self.word2idx.get(t, 1) for t in tokens]
        indices = indices[:self.max_title_len]
        indices += [0] * (self.max_title_len - len(indices))
        return indices

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.word2idx)


# ---------------------------------------------------------------------------
# GloVe loader
# ---------------------------------------------------------------------------
def load_glove(glove_path: str, word2idx: dict, embed_dim: int = 300) -> np.ndarray:
    """
    Reads GloVe vectors and returns an embedding matrix of shape
    (vocab_size, embed_dim).  Words not found in GloVe are randomly
    initialised with small values; the PAD row is kept as zeros.

    Args:
        glove_path: Path to the glove.6B.300d.txt file.
        word2idx:   Vocabulary mapping word → index.
        embed_dim:  Dimensionality of GloVe vectors (default 300).

    Returns:
        np.ndarray of shape (len(word2idx), embed_dim), dtype float32.
    """
    vocab_size = len(word2idx)
    matrix     = (np.random.randn(vocab_size, embed_dim) * 0.01).astype('float32')
    matrix[0]  = 0.0   # PAD stays zero

    found = 0
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.rstrip().split(' ')
            word  = parts[0]
            if word in word2idx:
                matrix[word2idx[word]] = np.array(parts[1:], dtype='float32')
                found += 1

    print(f'[GloVe] Coverage: {found}/{vocab_size} '
          f'({found / vocab_size * 100:.1f}%)')
    return matrix


# ---------------------------------------------------------------------------
# News encoding
# ---------------------------------------------------------------------------
def encode_news(news_df: pd.DataFrame, tokenizer: NewsTokenizer) -> dict:
    """
    Encodes every article title in news_df using the given tokenizer.

    Returns:
        dict mapping news_id → list[int] of length max_title_len.
    """
    encoded = {}
    for _, row in news_df.iterrows():
        encoded[row['news_id']] = tokenizer.encode_title(row['title'])
    print(f'[encode_news] Encoded {len(encoded):,} articles.')
    return encoded


# ---------------------------------------------------------------------------
# Behavior parsers
# ---------------------------------------------------------------------------
def parse_behaviors_train(
    behaviors_df: pd.DataFrame,
    news_encoded: dict,
    max_history: int = 50,
    neg_k: int = 4,
    seed: int = 42,
) -> list:
    """
    Converts training behavior logs into model-ready samples.

    Each positive click is paired with `neg_k` randomly sampled negatives
    from the same impression.  Returns a list of dicts with keys:
        history    : np.int32 array (max_history, title_len)
        hist_mask  : np.int32 array (max_history,)  — 1=real, 0=pad
        candidates : np.int32 array (1+neg_k, title_len)
        labels     : list[int]  — always [1, 0, 0, ...]
    """
    pad_title = [0] * (next(iter(news_encoded.values())).__len__())
    rng       = np.random.default_rng(seed)
    samples   = []

    for _, row in behaviors_df.iterrows():
        # History
        hist_ids  = (row['history'].split()
                     if pd.notna(row['history']) and str(row['history']).strip()
                     else [])
        hist_ids  = hist_ids[-max_history:]
        hist_enc  = [news_encoded.get(nid, pad_title) for nid in hist_ids]
        hist_len  = len(hist_enc)
        pad_count = max_history - hist_len
        hist_enc  = hist_enc + [pad_title] * pad_count
        hist_mask = [1] * hist_len + [0] * pad_count

        # Impressions
        if pd.isna(row['impressions']) or not str(row['impressions']).strip():
            continue

        pos_ids, neg_ids = [], []
        for item in str(row['impressions']).split():
            nid, label = item.rsplit('-', 1)
            (pos_ids if label == '1' else neg_ids).append(nid)

        if not pos_ids or not neg_ids:
            continue

        for pos_id in pos_ids:
            k_actual = min(neg_k, len(neg_ids))
            sampled  = rng.choice(neg_ids, size=k_actual, replace=False).tolist()
            cand_ids = [pos_id] + sampled
            # Pad to fixed size if not enough negatives
            while len(cand_ids) < 1 + neg_k:
                cand_ids.append(pos_id)

            candidates = [news_encoded.get(c, pad_title) for c in cand_ids]
            samples.append({
                'history'   : np.array(hist_enc,   dtype=np.int32),
                'hist_mask' : np.array(hist_mask,  dtype=np.int32),
                'candidates': np.array(candidates, dtype=np.int32),
                'labels'    : [1] + [0] * (len(cand_ids) - 1),
            })

    print(f'[parse_train] {len(samples):,} training samples generated.')
    return samples


def parse_behaviors_dev(
    behaviors_df: pd.DataFrame,
    news_encoded: dict,
    max_history: int = 50,
) -> list:
    """
    Converts dev/validation behavior logs into evaluation samples.

    Unlike training, ALL candidates per impression are kept (no negative
    sampling) so that ranking metrics (AUC, MRR, nDCG) can be computed.
    Degenerate impressions (all positive or all negative) are skipped.
    """
    pad_title = [0] * (next(iter(news_encoded.values())).__len__())
    samples   = []

    for _, row in behaviors_df.iterrows():
        hist_ids  = (row['history'].split()
                     if pd.notna(row['history']) and str(row['history']).strip()
                     else [])
        hist_ids  = hist_ids[-max_history:]
        hist_enc  = [news_encoded.get(nid, pad_title) for nid in hist_ids]
        hist_len  = len(hist_enc)
        pad_count = max_history - hist_len
        hist_enc  = hist_enc + [pad_title] * pad_count
        hist_mask = [1] * hist_len + [0] * pad_count

        if pd.isna(row['impressions']) or not str(row['impressions']).strip():
            continue

        cand_ids, labels = [], []
        for item in str(row['impressions']).split():
            nid, label = item.rsplit('-', 1)
            cand_ids.append(nid)
            labels.append(int(label))

        if sum(labels) == 0 or sum(labels) == len(labels):
            continue

        candidates = [news_encoded.get(c, pad_title) for c in cand_ids]
        samples.append({
            'history'   : np.array(hist_enc,  dtype=np.int32),
            'hist_mask' : np.array(hist_mask, dtype=np.int32),
            'candidates': candidates,   # variable length list
            'labels'    : labels,
        })

    print(f'[parse_dev] {len(samples):,} dev samples generated.')
    return samples


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------
class MINDTrainDataset(Dataset):
    """
    PyTorch Dataset wrapping fixed-size training samples
    (1 positive + NEG_K negatives per sample).
    """

    def __init__(self, samples: list):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            'history'   : torch.tensor(s['history'],    dtype=torch.long),
            'hist_mask' : torch.tensor(s['hist_mask'],  dtype=torch.long),
            'candidates': torch.tensor(s['candidates'], dtype=torch.long),
            'label'     : torch.tensor(0,               dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def save_processed(out_dir: str, train_samples: list, dev_samples: list,
                   embedding_matrix: np.ndarray, tokenizer: NewsTokenizer,
                   news_encoded: dict) -> None:
    """Saves all processed artefacts to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/train_samples.pkl',  'wb') as f: pickle.dump(train_samples,    f)
    with open(f'{out_dir}/dev_samples.pkl',    'wb') as f: pickle.dump(dev_samples,      f)
    with open(f'{out_dir}/tokenizer.pkl',      'wb') as f: pickle.dump(tokenizer,        f)
    with open(f'{out_dir}/news_encoded.pkl',   'wb') as f: pickle.dump(news_encoded,     f)
    np.save(f'{out_dir}/embedding_matrix.npy', embedding_matrix)
    print(f'[save_processed] Saved all artefacts to {out_dir}/')


def load_processed(out_dir: str) -> tuple:
    """Loads all processed artefacts from out_dir. Returns (train, dev, emb, tok, enc)."""
    with open(f'{out_dir}/train_samples.pkl',  'rb') as f: train_samples    = pickle.load(f)
    with open(f'{out_dir}/dev_samples.pkl',    'rb') as f: dev_samples      = pickle.load(f)
    with open(f'{out_dir}/tokenizer.pkl',      'rb') as f: tokenizer        = pickle.load(f)
    with open(f'{out_dir}/news_encoded.pkl',   'rb') as f: news_encoded     = pickle.load(f)
    embedding_matrix = np.load(f'{out_dir}/embedding_matrix.npy')
    return train_samples, dev_samples, embedding_matrix, tokenizer, news_encoded
