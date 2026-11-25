"""Tests de l'échantillonneur groupé par shard."""
from __future__ import annotations

import numpy as np

from src.train.dataset import ShardBatchSampler


def test_couvre_tout_sans_doublon():
    lengths = [1000, 700, 33]
    sampler = ShardBatchSampler(lengths, 64, shuffle=True, seed=3)
    seen = [index for batch in sampler for index in batch]
    assert len(seen) == sum(lengths)
    assert sorted(seen) == list(range(sum(lengths)))


def test_lots_groupes_par_shard():
    lengths = [1000, 700]
    sampler = ShardBatchSampler(lengths, 64, shuffle=True, seed=0)
    for batch in sampler:
        meme_shard = all(i < 1000 for i in batch) or all(i >= 1000 for i in batch)
        assert meme_shard, "un lot doit rester dans un seul shard"
        assert len(batch) <= 64


def test_ordre_change_a_chaque_epoque():
    sampler = ShardBatchSampler([500], 50, shuffle=True, seed=0)
    premiere = [index for batch in sampler for index in batch]
    seconde = [index for batch in sampler for index in batch]
    assert premiere != seconde


def test_longueur_en_lots():
    sampler = ShardBatchSampler([1000, 30], 64)
    assert len(sampler) == 16 + 1  # 1000 -> 16 lots, 30 -> 1 lot incomplet


def test_sans_melange_reste_ordonne():
    sampler = ShardBatchSampler([100, 100], 10, shuffle=False)
    premier_lot = next(iter(sampler))
    assert premier_lot == list(range(10))
