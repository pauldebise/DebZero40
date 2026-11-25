"""Dataset PyTorch pour les shards ``.npz`` produits par ``tools/lc0_to_shards.py``.

Un shard contient :

* ``planes``  : ``uint8``  ``(N, 17, 8, 8)`` (NCHW, comme stocké par l'outil) ;
* ``pol_idx`` : ``uint16`` ``(N, 64)`` indices des coups dans la table 1858 ;
* ``pol_prob``: ``float16`` ``(N, 64)`` probabilités associées ;
* ``pol_len`` : ``uint8``  ``(N,)`` nombre de coups effectivement présents ;
* ``wdl``     : ``float16`` ``(N, 3)`` distribution win/draw/loss.

Le réseau attend une entrée **NHWC** ``(B, 8, 8, 17)`` et un masque de légalité
``(B, 1858)``.  ``ShardDataset`` renvoie une position à la fois ; le ``collate_fn``
``collate_shard_batch`` construit la cible policy dense et renormalisée.

Un seul shard est gardé en mémoire à la fois (un shard de 200 000 positions pèse
environ 200 Mo : les charger tous serait intenable).
"""
from __future__ import annotations

import bisect
import glob
import os
from typing import Dict, Iterable, List, Sequence, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset

POLICY_SIZE = 1858
DEFAULT_PLANES = 17
MAX_MOVES = 64


def find_shards(path: Union[str, os.PathLike]) -> List[str]:
    """Liste triée des ``.npz`` d'un répertoire (ou normalise une liste de fichiers)."""
    if isinstance(path, (str, os.PathLike)):
        path = os.fspath(path)
        if os.path.isdir(path):
            shards = sorted(glob.glob(os.path.join(path, "*.npz")))
        else:
            shards = [path]
    else:
        shards = [os.fspath(p) for p in path]
    shards = [p for p in shards if os.path.abspath(p) != os.path.abspath("manifest.json")]
    if not shards:
        raise FileNotFoundError(f"Aucun shard .npz trouvé dans {path!r}")
    return shards


def _npy_length(path: str) -> int:
    """Lit uniquement l'en-tête d'un ``.npz`` pour connaître le nombre de positions."""
    with np.load(path) as data:
        if "planes" not in data:
            raise KeyError(f"{path}: clé 'planes' absente")
        return int(data["planes"].shape[0])


class ShardDataset(Dataset):
    """Dataset indexable sur un ensemble de shards ``.npz``.

    Parameters
    ----------
    shards:
        Répertoire contenant les shards, ou séquence de chemins.
    planes:
        Nombre de plans à charger (17 par défaut).  Les canaux au-delà sont ignorés.
    policy_size:
        Taille de la sortie policy (1858).
    max_moves:
        Nombre maximum d'entrées policy par position (64 dans les shards).
    shuffle:
        Mélange l'ordre des positions au moment de la construction.  L'entraînement
        utilise plutôt ``DataLoader(shuffle=True)``, qui remélange à chaque époque.
    seed:
        Graine du mélange initial.
    """

    def __init__(
        self,
        shards: Union[str, os.PathLike, Sequence[Union[str, os.PathLike]]],
        planes: int = DEFAULT_PLANES,
        policy_size: int = POLICY_SIZE,
        max_moves: int = MAX_MOVES,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        self.shards = find_shards(shards)
        self.planes = int(planes)
        self.policy_size = int(policy_size)
        self.max_moves = int(max_moves)

        self._lengths = [_npy_length(p) for p in self.shards]
        self._cumulative: List[int] = []
        total = 0
        for n in self._lengths:
            total += n
            self._cumulative.append(total)
        self._total = total

        self._cached_shard: int | None = None
        self._cache: Dict[str, torch.Tensor | int] | None = None
        self._cache_dict: Dict[int, Dict[str, torch.Tensor | int]] = {}

        self._order: np.ndarray | None = None
        if shuffle:
            rng = np.random.default_rng(seed)
            self._order = rng.permutation(total).astype(np.int64)

    # ------------------------------------------------------------------ infos
    def __len__(self) -> int:
        return self._total

    @property
    def shard_lengths(self) -> List[int]:
        return list(self._lengths)

    # --------------------------------------------------------------- chargement
    def _load_shard(self, shard_index: int) -> None:
        # Cache par shard, SANS éviction : avec un mélange global, un cache à une seule
        # entrée rechargeait un shard de 65 Mo presque à chaque échantillon (2,6 Go lus
        # par lot, entraînement bloqué à 0 % d'utilisation GPU).
        cached = self._cache_dict.get(shard_index)
        if cached is not None:
            self._cache = cached
            self._cached_shard = shard_index
            return
        path = self.shards[shard_index]
        with np.load(path) as data:
            # np.load est paresseux sur un .npz : la conversion ci-dessous force la
            # décompression du shard courant uniquement.
            planes = np.asarray(data["planes"])
            pol_idx = np.asarray(data["pol_idx"])
            pol_prob = np.asarray(data["pol_prob"])
            pol_len = np.asarray(data["pol_len"])
            wdl = np.asarray(data["wdl"])

        if planes.ndim != 4 or planes.shape[1] != self.planes:
            if planes.ndim == 4 and planes.shape[1] >= self.planes:
                planes = planes[:, : self.planes]
            else:
                raise ValueError(
                    f"{path}: planes de forme {planes.shape}, attendu (N, {self.planes}, 8, 8)"
                )

        self._cache = {
            "planes": torch.from_numpy(np.ascontiguousarray(planes)),
            "pol_idx": torch.from_numpy(np.ascontiguousarray(pol_idx)).long(),
            "pol_prob": torch.from_numpy(np.ascontiguousarray(pol_prob)).float(),
            "pol_len": torch.from_numpy(np.ascontiguousarray(pol_len)).long(),
            "wdl": torch.from_numpy(np.ascontiguousarray(wdl)).float(),
        }
        self._cache_dict[shard_index] = self._cache
        self._cached_shard = shard_index

    def _locate(self, index: int) -> Tuple[int, int]:
        if index < 0:
            index += self._total
        if index < 0 or index >= self._total:
            raise IndexError(index)
        shard = bisect.bisect_right(self._cumulative, index)
        row = index - (self._cumulative[shard - 1] if shard > 0 else 0)
        return shard, row

    # ------------------------------------------------------------------ accès
    def __getitem__(self, index: int):
        """Renvoie ``(x, mask, idx, prob, length, wdl)`` pour une position.

        * ``x``      : float32 ``(8, 8, planes)`` — NHWC ;
        * ``mask``   : float32 ``(1858,)`` — 1 pour un coup légal ;
        * ``idx``    : int64 ``(64,)`` — indices bruts du shard ;
        * ``prob``   : float32 ``(64,)`` — probabilités brutes ;
        * ``length`` : int64 scalaire ;
        * ``wdl``    : float32 ``(3,)``.
        """
        if self._order is not None:
            index = int(self._order[index])
        shard, row = self._locate(int(index))
        self._load_shard(shard)
        assert self._cache is not None

        planes = self._cache["planes"][row]  # uint8 (planes, 8, 8)
        # (planes, 8, 8) -> (8, 8, planes), exactement le format attendu par le réseau.
        x = planes.permute(1, 2, 0).to(torch.float32) / 1.0

        idx = self._cache["pol_idx"][row].clone()
        prob = self._cache["pol_prob"][row].clone()
        length = int(self._cache["pol_len"][row].item())
        length = max(0, min(length, self.max_moves, idx.numel()))

        legal_idx = idx[:length]
        mask = torch.zeros(self.policy_size, dtype=torch.float32)
        if length:
            valid = legal_idx[(legal_idx >= 0) & (legal_idx < self.policy_size)]
            mask[valid] = 1.0

        wdl = self._cache["wdl"][row].clone()
        if wdl.numel() != 3:
            raise ValueError(f"{self.shards[shard]}: wdl de forme {tuple(wdl.shape)}, attendu (3,)")

        return x, mask, idx, prob, torch.tensor(length, dtype=torch.long), wdl


def collate_shard_batch(batch: Iterable[Tuple[torch.Tensor, ...]]):
    """Assemble des positions et construit les cibles policy **denses**.

    Renvoie ``(x, mask, target, wdl)`` avec :

    * ``x``      : float32 ``(B, 8, 8, planes)`` ;
    * ``mask``   : float32 ``(B, 1858)`` ;
    * ``target`` : float32 ``(B, 1858)``, construit par ``scatter_add_`` puis
      renormalisé sur les coups légaux ;
    * ``wdl``    : float32 ``(B, 3)``.
    """
    batch = list(batch)
    if not batch:
        raise ValueError("batch vide")

    xs = torch.stack([item[0] for item in batch])
    masks = torch.stack([item[1] for item in batch])
    idx = torch.stack([item[2] for item in batch]).long()
    prob = torch.stack([item[3] for item in batch]).float()
    lengths = torch.stack([item[4] for item in batch]).long()
    wdl = torch.stack([item[5] for item in batch]).float()

    batch_size = xs.shape[0]
    moves = idx.shape[1]
    policy_size = masks.shape[1]
    target = torch.zeros(batch_size, policy_size, dtype=torch.float32)

    positions = torch.arange(moves, dtype=torch.long).unsqueeze(0)
    valid = positions < lengths.unsqueeze(1)
    safe_idx = idx.clamp(0, policy_size - 1)
    src = torch.where(valid, prob, torch.zeros_like(prob))
    target.scatter_add_(1, safe_idx, src)

    # Filet de sécurité : une position dont toutes les probabilités sont nulles
    # reçoit une distribution uniforme sur ses coups légaux.
    sums = target.sum(dim=1, keepdim=True)
    empty = sums.squeeze(1) <= 0
    if bool(empty.any()):
        fallback = masks[empty]
        denom = fallback.sum(dim=1, keepdim=True).clamp_min(1.0)
        target[empty] = fallback / denom
        sums = target.sum(dim=1, keepdim=True)

    target = target / sums.clamp_min(1e-12)
    return xs, masks, target, wdl


class ShardBatchSampler:
    """Lot d'indices groupés par shard : shards mélangés, lignes mélangées dedans.

    Un échantillonneur global oblige le dataset à garder **tous** les shards ouverts
    ou en cache : au-delà de quelques centaines de shards, la RAM explose (24 M de
    positions ≈ 50 Go de tenseurs) et la lecture disque devient aléatoire. Ici la
    mémoire reste bornée à un shard et le disque lit en séquence, pour un mélange
    équivalent en pratique.
    """

    def __init__(
        self,
        shard_lengths: Sequence[int],
        batch_size: int,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:
        self.shard_lengths = [int(n) for n in shard_lengths]
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.epoch = 0

        self.offsets: List[int] = []
        total = 0
        for n in self.shard_lengths:
            self.offsets.append(total)
            total += n

    def __len__(self) -> int:
        total = 0
        for n in self.shard_lengths:
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += -(-n // self.batch_size)
        return total

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        order = rng.permutation(len(self.shard_lengths)) if self.shuffle else range(
            len(self.shard_lengths)
        )
        for shard in order:
            base = self.offsets[int(shard)]
            n = self.shard_lengths[int(shard)]
            rows = rng.permutation(n) if self.shuffle else np.arange(n)
            for start in range(0, n, self.batch_size):
                chunk = rows[start : start + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                yield (base + chunk).tolist()
        self.epoch += 1


__all__ = [
    "MAX_MOVES",
    "POLICY_SIZE",
    "ShardBatchSampler",
    "ShardDataset",
    "collate_shard_batch",
    "find_shards",
]
