"""Gamme v3 : architectures définies par leur budget d'opérations, pas par leurs paramètres.

Principes, chacun choisi pour une raison mesurable :

* **bloc à goulot inversé** (convolution profonde k×k, élargissement 1×1, projection 1×1) :
  à nombre d'opérations égal, il porte plus de canaux que la convolution 3×3 classique ;
* **tête de politique bilinéaire** : le logit d'un coup est le produit scalaire entre les
  caractéristiques de sa case de départ et de sa case d'arrivée. Le réseau apprend des
  relations entre cases, pas des index de coups, et la tête ne coûte presque rien ;
* **tête de valeur WDL + coups restants (MLH)** : la seconde sortie force le tronc à
  encoder la phase de partie, pour un coût négligeable ;
* **entrée NHWC 17 plans**, même contrat que la gamme précédente : le moteur n'a pas à
  changer.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.nets import POLICY_SIZE, ResidualBlock
from src.utils.lc0_moves import LC0_MOVES

BOARD = 8


def _square(name: str) -> int:
    """``"e4"`` -> indice de case 0-63 dans l'ordre des plans (rang 8 en premier)."""
    file_idx = ord(name[0]) - ord("a")
    rank_idx = int(name[1]) - 1
    return rank_idx * BOARD + file_idx


def move_indices() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Construit les tables (départ, arrivée, type de promotion) des 1858 coups."""
    froms, tos, promos = [], [], []
    for uci in LC0_MOVES:
        froms.append(_square(uci[0:2]))
        tos.append(_square(uci[2:4]))
        suffix = uci[4:5]
        promos.append({"": 0, "q": 1, "r": 2, "b": 3}[suffix])
    return (
        torch.tensor(froms, dtype=torch.long),
        torch.tensor(tos, dtype=torch.long),
        torch.tensor(promos, dtype=torch.long),
    )


class SqueezeExcite(nn.Module):
    def __init__(self, channels: int, ratio: int = 8):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = x.mean(dim=(2, 3), keepdim=True)
        s = torch.sigmoid(self.fc2(F.silu(self.fc1(s))))
        return x * s


class InvertedBottleneck(nn.Module):
    """Convolution profonde k×k, élargissement, projection, excitation, résidu."""

    def __init__(self, channels: int, kernel: int = 5, expand: int = 3,
                 se: bool = True, se_ratio: int = 8):
        super().__init__()
        hidden = channels * expand
        self.dw = nn.Conv2d(channels, channels, kernel, padding=kernel // 2,
                            groups=channels, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.pw1 = nn.Conv2d(channels, hidden, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden)
        self.pw2 = nn.Conv2d(hidden, channels, 1, bias=False)
        self.se = SqueezeExcite(channels, se_ratio) if se else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.silu(self.bn1(self.dw(x)))
        y = F.silu(self.bn2(self.pw1(y)))
        y = self.se(self.pw2(y))
        return x + y


class GlobalContext(nn.Module):
    """Contexte global à coût quasi nul (GCNet) : agrège toute la position par canal."""

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.conv1 = nn.Conv2d(channels, hidden, 1)
        self.conv2 = nn.Conv2d(hidden, channels, 1)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3), keepdim=True)
        ctx = self.conv2(F.silu(self.conv1(pooled)))
        return x + ctx


class BilinearPolicyHead(nn.Module):
    """Logit(coup) = <q(case de départ), k(case d'arrivée)> + biais de promotion."""

    def __init__(self, channels: int, dim: int = 32, policy_size: int = POLICY_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(channels, dim, 1, bias=False)
        self.bn = nn.BatchNorm2d(dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.promo_bias = nn.Parameter(torch.zeros(4))
        froms, tos, promos = move_indices()
        self.register_buffer("froms", froms, persistent=False)
        self.register_buffer("tos", tos, persistent=False)
        self.register_buffer("promos", promos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn(self.proj(x)))              # (B, d, 8, 8)
        h = h.permute(0, 2, 3, 1).reshape(x.shape[0], BOARD * BOARD, -1)
        q = self.q(h)                                   # (B, 64, d)
        k = self.k(h)                                   # (B, 64, d)
        scores = q @ k.transpose(1, 2)                  # (B, 64, 64) : toutes les paires
        logits = scores[:, self.froms, self.tos]        # (B, 1858)
        return logits + self.promo_bias[self.promos]


class DensePolicyHead(nn.Module):
    """Tête historique : aplatissement puis deux couches denses (témoin de comparaison)."""

    def __init__(self, channels: int, bottleneck: int = 128, policy_size: int = POLICY_SIZE):
        super().__init__()
        self.proj = nn.Conv2d(channels, 32, 1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(32 * BOARD * BOARD, bottleneck)
        self.fc2 = nn.Linear(bottleneck, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.silu(self.bn(self.proj(x)))
        h = h.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        return self.fc2(F.silu(self.fc1(h)))


class HybridPolicyHead(nn.Module):
    """Bilinéaire + un biais global par coup, appris sur la position entière."""

    def __init__(self, channels: int, dim: int = 32, policy_size: int = POLICY_SIZE):
        super().__init__()
        self.bilinear = BilinearPolicyHead(channels, dim, policy_size)
        self.global_bias = nn.Linear(channels, policy_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=(2, 3))
        return self.bilinear(x) + self.global_bias(pooled)


class ValueHead(nn.Module):
    """Trois sorties WDL et une sortie « nombre de coups restants »."""

    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.conv = nn.Conv2d(channels, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(1)
        self.fc1 = nn.Linear(BOARD * BOARD, hidden)
        self.fc2 = nn.Linear(hidden, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v = F.silu(self.bn(self.conv(x)))
        v = v.permute(0, 2, 3, 1).reshape(x.shape[0], -1)
        return self.fc2(F.silu(self.fc1(v)))


@dataclass
class V3Config:
    """Description d'un modèle de la gamme.

    ``blocks`` et ``filters`` fixent le coût du tronc ; ``head`` le type de tête de
    politique ; ``block`` le type de bloc résiduel (``ib`` moderne ou ``res`` classique,
    ce dernier servant de témoin) ; ``context`` ajoute (ou non) un bloc de contexte global.
    """

    name: str = "nano"
    blocks: int = 6
    filters: int = 40
    expand: int = 3
    kernel: int = 5
    se: bool = True
    head: str = "bilinear"
    head_dim: int = 32
    block: str = "ib"
    context: int = 0
    planes: int = 17
    policy_size: int = POLICY_SIZE
    extras: dict = field(default_factory=dict)


#: Budgets visés, en millions d'opérations par position (convention : 1 MAC = 1 op).
BUDGETS = {"nano": 5.0, "small": 12.0, "medium": 25.0, "large": 70.0}


class DebZeroV3(nn.Module):
    def __init__(self, cfg: V3Config):
        super().__init__()
        self.cfg = cfg
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.planes, cfg.filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(cfg.filters),
            nn.SiLU(),
        )
        trunk = []
        for _ in range(cfg.blocks):
            if cfg.block == "ib":
                trunk.append(InvertedBottleneck(cfg.filters, cfg.kernel, cfg.expand, cfg.se))
            elif cfg.block == "res":
                # Témoin : le bloc classique de la gamme précédente, à filtres égaux.
                trunk.append(ResidualBlock(cfg.filters, use_se=cfg.se))
            else:
                raise ValueError(f"bloc inconnu : {cfg.block!r}")
        for _ in range(cfg.context):
            trunk.append(GlobalContext(cfg.filters))
        self.trunk = nn.Sequential(*trunk)

        if cfg.head == "bilinear":
            self.policy = BilinearPolicyHead(cfg.filters, cfg.head_dim, cfg.policy_size)
        elif cfg.head == "dense":
            self.policy = DensePolicyHead(cfg.filters, cfg.head_dim, cfg.policy_size)
        elif cfg.head == "hybrid":
            self.policy = HybridPolicyHead(cfg.filters, cfg.head_dim, cfg.policy_size)
        else:
            raise ValueError(f"tête inconnue : {cfg.head!r}")

        self.value = ValueHead(cfg.filters)

    def forward(self, x: torch.Tensor, legal_mask: torch.Tensor | None = None):
        x = x.permute(0, 3, 1, 2).contiguous()   # NHWC -> NCHW
        x = self.trunk(self.stem(x))

        logits = self.policy(x).float()
        if legal_mask is not None:
            logits = logits + (1.0 - legal_mask.float()) * -1e9
        policy = F.softmax(logits, dim=-1)

        v = self.value(x)
        wdl = F.softmax(v[:, :3], dim=-1)
        mlh = F.softplus(v[:, 3:4])              # coups restants, positif
        return policy, wdl, mlh


#: Modèles de la gamme, du plus léger au plus lourd.
PRESETS = {
    "nano": V3Config(name="nano", blocks=6, filters=40, expand=3, kernel=5,
                     head="bilinear", head_dim=32, context=0),
    # Variantes d'ablation pour le nano, à budget d'opérations comparable.
    "nano_dense": V3Config(name="nano_dense", blocks=6, filters=40, expand=3, kernel=5,
                           head="dense", head_dim=96, context=0),
    "nano_hybrid": V3Config(name="nano_hybrid", blocks=6, filters=40, expand=3, kernel=5,
                            head="hybrid", head_dim=32, context=0),
    "nano_res": V3Config(name="nano_res", blocks=6, filters=40, expand=1, kernel=3,
                         head="bilinear", head_dim=32, block="res", context=0),
    "nano_dw3": V3Config(name="nano_dw3", blocks=8, filters=40, expand=3, kernel=3,
                         head="bilinear", head_dim=32, context=0),
    "nano_res4": V3Config(name="nano_res4", blocks=4, filters=40, expand=1, kernel=3,
                          head="bilinear", head_dim=32, block="res", context=0),
    "nano_res4_wide": V3Config(name="nano_res4_wide", blocks=4, filters=48, expand=1, kernel=3,
                               head="bilinear", head_dim=40, block="res", context=0),
    "nano_ib48": V3Config(name="nano_ib48", blocks=6, filters=48, expand=3, kernel=5,
                          head="bilinear", head_dim=32, context=0),
    "nano_dense160": V3Config(name="nano_dense160", blocks=6, filters=40, expand=3, kernel=5,
                              head="dense", head_dim=160, context=0),
    "small": V3Config(name="small", blocks=8, filters=56, expand=3, kernel=5,
                      head="bilinear", head_dim=40, context=1),
    "medium": V3Config(name="medium", blocks=9, filters=72, expand=3, kernel=5,
                       head="bilinear", head_dim=48, context=1),
    "large": V3Config(name="large", blocks=12, filters=112, expand=4, kernel=5,
                      head="bilinear", head_dim=64, context=2),
}


def build_v3(name: str = "nano", **overrides) -> DebZeroV3:
    if name not in PRESETS:
        raise KeyError(f"modèle inconnu : {name!r} (connus : {sorted(PRESETS)})")
    cfg = V3Config(**{**PRESETS[name].__dict__, **overrides})
    return DebZeroV3(cfg)
