"""Réseaux résiduels PyTorch, transposition de l'architecture Keras de DebZero.

L'entrée est au format NHWC ``(batch, 8, 8, plans)``, comme le tenseur produit par le
moteur.  Les convolutions travaillent en NCHW : le ``forward`` fait donc le ``permute``
une seule fois, au début.

Architecture (identique au modèle Keras d'origine) :

* stem : ``Conv3x3(planes -> f) -> BatchNorm -> SiLU`` ;
* blocs résiduels : ``BN -> SiLU -> Conv3x3 -> BN -> SiLU -> Conv3x3 -> SE -> +`` ;
* SE : moyenne globale -> ``Conv1x1(f -> f//8)`` -> ReLU -> ``Conv1x1(f//8 -> f)``
  -> Sigmoid -> multiplication ;
* tête policy : ``Conv1x1(32) -> BN -> SiLU -> flatten -> Linear(f*2) -> SiLU
  -> Linear(1858)``, puis masquage des coups illégaux et softmax ;
* tête value : ``Conv1x1(1) -> BN -> SiLU -> flatten -> Linear(256) -> SiLU
  -> Linear(3)``, puis softmax.

Aucune dépendance à Keras ou TensorFlow : uniquement PyTorch.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

POLICY_SIZE = 1858
BOARD_SIZE = 8

#: Variantes demandées : (nombre de blocs, filtres, SE).
VARIANT_SPECS: Dict[str, Dict[str, object]] = {
    "nano": {"blocks": 2, "filters": 32, "se": False},
    "small": {"blocks": 4, "filters": 48, "se": True},
    "base": {"blocks": 6, "filters": 64, "se": True},
    "large": {"blocks": 10, "filters": 128, "se": True},
}


class SqueezeExcitation(nn.Module):
    """Bloc SE : attention par canal, équivalent des Dense Keras sur un GAP."""

    def __init__(self, channels: int, ratio: int = 8) -> None:
        super().__init__()
        hidden = max(channels // ratio, 1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = F.adaptive_avg_pool2d(x, 1)
        scale = F.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale


class ResidualBlock(nn.Module):
    """Bloc résiduel pré-activation avec SE optionnel."""

    def __init__(self, channels: int, use_se: bool = True, se_ratio: int = 8) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.se = SqueezeExcitation(channels, se_ratio) if use_se else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        if self.se is not None:
            out = self.se(out)
        return x + out


class DebZeroNet(nn.Module):
    """Réseau policy/value avec masque de légalité.

    ``forward(x, legal_mask)`` renvoie ``(policy_probs, wdl_probs)`` de formes
    respectives ``(B, 1858)`` et ``(B, 3)``.
    """

    def __init__(
        self,
        blocks: int = 6,
        filters: int = 64,
        planes: int = 17,
        use_se: bool = True,
        se_ratio: int = 8,
        policy_size: int = POLICY_SIZE,
        variant: str = "custom",
    ) -> None:
        super().__init__()
        self.variant = variant
        self.planes = planes
        self.blocks = blocks
        self.filters = filters
        self.use_se = use_se
        self.policy_size = policy_size

        self.stem = nn.Sequential(
            nn.Conv2d(planes, filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.SiLU(),
        )
        self.trunk = nn.Sequential(
            *[ResidualBlock(filters, use_se=use_se, se_ratio=se_ratio) for _ in range(blocks)]
        )

        # Tête policy.
        self.policy_conv = nn.Conv2d(filters, 32, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc1 = nn.Linear(32 * BOARD_SIZE * BOARD_SIZE, filters * 2)
        self.policy_fc2 = nn.Linear(filters * 2, policy_size)

        # Tête value.
        self.value_conv = nn.Conv2d(filters, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(BOARD_SIZE * BOARD_SIZE, 256)
        self.value_fc2 = nn.Linear(256, 3)

    @staticmethod
    def _flatten_nhwc(x: torch.Tensor) -> torch.Tensor:
        """Aplatit comme ``Flatten`` Keras sur une entrée NHWC (h, w, c)."""
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], -1)

    def forward(
        self,
        x: torch.Tensor,
        legal_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # NHWC -> NCHW, une seule fois.
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        x = self.trunk(x)

        p = F.silu(self.policy_bn(self.policy_conv(x)))
        p = self._flatten_nhwc(p)
        p = F.silu(self.policy_fc1(p))
        logits = self.policy_fc2(p).float()

        if legal_mask is not None:
            # Les logits illégaux reçoivent -1e9 AVANT le softmax : leur probabilité
            # est exactement zéro, pas seulement petite.
            logits = logits + (1.0 - legal_mask.float()) * -1e9
        policy = F.softmax(logits, dim=-1)

        v = F.silu(self.value_bn(self.value_conv(x)))
        v = self._flatten_nhwc(v)
        v = F.silu(self.value_fc1(v))
        wdl = F.softmax(self.value_fc2(v).float(), dim=-1)

        return policy, wdl


def build_net(variant: str = "base", planes: int = 17) -> DebZeroNet:
    """Construit un réseau du zoo.

    Parameters
    ----------
    variant:
        ``nano``, ``small``, ``base`` ou ``large``.
    planes:
        Nombre de plans d'entrée (17 par défaut).
    """
    key = str(variant).lower()
    if key not in VARIANT_SPECS:
        allowed = ", ".join(sorted(VARIANT_SPECS))
        raise ValueError(f"Variante inconnue {variant!r} ; choisir parmi : {allowed}")
    spec = VARIANT_SPECS[key]
    return DebZeroNet(
        blocks=int(spec["blocks"]),
        filters=int(spec["filters"]),
        planes=planes,
        use_se=bool(spec["se"]),
        variant=key,
    )


def count_parameters(model: nn.Module) -> int:
    """Nombre de paramètres entraînables."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = [
    "BOARD_SIZE",
    "POLICY_SIZE",
    "VARIANT_SPECS",
    "DebZeroNet",
    "ResidualBlock",
    "SqueezeExcitation",
    "build_net",
    "count_parameters",
]
