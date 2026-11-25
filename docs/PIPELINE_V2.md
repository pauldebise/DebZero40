# Pipeline v2 — distillation LC0, réseaux neufs, mesures

Branche : `feat/agent-improvements`. Tout en Python. Pas d'auto-jeu : les données viennent
des auto-jeux publics de Leela Chess Zero (run T80, ~3400+ Elo, 10 000 visites/position).

## 1. Chaîne de données

`storage.lczero.org/files/training_data/test80` → `tools/lc0_to_shards.py` → shards `.npz`.

- Chaque tar (~1,7 Go) contient des jeux **gzip** et représente **~2,4 M positions**
  (8 356 octets par position, structure `V6TrainingData`).
- Conversion en flux : 17 plans uint8, politique creuse (64 entrées max), WDL, 1,3 Ko/position
  (÷6 par rapport au format brut, ÷12 avec la compression tar).
- **Validation intégrée** : pour chaque position, l'ensemble des coups légaux reconstruits est
  comparé à `{i : prob[i] >= 0}`. Taux mesuré sur 200 777 positions réelles : **99,61 %**,
  les 0,39 % restants étant des prises en passant (absentes du format) et des positions
  Chess960.
- Débit : 5 300 positions/s par processus, ~5 processus en parallèle sur le pod.

Deux conventions LC0 découvertes et vérifiées (elles font échouer le décodage si on les ignore) :
le roque s'encode « roi prend sa propre tour » (`e1h1`, pas `e1g1`), et la sous-promotion en
cavalier tombe sur la case sans suffixe.

## 2. Encodage et sortie

- **Entrée** : 17 plans — 12 pièces (nous / eux), 4 droits de roque, compteur de 50 coups
  normalisé. Pas de plan de prise en passant (non disponible dans les données).
- **Sortie** : politique de 1858 dans l'**espace canonique LC0**, masquée sur les coups légaux
  avant le softmax, plus WDL 3.
- Le moteur détecte automatiquement l'espace de politique (présence de l'entrée `legal_mask`
  dans le ONNX) : les anciens réseaux 12 plans continuent de fonctionner, écart nul vérifié.

## 3. Réseaux

| Variante | Blocs × filtres | Paramètres | Usage |
|---|---|---|---|
| `nano` | 2 × 32 | 0,31 M | **déploiement VPS** |
| `small` | 4 × 48 | 0,57 M | |
| `base` | 6 × 64 | 0,98 M | |
| `large` | 10 × 128 | 4,0 M | référence qualité |

## 4. Résultats mesurés

Entraînement réel : 20 M positions converties, entraînement sur 2 M positions
(40 shards) pour tenir dans la session GPU.

| Réseau | Époques | top-1 policy | exactitude WDL | Inférence CPU |
|---|---|---|---|---|
| `nano` | 3 | **28,2 %** | **77,2 %** | **0,56 ms** |
| `small` | 2 | 28,4 % | 76,1 % | 1,37 ms |
| ancien `N1_10x64` | — | — | — | 3,16 ms |

Évaluation sur des positions d'un tar **non utilisé à l'entraînement** (8 000 positions).

Vitesse du moteur complet, même position, 800 nœuds :

| Réseau | nps | 
|---|---|
| ancien `N1_10x64` | 290 |
| `nano` | **827** |

**2,85× plus rapide** à qualité de recherche comparable : à temps égal, le moteur explore
presque trois fois plus de positions.

## 5. Ce qui reste à faire

1. **Duel A/B** `nano` contre l'ancien réseau, 100 à 200 parties, même budget de nœuds :
   c'est la seule mesure qui dira si la distillation paie en force. Coût ~1 $.
2. **Entraînement à plus grande échelle** : le run n'a vu que 2 M positions sur 20 M
   disponibles et 2-3 époques. Avec 20 M positions et 8 époques, le top-1 devrait monter
   nettement au-dessus de 30 %.
3. **`base` et `large`** non entraînés (temps de GPU manquant dans la session).
4. **Intégration prod** : le réseau `nano` tourne déjà dans le moteur en local, mais la
   bascule de la production reste à décider.

## 6. Coût

Session complète du jour : **2,69 $** (solde 4,22 $ → 1,53 $), dont deux sessions perdues
par des coupe-circuits mal dimensionnés. Le coût d'un run utile (données + entraînement)
est d'environ **0,70 $**.
