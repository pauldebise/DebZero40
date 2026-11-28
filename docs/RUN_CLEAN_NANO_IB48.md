# Run clean `nano_ib48` — MLH et sélection combinée

## Objectif

Reprendre l'entraînement du réseau `nano_ib48` (111 790 paramètres) avec les trois
corrections identifiées après le run `nano_v3` :

1. **Cible MLH** : écrire `plies_left` dans les shards et entraîner la 4e sortie de la
   tête de valeur (Huber, poids 0,02, cible normalisée par 100).
2. **Filtre de visites** : ne conserver que les positions dont la recherche LC0 a au
   moins 100 visites (`--min-visits 100`).
3. **Sélection et checkpoints** : sélectionner sur un score combiné
   `top-1 + 0,25 × exactitude WDL` (au lieu de la perte totale), et sauvegarder
   `best_policy.pt`, `best_wdl.pt`, `best_combined.pt`, `last.pt` ainsi qu'un
   `epoch_XX.pt` par époque pour un model soup / SWA.

## Changements de code

Commit `70f9e5a` :

- `tools/lc0_to_shards.py` : champ `plies_left` (float16) dans chaque shard ;
- `src/train/dataset.py` : lecture optionnelle de `plies_left`, masque `has_mlh`
  (les anciens shards restent lisibles) ;
- `src/train/train.py` : perte MLH auxiliaire, métriques `mlh_loss` / `mlh_mae`,
  sélection `combined`, checkpoints par époque et checkpoints annexes ;
- `tests/test_mlh.py` : couverture de la cible, de la perte et de l'écriture des shards.

## Données

- 10 tars LC0 `test80` du 2 avril 2024 (`training-run1-test80-20240402-*.tar`),
  soit 20 M positions ;
- 400 shards de 50 000 positions, nommés `s01_00.npz` … `s10_39.npz` ;
- filtre `--min-visits 100` ;
- 5 derniers shards en validation (250 000 positions).

## Recette

```bash
python -m src.train.train \
  --shards /workspace/shards/clean \
  --arch v3 --variant nano_ib48 \
  --epochs 16 --batch 4096 --lr 3e-3 --amp \
  --workers 8 --shard-local \
  --wdl-weight 1.5 --mlh-weight 0.02 --mlh-scale 100 \
  --val-shards 5 \
  --select-metric combined --combined-wdl-weight 0.25 \
  --save-every-epoch --patience 4 --min-delta 5e-5 \
  --out /workspace/runs/clean
```

## Résultats

Run mené dans la nuit du 27 au 28/11, terminé à 02 h 00 (heure de Paris) : 16/16 époques,
2 h 47 de calcul sur une L4 à 0,69 $/h. Aucun arrêt anticipé : la patience de 4 époques n'a
jamais été atteinte.

| époque | perte val | perte politique | perte WDL | top-1 | exactitude WDL | MAE MLH | durée |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,5460 | 2,3789 | 1,4420 | 35,25 % | 66,10 % | 53,5 | 631 s |
| 2 | 3,9985 | 2,3120 | 1,1216 | 37,63 % | 70,78 % | 53,3 | 633 s |
| 3 | 3,8064 | 2,2724 | 1,0201 | 38,51 % | 73,98 % | 52,9 | 620 s |
| 4 | 3,7081 | 2,2660 | 0,9588 | 39,12 % | 64,42 % | 53,1 | 622 s |
| 5 | 3,9865 | 2,2751 | 1,1381 | 38,77 % | 52,42 % | 53,5 | 676 s |
| 6 | 4,6261 | 2,2756 | 1,5640 | 38,93 % | 44,97 % | 56,2 | 617 s |
| 7 | 4,0181 | 2,2398 | 1,1830 | 40,02 % | 72,05 % | 51,9 | 614 s |
| 8 | 4,6695 | 2,2482 | 1,6112 | 40,01 % | 55,94 % | 54,8 | 612 s |
| 9 | 4,7405 | 2,2483 | 1,6585 | 39,75 % | 50,51 % | 55,1 | 613 s |
| 10 | 4,1446 | 2,2201 | 1,2804 | 40,83 % | 72,44 % | 51,9 | 620 s |
| 11 | 3,4029 | 2,2095 | 0,7930 | 41,48 % | 78,41 % | 52,0 | 635 s |
| 12 | 3,2381 | 2,1897 | 0,6964 | 42,04 % | 86,24 % | 51,1 | 598 s |
| 13 | 3,6973 | 2,1946 | 0,9992 | 41,14 % | 76,62 % | 51,6 | 616 s |
| 14 | 3,4667 | 2,1881 | 0,8498 | 42,46 % | 76,99 % | 51,8 | 624 s |
| 15 | 3,2734 | 2,1794 | 0,7268 | 42,07 % | 84,13 % | 51,2 | 672 s |
| **16** | **3,2466** | **2,1743** | **0,7124** | **42,98 %** | **85,71 %** | **51,1** | **619 s** |

Lecture des courbes (`docs/mesures/courbes_clean_nano_ib48.png`) :

- **Perte de politique** : descente régulière, de 2,379 à 2,174, sans cassure. C'est la seule
  courbe propre du run et le signal que l'apprentissage fonctionne.
- **Perte WDL** : oscille de 0,70 à 1,66 d'une époque à l'autre, très au-delà du hasard (1,099).
  L'oscillation est celle déjà observée sur `nano_ib48` ; elle interdit de choisir le réseau sur
  la perte totale.
- **Top-1** : 35,25 % → 42,98 %, progression monotone en tendance malgré le bruit de validation.
- **MAE MLH** : 51 à 56 plies, sans amélioration nette après l'époque 1. La cible auxiliaire est
  apprise (la perte ne diverge pas) mais reste grossière ; le wrapper d'inférence l'ignore, donc
  aucun effet attendu sur la force à ce stade.
- **Learning rate** : cosine complet jusqu'à zéro, comme prévu.

Sélection combinée (`top-1 + 0,25 × exactitude WDL`) : l'époque 16 gagne avec 0,6441, devant
l'époque 12 (0,6360) et l'époque 11 (0,6108). `best.pt`, `best_combined.pt` et `last.pt` sont donc
les mêmes poids ; `best_policy.pt` et `best_wdl.pt` conservent les meilleures époques par tête.

Comparaison avec le run `nano_ib48` du 26/11 (sans MLH, sans filtre de visites) : à l'époque 16,
43,34 % de top-1 et 83,34 % de WDL ; au meilleur checkpoint sur la perte totale (époque 15),
42,87 % et 85,81 %. Le run clean donne 42,98 % et 85,71 %. **Écart non significatif** : le filtre
`--min-visits 100` et la cible MLH n'ont pas déplacé la validation.

## Évaluation externe

**En attente.** `best.pt` (= `last.pt`, époque 16) doit être évalué sur les 250 000 positions
externes (`s9_5` … `s9_9`) avec `bench/eval_net.py`, puis comparé au 43,75 % du run précédent.
Coût d'un pod L4 pour cette évaluation : ~0,10 à 0,15 $, à lancer après accord (solde Runpod
2,61 $ au 28/11 à 12 h).

## Décision

**En attente de l'évaluation externe.** Si le top-1 externe égale ou dépasse 43,75 %, le réseau
est équivalent au réseau déployé et le protocole MLH/filtre n'apporte rien : ne pas déployer, ne
pas refaire ce run. S'il est nettement supérieur (> 44,5 %), enchaîner sur un duel au temps égal
contre la production avant toute bascule.
