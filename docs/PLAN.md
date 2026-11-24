# Plan d'amélioration DebZero40 — branche `feat/agent-improvements`

Base de départ : tag `baseline-paul` (commit 9a397bf), code entièrement écrit par Paul.
Contraintes respectées : **tout reste en Python** (pas de C++/Rust, le MCTS reste en Python),
pas de gros run d'entraînement pour l'instant, budget Runpod plafonné à 3 $.

## Mesures de départ (relevées sur le VPS, 2 threads CPU)

| Mesure | Valeur baseline |
|---|---|
| Inférence réseau (N1_10x64, CPU) | ~3,2 ms / position |
| Vitesse de recherche | **~200-250 nps**, profondeur 5-6 |
| Temps consommé par coup (blitz 3+2) | ~0,9 s (alloué) mais plafonné par overhead |
| Policy du réseau sur coups **légaux**, milieu de partie | **64 %** → 36 % de la masse sur des coups impossibles |
| Probabilité du réseau pour O-O | **0,007 %** (le roque est quasi invisible) |
| Coups légaux absents du mapping 1858 | 3 trouvés sur 14 771 (sous-promotions `h2h1r/b/n`) |

## Défauts confirmés dans le code (à corriger)

1. **Encodage sans droits de roque / prise en passant** (`board_encoder.py`) :
   le réseau ne peut pas savoir que le roque est disponible. Corrigé à l'inférence par des
   plans supplémentaires « copiables », sans casser les réseaux existants (voir plus bas).
2. **Mapping 1858 incomplet** : les sous-promotions depuis la rangée 2 (h2h1N/R/B) manquent.
3. **Réseau non informé de la légalité** : la policy est normalisée sur les coups légaux après
   un softmax sur 1858 sorties ; en milieu de partie 36 % de la masse est gaspillée.
   Le MCTS ne peut pas non plus exploiter cette information comme signal de « position connue ».
4. **Température appliquée à l'inférence** (`temperature=1.5`, et 8.0 en finale) : aplatit la prior
   de recherche, donc du bruit dans le MCTS. La température appartient au choix du coup à
   l'entraînement (self-play), pas à la probabilité transmise à la recherche.
5. **Table de transposition à chevauchement de clés** : deux positions différentes peuvent avoir
   le même hash → nœuds fusionnés avec des visites fausses, jamais détecté.
6. **Élagage de sous-promotions incohérent** : le multithread filtre D/Q mais pas R/B, le
   singlethread ne filtre rien ; les sous-promotions inventées polluent la policy normalisée.
7. **Gestion du temps dangereuse** : `t_sec = max(t/1000 - 0.2, 0.1)`, `our_time/20 + inc/2`
   sans garde-fou ; en bullet/faible temps restant le bot joue 0,1 s sans pondération d'urgence.
8. **`Threads` par défaut à 16** alors que le conteneur a 2 vCPU : thread de batch + 16 workers
   qui se battent pour le GIL, donc paralysie et nps ridicule.
9. **Multithread** : descente sérialisée par un verrou global autour du réseau (sérialisation de
   fait), et la board est partagée : `board.is_game_over()` hors verrou sur une board valant
   encore un chemin d'un autre worker.

## Ordre des travaux

- **Étape 1 (fait)** : sauvegarde (tag + bundle) et diagnostic chiffré.
- **Étape 2** : correctifs sûrs et mesurables, sans changer de réseau :
  mapping complet, filtrage cohérent des promotions, vraies clés de transposition,
  garde-fous de temps, température retirée de la recherche, `Threads` par défaut = cœurs réels.
- **Étape 3** : moteur de recherche mono-processus propres (partagé par les deux chemins),
  avec éval batchée et gestion de la répétition.
- **Étape 4** : plans de roque/EP injectés à l'inférence (compatibles avec les réseaux actuels
  qui les ignorent : plans à zéro par défaut côté entraînement).
- **Étape 5** : harnais de non-régression (A/B entre versions, tests de légalité, tests de temps).
- **Étape 6** : plan d'entraînement du prochain réseau (document séparé `docs/RESEARCH_PLAN.md`),
  avec un test court Runpod ≤ 3 $ pour valider le pipeline, sans grosse campagne.

## Critères de réussite

- Aucune régression de légalité (0 coup illégal), mêmes coups ou meilleurs sur les positions test.
- nps nettement supérieur à 200 en monothread (objectif ×2-3 en pur Python).
- A/B sur un petit jeu de positions : la nouvelle version gagne plus de duels que la baseline.
- Comportement de temps sûr : jamais plus de 50 % du temps restant consommé, aucun drapeau.
- Code testé, mode `Threads=1` et multithread validés, prod non touchée.
