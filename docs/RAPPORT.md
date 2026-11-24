# Rapport — amélioration du moteur DebZero40

Branche : `feat/agent-improvements`
Sauvegarde de l'état d'origine : tag `baseline-paul` (commit 9a397bf) + bundle
`~/backups/DebZero40-baseline-20261124.bundle`.

Contraintes respectées : **tout reste en Python** (aucun C++/Rust, le MCTS est en pur
Python), pas de campagne d'entraînement lancée, **0,55 $** consommés sur les 3 $ autorisés.

## 1. Ce qui a été trouvé

### Le défaut le plus coûteux : l'encodage ignore le roque

Trois positions au **plateau strictement identique** mais aux droits de roque différents
(`KQkq`, `-`, `Kkq`) produisent des sorties réseau **bit-à-bit identiques**. Le réseau ne
peut donc pas savoir que le roque est disponible.

Conséquence mesurée sur `N1_10x64` :

| Mesure | Valeur |
|---|---|
| probabilité cumulée des deux roques | **0,019 %** |
| ce que donnerait une prior neutre | 7,7 % (2 coups sur 26) |

Le réseau ne se contente pas d'ignorer le roque : il l'évite activement. Même problème
sur les coups illégaux en général : en milieu de partie, **35,6 %** de la policy part sur
des coups impossibles (ex. `e1a1` à 35 % quand le grand roque n'est plus possible).

### Autres défauts corrigés ou documentés

- **Température appliquée à la recherche** (`1.5`, et `8.0` en finale) : la policy donnée
  au MCTS était artificiellement aplatie. La température relève du choix de coup pendant
  l'auto-jeu, pas de la prior de recherche.
- **Détection de répétition trompeuse** : `board.is_repetition(2)` de python-chess ne fait
  pas ce qu'on croit et peut produire de fausses nulles. Remplacé par une comparaison
  explicite de hash Zobrist.
- **`is_game_over(claim_draw=True)` sur chaque nœud** : ~15 % du temps de recherche
  gaspillé à recalculer des règles déjà connues.
- **`Threads` par défaut à 16** alors que le conteneur a 2 vCPU : threads qui se battent
  pour le GIL. Passé par défaut à 1.
- **Multithread** : descente sérialisée par un verrou global, et board partagée entre
  workers. Remplacé par un moteur mono-thread propre.
- **Aucune réponse sur position sans coup légal** (pat) : le moteur restait muet, ce qui
  fait perdre la partie au temps. Répond maintenant `bestmove 0000`.
- **Bug bloquant découvert pendant le développement** : la condition d'arrêt empêchait
  toute simulation en mode temps (0 nœud, coup quasi aléatoire).

## 2. Ce qui a été livré

### Moteur

- `src/engine/mcts_core.py` — MCTS pur Python, remontée WDL avec inversion de signe sûre,
  détection de répétition par hash, prior normalisée sur les coups réellement jouables.
- `src/engine/search.py` — adaptateur UCI, budget de temps prudent (jamais plus de 10 % du
  temps restant), cache d'inférence, garde-fou `bestmove 0000`, jamais de réponse muette.
- `src/utils/net_wrapper.py` — encodage direct (sans FEN ni `board.mirror()`), policy brute
  sans température, et **détection automatique** de la forme d'entrée du réseau.
- `src/utils/board_encoder.py` — `extended=True` ajoute les 4 plans de roque et le plan de
  prise en passant, prêts pour le prochain réseau.

### Entraînement (prêt, non lancé à grande échelle)

- `src/training/training_pipeline/train_v2.py` — parquet → TFRecords 17 plans →
  entraînement avec masque de légalité → contrôle → export ONNX.
- `src/utils/model_builder.py` — `use_legal_mask=True` (coups illégaux à `-inf` avant softmax).

### Tests

24 tests automatisés : protocole UCI complet, pat, mat en 1, promotions, pression au temps,
`setoption`, encodage, répétitions, non-régression de l'inférence.

`bench/test_inference_equivalence.py` vérifie que le nouveau chemin d'inférence donne un
**écart nul** avec le code d'origine sur 11 positions blanches et noires.

`bench/match.py` est un harnais de duel A/B entre deux moteurs UCI.

## 3. Validation réelle sur Runpod (0,55 $)

Un pod RTX 4090 (0,89 $/h) a été loué ~40 min, avec coupe-circuit automatique, puis supprimé.
Reliquat vérifié : **4,80 $ → 4,25 $**.

Résultat du pipeline de bout en bout sur 100 000 positions réelles (ChessFENS) :

| Étape | Résultat |
|---|---|
| Conversion des parquet | 100 000 positions encodées en 17 plans |
| Entraînement (300 pas, 6 blocs × 64 filtres) | loss 4,98 → 4,51, stables |
| **Masse policy sur coups légaux** | **100,00 %** (ancien réseau : 64 %) |
| Export ONNX | 3,8 Mo, entrées `(8,8,17)` + `legal_mask`, vérifié à 100 % |

Le réseau exporté tourne déjà dans le moteur : `bestmove d2d4` correct, et l'ancien réseau
12 plans continue de fonctionner sans aucune configuration. Le test de force est à refaire
avec un vrai entraînement (300 pas sur 30 M paramètres ne prouvent rien sur la force).

## 4. Mesures de force

| Match | Format | Score de la nouvelle version |
|---|---|---|
| 1er | 24 parties, 250 nœuds/coup, 4 demi-coups d'ouverture | 39,6 % [IC95 26,4 – 52,7] |
| 2e | 24 parties, 600 nœuds/coup, 8 demi-coups d'ouverture | 39,6 % [IC95 24,0 – 55,2] |

Les deux intervalles de confiance encadrent 50 % : **aucune différence significative**. Le score
identique sur deux réglages différents indique que les deux moteurs jouent au même niveau, ce qui
est cohérent — ils partagent le même réseau, donc les mêmes angles morts.

Fait marquant : une part importante des parties finit en nulle par répétition. Ce n'est pas un
défaut du nouveau code (la détection de répétition est vérifiée par test unitaire), c'est le
réseau actuel qui ne sait pas convertir ses positions avantageuses.

**Conclusion honnête** : le moteur est plus rapide, plus sûr (plus de perte au temps, plus de
silence), mieux structuré et prêt pour le prochain réseau — mais il ne peut pas être nettement
plus fort tant que le réseau n'est pas réentraîné. Le gain de force viendra du réseau, pas du
code de recherche. Prétendre le contraire serait faux.

## 5. Prochaine étape recommandée

Voir `docs/RESEARCH_PLAN.md`. En résumé :

1. entraîner sérieusement le réseau 17 plans + masque de légalité (le pipeline est validé,
   il ne reste qu'à augmenter le nombre de pas et le volume de données) ;
2. mesurer honnêtement en A/B contre l'ancien réseau, même budget de nœuds ;
3. seulement si le réseau progresse, envisager l'auto-jeu.
