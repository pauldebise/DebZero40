# Plan d'entraînement du prochain réseau DebZero

État : propositions à valider. Aucun gros entraînement lancé. Budget Runpod plafonné à 3 $.

## 1. Ce que disent les mesures (état actuel)

- Réseau `N1_10x64` (6 blocs, 64 filtres) : ~3,2 ms par position sur un vCPU du VPS.
  C'est ça qui borne le moteur à ~200-260 nps, donc une profondeur 5-6.
- **36 % de la policy du réseau part sur des coups illégaux** en milieu de partie.
  Le réseau n'a pas appris la légalité, donc une grosse part de sa capacité est gaspillée
  et la prior donnée au MCTS est bruitée.
- Le roque est quasi inexistant dans la policy (~0,007 %) : l'encodage `(8,8,12)` ne
  contient ni les droits de roque, ni la case de prise en passant, ni le compteur de
  coups. Le réseau ne peut pas savoir que le roque est disponible autrement qu'en
  devinant.
- Le réseau juge mal des positions matériellement perdues (dame en prise non vue).

## 2. Trois leviers, par rapport qualité/prix décroissant

### Levier A — Enrichir l'encodage (le plus rentable)

Passer de `(8,8,12)` à `(8,8,17)` en ajoutant 5 plans de **constantes** :

| Plan | Contenu |
|---|---|
| 12 | `1` si le grand roque blanc est encore possible |
| 13 | `1` si le petit roque blanc est encore possible |
| 14 | `1` si le grand roque noir est encore possible |
| 15 | `1` si le petit roque noir est encore possible |
| 16 | case de prise en passant (un 1 sur la case cible, 0 sinon) |

Coût : négligeable en calcul, ~5 canaux de plus en entrée. C'est ce que fait Leela
(elle utilise 112 plans) et c'est la correction du défaut structurel le plus visible.

**Compatibilité** : les réseaux actuels restent utilisables. Le `NetWrapper` détecte la
forme d'entrée (12 ou 17 canaux) et les plans supplémentaires sont à zéro pour les
anciens réseaux, ce qui reproduit exactement leur comportement.

### Levier B — Apprendre la légalité des coups

Deux options, cumulables :

1. **Mask de légalité** appliqué juste avant le softmax (le réseau sort des logits, le
   masque met `-inf` sur les coups illégaux). Simple, supprime les 36 % de masse perdue,
   et la cross-entropy devient bien plus informative. Nécessite un export ONNX avec le
   masque intégré, ou un masque calculé côté moteur (cosmétique mais simple).
2. **Sortie par plans 64 × 73** (comme Leela) plutôt qu'un vecteur de 1858 :
   même information, mais la structure spatiale est préservée et le réseau généralise mieux
   (un coup `e2e4` et `e7e5` partagent la même représentation). C'est la vraie amélioration
   de fond, mais elle change le format d'export et le mapping.

Recommandation : commencer par (1), qui est peu risqué et directement mesurable
(masse sur coups légaux doit passer de 64 % à 100 %), puis évaluer (2) plus tard.

### Levier C — Architecture et taille du réseau

- Le réseau actuel a 6 blocs × 64 filtres, avec blocs résiduels + SE. Honnête pour sa
  taille. Passer à 10×128 doublerait la force potentielle mais quadruplerait le temps
  d'inférence (le VPS est le facteur limitant).
- Piste plus intéressante à coût constant : **moins de blocs mais des convolutions
  dilatées ou une première couche 5×5**, qui donnent du champ de vision sans ajouter
  d'inférence. À tester, pas certain.

## 3. Données d'entraînement — état réel (vérifié)

Source : `Maxlegrec/ChessFENS` sur Hugging Face (732 M positions, 133 Go, 1 M lignes
par fichier parquet). Le fichier `data/parquet_files/where_to_find_parquet_files` le dit.

Colonnes : `fen`, `wdl`, `policy` (1858, **-1 sur les coups illégaux**),
`game_winner`. Chaque position est vue **du point de vue des blancs**.

### Ce que ça implique

**Le masque de légalité existe déjà dans les données** : les `-1` donnent gratuitement les
coups légaux et le bon index 1858. Le Levier B (apprendre la légalité) peut donc être
implémenté sans coût d'annotation. À l'inférence, il suffit de sortir des logits, de les
masquer, puis de normaliser ; le MCTS retrouve une prior de qualité.

**Mais les WDL viennent du résultat de la partie**, pas d'une recherche. C'est la limite
de fond : le réseau apprend « qui a gagné la partie », pas « quelle est la valeur de
cette position ». Le `game_winner` d'une position où un joueur a déjà gagné vaut toujours
la même chose, quelle que soit la suite — d'où les évaluations incohérentes observées en
fin de partie (dame en prise jugée +0,30 pour le camp qui la perd).

**Les positions ne sont pas toutes issues du départ réel** : la FEN est donnée avec
`w - - 23 1` ou `w K - 2 1`, c'est-à-dire sans l'historique. Conséquence directe sur
notre moteur : la répétition triple est **non détectable** depuis le dataset, donc la
policy n'a jamais appris à éviter une nulle par répétition. C'est un argument pour que le
moteur compense côté recherche (ce qui est fait).

### Ordre recommandé

1. **Réentraîner tel quel** avec le nouvel encodage (17 canaux) sur un sous-ensemble :
   on vérifie que le pipeline tourne et que le réseau exploite le roque. Objectif modeste,
   coût faible.
2. **Corriger la cible WDL** : remplacer le résultat de partie par une évaluation à
   profondeur (recherche courte, ou value network) sur les positions calmes, et garder
   le résultat de partie pour les positions terminales. C'est le vrai gain de force.
3. **Auto-jeu** ensuite, quand le moteur est à ~300 nps : les cibles policy deviennent des
   distributions de visites MCTS, ce qui est la seule façon d'aller au-delà de la force
   des humains qui ont produit le dataset.

## 4. Ce qu'il faut mesurer à chaque étape

| Étape | Métrique | Seuil de réussite |
|---|---|---|
| Encodage 17 canaux | masse policy sur coups légaux | > 99 % |
| Mask de légalité | idem | 100 % |
| Pipeline d'entraînement | loss policy en baisse, top-1 accuracy en hausse | monotone |
| Nouveau réseau | A/B contre l'ancien, même budget de nœuds | score > 55 % sur 100 parties |
| Moteur complet | nps sur le VPS | > 300 |

## 5. Test Runpod à budget maîtrisé (≤ 3 $)

But : vérifier que le pipeline d'entraînement tourne avec le nouvel encodage, **pas** de
produire un réseau fort.

- GPU : RTX 4090 (~0,7 $/h) ou A40. Pas de H100.
- Durée cible : **2 h maximum** (~1,5 $), plus le stockage réseau.
- Run : quelques milliers de pas seulement, sur un sous-ensemble des données.
- Livrables : courbes de loss, masse légale, et un `.keras` exportable en ONNX.
- Garde-fou : arrêt automatique à 2 h (timeout script) et vérification du coût en fin de
  session via l'API Runpod.

À ne lancer qu'après validation du point 3 (nature des données). Le budget sera annoncé
avant lancement, avec le coût estimé écrit noir sur blanc.
