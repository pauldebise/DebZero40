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

### Correction : la colonne `wdl` n'est PAS le résultat de la partie

Mesuré sur 100 000 positions réelles :

| Contrôle | Résultat |
|---|---|
| positions où `wdl` est quasi binaire (max > 0,95) | 16,5 % seulement |
| accord `argmax(wdl)` avec le gagnant de la partie | 67,5 % (33 % au hasard) |
| corrélation matériel / P(les blancs gagnent) | 0,558 |
| avec ≥ 8 points de matériel d'avance, P(gain du camp en avance) | **0,974** |
| dispersion de `wdl` à matériel équilibré | de 0,0 à 0,992 |

Conclusion : `wdl` est une **vraie évaluation de position** (elle respecte fortement le
matériel tout en capturant plus que lui), et `game_winner` est séparément le résultat de la
partie. La cible de valeur est donc utilisable — c'est une bonne nouvelle par rapport à ce
que j'avais écrit le 24 novembre.

### Point à vérifier avant un gros run

À matériel équilibré, la moyenne de P(les blancs gagnent) vaut **0,239**, pas 0,5. Le dataset
est massivement filtré sur les positions à matériel équilibré (2 235 positions sur 3 000), donc
ce n'est pas un artefact d'échantillonnage : il y a un biais systématique. Hypothèse la plus
probable : le dataset contient des positions miroir de positions « noirs au trait », et le `wdl`
n'a peut-être pas été retourné en même temps que la FEN. Si c'est le cas, une partie des labels
de valeur est du bruit, voire pire.

**À faire avant tout entraînement sérieux** : comparer la distribution de `wdl` entre positions
d'origine et positions miroir (par exemple via les droits de roque `K` vs `k`), et corriger ou
filtrer en conséquence. Un script de contrôle est à écrire ; sans lui, on risque d'entraîner une
tête de valeur sur des étiquettes incohérentes.

### Le masque de légalité existe déjà dans les données

Les `-1` donnent gratuitement les coups légaux et le bon index 1858. À l'inférence, il suffit
de sortir des logits, de les masquer, puis de normaliser.

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

## 5. Coûts mesurés (RTX 4090 à 0,89 $/h)

Débits relevés le 25 novembre sur un pod réel, code PyTorch équivalent à l'architecture
existante, données réelles :

| Réseau | Débit d'entraînement | Passage complet sur 732 M positions |
|---|---|---|
| 6×64 (0,98 M params) | **82 600 pos/s** (lot 2048, fp16) | **3,70 $** |
| 8×96 | 33 800 pos/s | 5,36 $ |
| 10×128 (4,04 M params) | 30 000 pos/s | 6,02 $ |
| 12×128 (4,64 M params) | 13 100 pos/s | 13,81 $ |
| 15×256 (20 M params) | 5 550 pos/s | 32,58 $ |

Le **chargement des données** plafonne à 49 000 positions/s avec 8 cœurs (14 500 par cœur
pour la conversion FEN → plans). Au-delà de 8 processus, le multitraitement régresse.

Scénarios réalistes :

| Scénario | Coût mesuré |
|---|---|
| 20 M positions, conversion unique + 10 époques en 6×64 | **0,70 $** |
| 100 M positions, 6×64 | 0,51 $ |
| 100 M positions, 10×128 | 0,82 $ |
| Évaluation A/B de 100 parties (600 nœuds, 8 en parallèle) | 1,07 $ |
| Auto-jeu : 10 000 parties à 800 nœuds sur 16 vCPU | ~71 $ |

**Conclusion** : l'entraînement supervisé n'est pas le poste de coût. Générer des données
par auto-jeu coûte 20 à 100 fois plus cher que de consommer un dataset existant. Toute la
stratégie doit donc rester supervisée tant que le budget est de l'ordre de quelques dollars.

## 6. Refonte du pipeline proposée

Le pipeline TensorFlow d'origine cumule trois défauts : installation lourde, conflits de
versions (TF 2.17 refuse numpy 2 ; il faut `tf2onnx` et `tensorflow[and-cuda]` pour le GPU),
et alimentation inefficace (lot 256, pas de précision mixte).

Remplacer par PyTorch, mesuré sur le pod :

- **PyTorch est déjà installé** sur les images Runpod : aucun téléchargement de 3 Go, aucun
  conflit de versions. C'est ce qui a fait perdre 40 minutes hier.
- Même architecture, mêmes conventions d'entrée NHWC, export ONNX natif par
  `torch.onnx.export` (donc le moteur Python ne change pas d'une ligne).
- Lot 2048 + précision mixte fp16 : **3 fois plus rapide** que lot 256 fp32.
- Conversion des données **une seule fois** vers un format binaire compact, puis entraînement
  de plusieurs époques depuis ce cache, au lieu de reconvertir à chaque époque.

## 7. Protocole d'un premier vrai run (budget 3 $)

Ordre proposé, chaque étape validée avant la suivante :

1. **Contrôle des labels** (0 $, sur le VPS) : vérifier le biais de `wdl` entre positions
   d'origine et positions miroir. Si le biais est confirmé, filtrer. Aucun entraînement
   sérieux avant ce contrôle.
2. **Run de référence 6×64** sur 20 M positions, 10 époques : ~0,70 $. But : avoir une
   courbe de perte propre et un réseau exportable, pas encore la force.
3. **Run 10×128** sur les mêmes 20 M positions : ~0,80 $. Comparer les deux en validation
   (perte de policy et de valeur), pas en parties.
4. **Évaluation A/B** du meilleur des deux contre l'ancien réseau, 100 à 200 parties à
   budget de nœuds égal : ~1,07 $ à 2,15 $.
5. Total : **2,5 à 3 $**, soit moins que le solde actuel (4,22 $).

Garde-fous systématiques : coupe-circuit par `sleep` + suppression du pod avant de commencer,
pod supprimé et solde relu en fin de session, aucun volume réseau permanent (le stockage coûte
plus cher que le calcul pour un run d'une heure).
