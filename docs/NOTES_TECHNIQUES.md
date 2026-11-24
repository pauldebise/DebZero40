# Notes techniques — pièges et incréments (DebZero40)

## 1. Miroir des noirs : ne PAS appliquer de rotation supplémentaire

`NetWrapper.inference` de Paul faisait :

```python
board_to_encode = board if board.turn == chess.WHITE else board.mirror()
encoded = fen_to_planes_int8(board_to_encode.fen())
```

`chess.Board.mirror()` de python-chess inverse **couleurs ET rangées/colonnes** (rotation
de 180° + échange des camps). Mais `fen_to_planes_int8` inverse **déjà** la rangée quand
c'est aux noirs de jouer et range les canaux par « mes pièces (0-5) / ses pièces (6-11) ».

Résultat vérifié : `fen_to_planes_int8(board.mirror().fen()) == board_to_planes_int8(board)`
exactement, pour n'importe quelle position. Les deux font un retournement, ils s'annulent.

**Règle** : encoder directement la board telle quelle. Toute rotation supplémentaire
(`[::-1]`, `[::-1, ::-1]`, échange de canaux) casse la corrélation avec le réseau.

Garde-fou : `bench/test_inference_equivalence.py` compare l'ancien et le nouveau chemin
sur 11 positions blanches et noires ; l'écart doit rester nul (policy ET wdl).

## 2. `board.is_repetition(2)` ne détecte pas une vraie répétition

Il compare seulement la position courante à son historique direct et se déclenche sur des
positions qui ne se sont jamais répétées dans une partie réelle. Utiliser
`repetition_count()` (`src/engine/mcts_core.py`), qui compare les hash Zobrist en remontant
l'historique de 2 en 2 demi-coups sans modifier la board passée.

## 3. Le VPS est partagé avec le bot de production

Le conteneur `lichess-bot` tourne en permanence (~200 nps). Toute mesure de performance
faite sur le VPS est bruitée. Pour un chiffre fiable, mesurer sur un pod Runpod dédié.

## 4. Température sur la policy

`NetWrapper.temperature = 1.5` (et 8.0 en finale) aplatissait la policy donnée au MCTS.
La température doit s'appliquer au **choix du coup** pendant l'auto-jeu, pas à la
probabilité transmise à la recherche. La recherche utilise maintenant la policy brute.

## 6. L'encodage ne contient pas les droits de roque : défaut majeur prouvé

Preuve : trois positions au **plateau identique** mais aux droits de roque différents
(`KQkq`, `-`, `Kkq`) produisent des sorties réseau **bit-à-bit identiques**.
`board_to_planes_int8` n'encode que les 12 plans de pièces.

Conséquence mesurée sur `N1_10x64`, position `r3k2r/8/8/8/8/8/8/R3K2R w KQkq` :

| Mesure | Valeur |
|---|---|
| probabilité cumulée des deux roques | **0,0185 %** |
| prior « uniforme » si le réseau ne savait rien | 7,7 % (2 coups sur 26) |

Le réseau est donc bien plus bas que le hasard sur le roque : il a appris à **éviter** les
coups dont il ne peut pas vérifier la légalité (même situation pour `e1a1` joué à 35 % dans
une position où le grand roque est impossible : 35 % de la masse policy part sur un coup
illégal).

C'est le défaut le plus coûteux du moteur. Deux remèdes, de coût très différent :

- **Sans réentraîner (immédiat)** : le problème n'est pas que le MCTS reçoive des coups
  illégaux (il ne les joue jamais), c'est que le softmax du réseau a déjà consommé 35 % de
  sa masse sur des coups impossibles, et que le roque est plombé. On corrige donc :
  normalisation de la prior sur les coups légaux **avant** d'envoyer au MCTS (déjà fait),
  et surtout réentraînement avec le bon encodage pour réapprendre le roque.
- **Avec réentraînement (la vraie solution)** : ajouter les 5 plans roque/EP permet au
  réseau de savoir que le roque est disponible et de lui redonner une prior correcte.

## 7. Conclusion pour le prochain réseau

Le vrai correctif est l'encodage 17 canaux **au moment de l'entraînement**, sur les
données `ChessFENS` qui contiennent déjà les `-1` de légalité. À l'inférence, les plans
supplémentaires à zéro reproduisent exactement le comportement des anciens réseaux, donc
la compatibilité est assurée.
