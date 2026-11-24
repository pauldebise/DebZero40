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

## 5. Le champ `Threads` de l'UCI

Défaut à 16 alors que le conteneur a 2 vCPU. Le MCTS actuel crée un thread de batch et
`num_threads` workers qui se disputent le GIL : au-delà de 2-3 threads, la vitesse baisse.
Aligné sur le nombre de cœurs réellement disponibles.
