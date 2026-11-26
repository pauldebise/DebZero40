# Optimisation du moteur — mesures

Objectif : réduire le coût du Python dans la recherche, sans toucher à la force.

## Méthode

Profilage `cProfile` sur des recherches d'une seconde, puis deux mesures :

- **banc à nœuds fixes** : 400 nœuds, cache vidé, trois positions, médiane de quatre passes.
  C'est la mesure la plus stable du coût d'un nœud ;
- **mesure en conditions réelles** : recherches d'une seconde entières, huit alternances
  ancien/nouveau pour moyenner la charge de la machine, médiane des nœuds par seconde.
  C'est celle qui compte, mais elle est bruitée : le bot de production tourne sur le même
  processeur et le coût du réseau varie du simple au double selon la charge.

## Ce qui a été trouvé

Le profil désignait un coupable inattendu : `work.set_fen(root_fen)` à chaque nœud, pour
revenir à la position racine. 0,27 ms par nœud passées à reconstruire un plateau depuis une
chaîne FEN, alors qu'un `push`/`pop` coûte 9 µs.

Derrière, dans l'ordre : le hash Zobrist polyglot (31 µs par appel, deux fois par nœud), une
génération complète de coups pour `is_checkmate()` alors que la liste était calculée juste
après, deux copies de dictionnaire de politique par nœud, et la construction d'une chaîne
UCI par coup légal pour retrouver son index.

## Ce qui a été changé

| changement | gain au banc |
|---|---|
| `push`/`pop` au lieu de `set_fen` pour revenir à la racine | ~0,20 ms/nœud |
| `_transposition_key()` au lieu du hash polyglot (cache et répétitions) | ~0,08 ms/nœud |
| une seule génération de coups par nœud, partagée entre détection terminale et expansion | ~0,05 ms/nœud |
| suppression des tests de mat/pat dans la descente (impossibles sur un nœud déjà développé) | ~0,03 ms/nœud |
| table `(départ, arrivée, promotion) → index LC0` au lieu d'un `uci()` par coup | ~0,04 ms/nœud |
| plus de copie du dictionnaire de politique sur les succès de cache | ~0,02 ms/nœud |
| `value()` inliné dans `select_child` | ~0,01 ms/nœud |

## Résultat

Banc à nœuds fixes (400 nœuds, cache froid) :

| position | avant | après | gain |
|---|---|---|---|
| milieu de partie (40 coups légaux) | 1,728 ms | 1,285 ms | −26 % |
| ouverture | 1,659 ms | 1,180 ms | −29 % |
| finale | 1,654 ms | 1,000 ms | −40 % |

Conditions réelles, recherches d'une seconde sur le VPS, huit alternances :

| | avant | après | gain |
|---|---|---|---|
| nœuds par seconde | 572 | 664 | **×1,16** |

L'écart entre les deux mesures s'explique : dans une vraie recherche, l'appel ONNX pèse
lourd (0,46 ms machine au repos, jusqu'à 0,83 ms quand le bot tourne) et il n'a pas bougé.
Le Python, lui, est passé d'environ 0,9 à 0,5 ms par nœud. Sur une machine rapide et au
repos, le gain serait plus proche de ×1,35 ; sur le VPS chargé, il est de ×1,16.

Autrement dit : **+16 % de nœuds à temps égal, soit une quinzaine d'Elo** — pas les +22 à
+69 espérés avant mesure. Le réseau est devenu le poste dominant, et c'est là qu'il faudra
chercher la suite (batching, ou réseau plus économe).

## Garde-fous

- 72 tests unitaires verts.
- Table d'index vérifiée sur **548 343 coups** issus de 300 parties aléatoires : 0 écart avec
  `lc0_index_of`, y compris 763 roques, 552 promotions et 272 481 coups noirs.
  Un bug a été trouvé et corrigé à cette occasion : la sous-promotion en cavalier n'a pas
  d'entrée propre dans l'espace LC0, elle utilise la case nue.
- Partie complète via l'interface UCI : le moteur joue, annonce un `bestmove` légal et un
  `ponder` cohérent.

## Duel de vérification

200 parties, même réseau, 1 000 ms par coup, 6 demi-coups d'ouverture, 10 parties en
parallèle, graine 7. Fichiers : `docs/mesures/duel_moteur.{json,log}`.

| | valeur |
|---|---|
| score | 59,5 % (87 V, 64 N, 49 D) |
| IC 95 % | 53,9 – 65,1 % |
| Elo | +67 [ +27 ; +108 ] |
| nœuds par coup | 340 (ancien) / 368 (nouveau), soit ×1,08 |

**Ce résultat ne mesure pas ce qu'on croit.** Trois faits le disent :

1. à nombre de nœuds fixé (800), les deux moteurs jouent **exactement le même coup** sur
   cinq positions testées — le changement est purement une accélération ;
2. le gain de nœuds mesuré dans le duel n'est que de 8 %, pas les 16 % attendus : sous
   20 processus en parallèle, le processeur est saturé et les deux moteurs se partagent
   les cycles ;
3. le score est **inversement** corrélé à l'avantage de nœuds : 66,7 % quand le nouveau
   moteur n'a pas d'avantage, 52,6 % quand il en a un.

Autrement dit, le +67 Elo vient de la divergence chaotique des parties à ~350 nœuds, pas de
la vitesse. À ce niveau de recherche, huit nœuds de plus changent un coup, puis toute la
partie. Le duel ne sait pas résoudre un effet de cette taille : il faudrait plusieurs
milliers de parties.

Ce qui est établi : le moteur est **plus rapide** (×1,16 à temps égal, mesuré proprement),
il joue **les mêmes coups** à nœuds égaux, et il **ne régresse pas**. L'apport attendu en
force reste une dizaine d'Elo, pas soixante-sept.
