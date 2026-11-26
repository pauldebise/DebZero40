# nano — entraînement complet (16 époques, 20 M positions)

Run du 25-26/11, réseau `nano` (2 blocs × 32 filtres, 312 519 paramètres), distillation
des parties de Leela Chess Zero (run T80). Aucun auto-jeu.

## Données

- 10 taras LC0 → **400 shards = 20 000 000 positions** (26 Go).
- 5 shards (250 000 positions) mis de côté pour la validation.
- 16 époques, soit 320 M d'exemples vus, learning rate cosine de 3e-3 à 0.
- 6 min 25 par époque sur RTX 4090, **1 h 42 de calcul**.

## Résultats

| Métrique | Époque 1 | Époque 8 | Époque 15 (retenue) | Époque 16 |
|---|---|---|---|---|
| perte entraînement | 3,92 | 3,59 | 3,53 | 3,52 |
| perte validation | 4,93 | 3,64 | **3,54** | 3,56 |
| top-1 entraînement | 26,3 % | 33,3 % | 35,1 % | 35,2 % |
| top-1 validation | 29,2 % | 32,9 % | **35,3 %** | 35,2 % |

**Évaluation externe** (12 000 positions d'un tar jamais utilisé) :

| Modèle | top-1 policy | exactitude WDL | inférence CPU |
|---|---|---|---|
| nano, 3 époques (run précédent) | 28,2 % | 77,2 % | 0,56 ms |
| **nano, 16 époques** | **36,0 %** | 78,3 % | 0,47 ms |
| ancien `N1_10x64` (production) | — | — | 3,16 ms |

**Moteur complet**, 800 nœuds : 314 nps avec l'ancien réseau, **912 à 1 131 nps** avec le
nouveau — environ trois fois plus de positions explorées à temps égal.

## Ce que les courbes disent

- La perte de politique descend **sans interruption** de 2,675 à 2,411, et les courbes
  d'entraînement et de validation se superposent : aucun surapprentissage après 16 époques
  sur 20 M de positions.
- La perte WDL de validation **oscille violemment** (0,75 → 2,71) alors que celle
  d'entraînement reste plate à 0,74. Sur des données externes, l'exactitude WDL est saine
  (78 %). Le problème vient donc des shards de validation, dont les étiquettes de résultat
  sont probablement mal orientées — cohérent avec le biais mesuré (P(gain blanc) = 0,24 à
  matériel égal au lieu de 0,50).
- **La courbe n'avait pas plafonné à l'époque 16** : le learning rate était à zéro et la
  perte descendait encore de 0,002 par époque. Le facteur limitant est le **volume de
  données**, pas le nombre d'époques.

## Prochaine étape recommandée

1. **Auditer l'orientation des étiquettes WDL** avant tout nouveau run (comparer une
   position et son miroir : les probabilités doivent s'inverser). Sinon la sélection du
   meilleur modèle reste pilotée par du bruit.
2. Sélectionner le meilleur modèle sur la **perte de politique**, stable, et non sur la
   perte totale.
3. Passer à **50 à 100 M de positions** : à 20 M la courbe ne plafonne pas encore, et le
   coût reste d'environ 0,9 $/heure.

## Coût

- Run utile : **1 h 42 de GPU à 0,89 $/h ≈ 1,55 $**, plus ~0,10 $ de pod de récupération.
- Total de la journée sur ce réseau : **3,3 $** (dont 1,7 $ perdus en tentatives ratées :
  coupe-circuits mal réglés, module `chess` absent, cache non borné).
