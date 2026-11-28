# Journal des expériences

Toute expérience qui coûte de l'argent laisse une ligne ici, le jour même, avec son coût
réel et sa conclusion. L'objectif est simple : ne jamais repayer pour un essai déjà fait,
et pouvoir dire à tout moment combien le projet a coûté et ce que ça a rapporté.

Règles :

- une ligne par run payant (pod, GPU, API), jamais un lot ;
- le coût est le coût réel constaté sur la facture Runpod, pas une estimation, sauf mention ;
- la colonne « verdict » dit ce qu'on en retient, pas ce qu'on a fait ;
- les expériences gratuites (local, VPS) sont notées « 0 $ » : elles comptent aussi.

| date | expérience | coût | résultat | verdict |
|---|---|---|---|---|
| 24/11 | nano v2 : entraînement complet + premier duel au temps égal | ~2,5 $ (est.) | 45,8 % contre la production, Elo −30 [−69 ; +9] | perdant, mais la méthode (temps égal, ouvertures appariées) est validée |
| 26/11 | criblage de 6 architectures nano v3, 2 époques, 2 graines | 0,60 $ | res4 33,1 %, ib48 32,9 %, hybride 32,0 %, nano 31,8 %, dense 30,1 %, dense160 29,0 % | les têtes denses sont éliminées ; l'écart entre graines (1,8 pt) dépasse l'écart entre architectures |
| 26/11 | entraînement nano_ib48 complet, 16 époques, 20 M positions | 1,90 $ | top-1 validation 43,75 %, WDL 82,86 % | +7,75 points de top-1 contre le nano v2 |
| 26/11 | duel nano v3 contre production, 200 parties, 1 000 ms/coup | 0,60 $ | 77,5 %, +215 Elo [+169 ; +268] | adopté ; l'avantage de nœuds est inchangé (1,83×), donc le gain vient de l'architecture |
| 26/11 | quantification fp32 / dynamique / statique du nano v3 | 0 $ | 0,48 / 2,69 / 1,39 ms et 43,98 / 29,68 / 5,88 % de top-1 | perdant sur ce processeur (pas de VNNI, convolutions profondes) ; ne pas refaire |
| 26/11 | profilage et optimisation du moteur (Python) | 0 $ | banc : 1,73 -> 1,29 ms/nœud ; conditions réelles : 572 -> 664 nœuds/s (x1,16) | +16 % de nœuds pour zéro euro ; le réseau est désormais le poste dominant |
| 26/11 | duel de vérification moteur ancien/nouveau, 200 parties, même réseau (dont un premier pod raté, image sans SSH) | 1,70 $ | 59,5 % [+67 Elo, IC +27/+108], mais 8 % de nœuds en plus seulement et score inversement corrélé à l'avantage de nœuds | gain réel ~+10 Elo ; le duel à 350 nœuds/coup est trop bruité pour trancher, ne pas refaire à cette taille |
| 27/11 | model soup best/last (α = 0,5) et audit de calibration WDL, 250 k positions de validation (pod CPU de récupération des checkpoints) | ~0,01 $ | top-1 43,42 %, WDL 0,740, exactitude WDL 84,88 %, Brier 0,0835 | domine strictement `last` sur les six métriques ; à valider en duel avant déploiement |
| 27-28/11 | run clean `nano_ib48` : cible MLH, filtre `--min-visits 100`, sélection combinée, 16 époques | 1,94 $ (calcul) + 7,03 $ de pod resté inactif après la fin | top-1 validation 42,98 %, WDL 85,71 %, MAE MLH 51 plies ; époque 16 retenue (0,6441) | pas de gain mesurable sur la validation contre le run du 26/11 ; évaluation externe encore nécessaire. Le pod aurait dû être supprimé à 02 h 00, pas à 12 h 11 |

## Ce que ça a coûté jusqu'ici

Environ **16,3 $** de calcul GPU, pour un réseau qui bat la production de 215 Elo — dont
**7,03 $ perdus** par un pod laissé allumé dix heures après la fin du run clean. Ce poste
inutile représente presque la moitié de la dépense : le chien de garde qui supprime un pod
inactif n'est pas une option.
S'y ajoute le volume réseau `debzero-shards` (4 Go), facturé 0,004 $/h en continu. Le poste
le plus rentable n'est pas le GPU : c'est le temps passé à mesurer (duel au temps égal,
criblage à deux graines) qui a évité d'entraîner trois architectures perdantes.

## À ne pas refaire

- **Quantifier cette architecture sur ce processeur** : mesuré, perdant sur les trois plans.
- **Cribler sur une seule graine** : l'écart entre graines (1,8 point de top-1) est du même
  ordre que l'écart entre architectures. Deux graines minimum, sinon on choisit au hasard.
- **Comparer en nœuds égaux** : c'est le temps qui compte pour un bot en ligne.
