# Duel nano contre le réseau de production (temps égal)

Mesure du 26/11. Protocole : même moteur des deux côtés (le moteur de la branche), seul le
réseau change. **Budget de temps égal** — 1 000 ms par coup, pas un nombre de nœuds fixe,
parce que c'est le temps qui compte pour un bot en ligne. 100 ouvertures aléatoires jouées
deux fois, couleurs inversées, 200 parties, 10 parties en parallèle.

- Référence (A) : `N1_10x64` (10 blocs × 64 filtres, 0,98 M paramètres) — celui de la prod.
- Candidat (B) : `nano` distillé (2 blocs × 32 filtres, 0,31 M paramètres, 17 plans).

## Résultat

| | valeur |
|---|---|
| victoires / nulles / défaites (B) | 55 / 73 / 72 |
| score | **45,8 %** |
| intervalle de confiance 95 % | 40,2 % – 51,3 % |
| écart Elo | **−30** [−69 ; +9] |
| nœuds par coup (A) | 344 |
| nœuds par coup (B) | **629** (1,83× plus) |

Conclusion : **le candidat n'est pas plus fort**, et le point d'estimation est même en
retrait. L'intervalle encadre 50 %, donc « pas de différence significative » est la
formulation rigoureuse, mais l'hypothèse « nano ≥ prod » n'est pas soutenue.

## Ce que ça dit

Le gain de vitesse est réel (1,83× plus de nœuds dans le même temps, mesuré à 344 contre
629) mais il ne compense pas la perte de capacité : 0,31 M de paramètres contre 0,98 M.
Distiller 20 M de positions ne suffit pas à faire tenir la force d'un réseau trois fois plus
gros dans un réseau trois fois plus petit.

## Premier duel : invalide, et pourquoi

Une première passe avait donné 51,2 % [46,4 – 56,1]. Elle est **à jeter** : le chemin du
réseau de référence était relatif (`nets/N1_10x64/...`) et `get_absolute_path` ne le
résolvait pas. L'option `NetPath` échouait **silencieusement** (seule une ligne
`info string` l'indiquait) et le moteur gardait son réseau par défaut : les deux camps ont
joué avec le même réseau. Diagnostic : `bench/match_time.py` avec chemins absolus, plus un
script de vérification qui lit les `info string`.

Correctif à prévoir côté moteur : un `NetPath` invalide devrait être signalé bruyamment,
pas ignoré.

---

# Duel nano v3 (nano_ib48) contre la production — 26/11, second duel

Même protocole, mêmes 100 ouvertures, mêmes 1 000 ms par coup, 200 parties.

- Référence (A) : `N1_10x64`, 0,98 M paramètres, 3,16 ms/position sur le VPS.
- Candidat (B) : `nano_ib48` v3 (blocs à goulot inversé, tête bilinéaire), 111 790
  paramètres, 6,35 M opérations, 0,46 ms/position sur le VPS.

## Résultat

| | valeur |
|---|---|
| score | **77,5 %** |
| intervalle de confiance 95 % | 72,6 % – 82,4 % |
| écart Elo | **+215** [+169 ; +268] |
| nœuds par coup (A) | 211 |
| nœuds par coup (B) | 383 (1,81× plus) |

Conclusion : le candidat est **significativement plus fort**, et l'intervalle est loin
d'encadrer 50 %.

## Le point important

L'avantage de nœuds est **le même que lors du premier duel** (1,81× contre 1,83×). Ce qui a
changé, c'est la qualité du réseau : 43,75 % de top-1 sur les positions de validation
contre 36,0 % pour le nano précédent, soit **+7,75 points**. Le gain de +215 Elo vient donc
de l'architecture, pas de la vitesse. C'est la démonstration que le calage en opérations
plutôt qu'en paramètres était le bon choix : le candidat a **neuf fois moins de paramètres**
que la production et gagne largement.

## Réserve honnête

Deux parties sur 200 se sont terminées en 7 coups avec zéro nœud des deux côtés : les deux
moteurs ont été coupés juste après l'ouverture, et l'outil a attribué le point à l'un puis à
l'autre. Les deux vont dans des sens opposés, donc le score n'est pas biaisé, mais le banc
devrait rejouer ces parties au lieu de les compter.
