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
