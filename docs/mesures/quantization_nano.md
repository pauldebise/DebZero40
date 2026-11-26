# Quantization du nano v3 (nano_ib48), mesurée sur le VPS

Réseau : `nano_ib48_last.onnx`, 111 790 paramètres, 6,35 M opérations.
Processeur du VPS : AVX2 seulement, **pas d'AVX-512, pas de VNNI**. Un cœur.
Calibrage : 500 positions réelles. Évaluation : 5 000 positions, minimum de 60 passes.

| variante | taille | latence | top-1 |
|---|---|---|---|
| fp32 d'origine | 492 Ko | 0,48 ms | 43,98 % |
| quantization dynamique | 227 Ko | 2,69 ms | 29,68 % |
| quantization statique QDQ | 294 Ko | 1,39 ms | 5,88 % |

## Verdict

La quantization est **perdante sur les trois plans** : plus lente, moins précise, et la
variante statique est même inutilisable (5,88 % de top-1, à peine mieux que le hasard).

Deux causes identifiées :

1. **Pas de VNNI.** Les noyaux int8 d'ONNX Runtime n'ont pas le chemin rapide sur ce
   processeur ; le coût des conversions entre tenseurs fp32 et int8 dépasse le gain.
2. **Convolutions profondes.** Les blocs à goulot inversé sont majoritairement des
   convolutions 3x3 par canal, qu'ONNX Runtime quantifie mal : il insère des nœuds de
   conversion autour de chaque bloc. La tête bilinéaire (produit scalaire départ/arrivée
   sur les logits) amplifie l'erreur de quantification.

Conclusion : **ne pas quantifier cette architecture sur ce matériel**. La piste reste
ouverte pour deux cas seulement — un processeur avec VNNI, ou une architecture à
convolutions classiques (le bloc `nano_res4`), qui se quantifie beaucoup mieux.
