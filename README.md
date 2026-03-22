<p align="center">
  <h1 align="center">DebZero</h1>
</p>

<p align="center">
  <strong>Un moteur d'échecs open source basé sur un algorithme MCTS dirigé par un réseau de neurones convolutif.</strong><br>
  Inspiré par l'architecture d'AlphaZero et de LeelaChessZero.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue" alt="Language">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-ee4c2c" alt="Framework">
  <img src="https://img.shields.io/badge/License-GPL%203.0-green" alt="License">
</p>

---

## À propos du projet

**DebZero** est un moteur d'échecs expérimental développé dans le but d'explorer l'apprentissage par renforcement profond (Deep Reinforcement Learning).
Il est actuellemnt déployé sur [Lichess](https://lichess.org/@/debzero) où il tourne sur un VPS Hetzner (2 AMD CPU cores) à environ 500 noeuds par secondes.

### Fonctionnalités clés
* **Recherche Arborescente Monte-Carlo (MCTS) :** Exploration de l'arbre des coups possibles basée sur des probabilités.
* **Réseaux de Neurones Convolutifs Résiduels (ResNet) :** Utilisé pour l'évaluation des positions et la politique (probabilité des coups).
* **Support du protocole UCI :** Compatible avec les interfaces graphiques d'échecs standards (Arena, Cute Chess...).

## Architecture et Fonctionnement

1. **Réseau de neurones :** Le modèle prend l'état actuel du plateau (sous forme d'un tenseur de taille `(8, 8, 12)`) en entrée et sort deux valeurs :
   - `Policy (p)` : Un vecteur de probabilités pour tous les coups légaux.
   - `Value (v)` : Un vecteur de probabilités WDL (Win/Draw/Loss) qui tente de prédire l'issue de la partie.
2. **MCTS :** La recherche est guidée par les prédictions du réseau de neurones, ce qui permet à l'algorithme d'ignorer les mauvaises branches très tôt dans le processus.

## Installation

### Prérequis
* Python 3.11
* Éventuellement une interface compatible UCI

### Étapes
```bash
# Cloner le dépôt
git clone [https://github.com/pauldebise/DebZero40.git](https://github.com/pauldebise/DebZero40.git)
cd DebZero40

# Créer un environnement virtuel (en pyton 3.11)
python3.11 -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
