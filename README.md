<p align="center">
  <h1 align="center">DebZero40</h1>
</p>

<p align="center">
  <strong>An open-source chess engine based on an MCTS algorithm guided by a convolutional neural network.</strong><br>
  Inspired by the architecture of AlphaZero and LeelaChessZero.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Language-Python-blue" alt="Language">
  <img src="https://img.shields.io/badge/Framework-TensorFlow-ee4c2c" alt="Framework">
  <img src="https://img.shields.io/badge/License-GPL%203.0-green" alt="License">
</p>

---

## About the project

**DebZero** is an experimental chess engine developed with the goal of exploring Deep Reinforcement Learning.
It is currently deployed on [Lichess](https://lichess.org/@/debzero) where it runs on a Hetzner VPS (2 AMD CPU cores) at around 500 nodes per second.

### Key features
* **Monte Carlo Tree Search (MCTS):** Exploration of the possible moves tree based on probabilities.
* **Residual Convolutional Neural Networks (ResNet):** Used for position evaluation and policy (move probabilities).
* **UCI Protocol Support:** Compatible with standard chess GUIs (Arena, Cute Chess...).

## Architecture and How It Works

1. **Neural Network:** The model takes the current board state (as a tensor of size `(8, 8, 12)`) as input and outputs two values:
   - `Policy (p)`: A probability vector for all legal moves.
   - `Value (v)`: A WDL (Win/Draw/Loss) probability vector that attempts to predict the outcome of the game.
2. **MCTS:** The search is guided by the neural network's predictions, allowing the algorithm to ignore bad branches very early in the process.

## Installation

### Prerequisites
* Python 3.11
* Optionally, a UCI-compatible GUI

### Steps
```bash
# Clone the repository
git clone [https://github.com/pauldebise/DebZero40.git](https://github.com/pauldebise/DebZero40.git)
cd DebZero40

# Create a virtual environment (with Python 3.11)
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
