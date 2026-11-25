"""
Benchmark de débit d'entraînement + vérification de la nature des labels.

But : répondre à « combien coûte un réentraînement » avec des chiffres mesurés,
pas des estimations. Mesure :
  1. la vitesse de conversion FEN -> plans (mono-coeur et multi-processus) ;
  2. la nature réelle de la colonne `wdl` (évaluation moteur ou résultat de partie ?) ;
  3. le débit d'entraînement (positions/s) pour plusieurs tailles de réseau,
     en fp32 et en précision mixte fp16 ;
  4. l'extrapolation en heures et en dollars.

Usage :
    python bench_training.py --parquet /workspace/parquet/dataset-00000.parquet \
        --rate 0.89 --rows 200000
"""
import argparse
import glob
import os
import time
import multiprocessing as mp

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- modèle
# Reproduction fidèle de l'architecture Keras de Paul, mais en PyTorch :
#   stem conv3x3 -> N blocs résiduels (BN/Swish/conv3x3/BN/Swish/conv3x3/SE/add)
#   tête policy : conv1x1 32 -> BN -> Swish -> flatten -> dense(f*2) -> dense(1858)
#   tête value  : conv1x1 1  -> BN -> Swish -> flatten -> dense(256)  -> dense(3)
# L'entrée reste au format du moteur : (batch, 8, 8, plans) en NHWC.
class SE(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        # Équivalent des Dense sur un GlobalAveragePooling2D de Keras.
        self.fc1 = nn.Conv2d(channels, channels // ratio, 1, bias=False)
        self.fc2 = nn.Conv2d(channels // ratio, channels, 1, bias=False)

    def forward(self, x):
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class ResBlock(nn.Module):
    def __init__(self, channels, ratio=8):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.se = SE(channels, ratio)

    def forward(self, x):
        out = self.conv1(F.silu(self.bn1(x)))
        out = self.conv2(F.silu(self.bn2(out)))
        return x + self.se(out)


class DebZeroNet(nn.Module):
    def __init__(self, blocks=6, filters=64, planes=17, ratio=8):
        super().__init__()
        self.planes = planes
        self.stem = nn.Sequential(
            nn.Conv2d(planes, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.SiLU(),
        )
        self.blocks = nn.Sequential(*[ResBlock(filters, ratio) for _ in range(blocks)])

        self.p_conv = nn.Conv2d(filters, 32, 1, bias=False)
        self.p_bn = nn.BatchNorm2d(32)
        self.p_fc1 = nn.Linear(32 * 64, filters * 2)
        self.p_fc2 = nn.Linear(filters * 2, 1858)

        self.v_conv = nn.Conv2d(filters, 1, 1, bias=False)
        self.v_bn = nn.BatchNorm2d(1)
        self.v_fc1 = nn.Linear(64, 256)
        self.v_fc2 = nn.Linear(256, 3)

    def forward(self, x, legal_mask=None):
        # x : (B, 8, 8, plans) NHWC -> NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.stem(x)
        x = self.blocks(x)

        p = F.silu(self.p_bn(self.p_conv(x)))
        # Flatten Keras (NHWC) = permute puis flatten pour retrouver le même ordre
        p = p.permute(0, 2, 3, 1).reshape(p.shape[0], -1)
        p = F.silu(self.p_fc1(p))
        logits = self.p_fc2(p)

        v = F.silu(self.v_bn(self.v_conv(x)))
        v = v.permute(0, 2, 3, 1).reshape(v.shape[0], -1)
        v = F.silu(self.v_fc1(v))
        value = F.softmax(self.v_fc2(v), dim=-1)

        if legal_mask is not None:
            logits = logits + (1.0 - legal_mask) * -1e9
        return F.softmax(logits, dim=-1), value


# --------------------------------------------------------------------------- données
def make_lookup():
    """Table UCI -> index 1858, construite comme le moteur (ordre identique)."""
    import chess
    moves = []
    for from_sq in range(64):
        for to_sq in range(64):
            if from_sq == to_sq:
                continue
            rd = abs(chess.square_rank(from_sq) - chess.square_rank(to_sq))
            fd = abs(chess.square_file(from_sq) - chess.square_file(to_sq))
            if (rd, fd) in ((1, 2), (2, 1)) or rd == 0 or fd == 0 or rd == fd:
                moves.append(chess.Move(from_sq, to_sq).uci())
                if to_sq is not None and chess.square_rank(to_sq) == 7 and chess.square_rank(from_sq) == 6 \
                        and fd <= 1 and (rd == 0 or fd == 0 or rd == fd):
                    moves.append(chess.Move(from_sq, to_sq, chess.KNIGHT).uci())
                    moves.append(chess.Move(from_sq, to_sq, chess.BISHOP).uci()
                                 if fd <= 1 else chess.Move(from_sq, to_sq, chess.KNIGHT).uci())
                    moves.append(chess.Move(from_sq, to_sq, chess.ROOK).uci())
    return moves


def encode_fen(fen, planes=17):
    """FEN -> plans (8,8,planes) uint8, mêmes conventions que le moteur."""
    import chess
    board = chess.Board(fen)
    out = np.zeros((8, 8, planes), dtype=np.uint8)
    black = board.turn == chess.BLACK
    for square, piece in board.piece_map().items():
        rank = chess.square_rank(square)
        if black:
            rank = 7 - rank
        ch = piece.piece_type - 1 + (0 if piece.color == board.turn else 6)
        out[rank, chess.square_file(square), ch] = 1
    if planes >= 17:
        me, them = board.turn, not board.turn
        if board.has_queenside_castling_rights(me):
            out[:, :, 12] = 1
        if board.has_kingside_castling_rights(me):
            out[:, :, 13] = 1
        if board.has_queenside_castling_rights(them):
            out[:, :, 14] = 1
        if board.has_kingside_castling_rights(them):
            out[:, :, 15] = 1
        if board.ep_square is not None:
            r = chess.square_rank(board.ep_square)
            out[7 - r if black else r, chess.square_file(board.ep_square), 16] = 1
    return out


def _work(fen):
    return encode_fen(fen)


def bench_conversion(fens, workers):
    t0 = time.time()
    for f in fens:
        encode_fen(f)
    single = len(fens) / (time.time() - t0)

    t0 = time.time()
    with mp.Pool(workers) as pool:
        list(pool.imap_unordered(_work, fens, chunksize=256))
    multi = len(fens) / (time.time() - t0)
    return single, multi


# --------------------------------------------------------------------------- benchmark
def bench_model(blocks, filters, batch, amp, steps, device):
    model = DebZeroNet(blocks=blocks, filters=filters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    x = torch.rand(batch, 8, 8, 17, device=device)
    m = torch.zeros(batch, 1858, device=device)
    for i in range(batch):
        m[i, torch.randint(0, 1858, (30,), device=device)] = 1
    tp = torch.rand(batch, 1858, device=device) * m
    tp = tp / tp.sum(1, keepdim=True).clamp(min=1e-9)
    tv = F.one_hot(torch.randint(0, 3, (batch,), device=device), 3).float()

    def one_step():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            p, v = model(x, m)
            loss = -(tp * (p + 1e-9).log()).sum(1).mean() + \
                   -(tv * (v + 1e-9).log()).sum(1).mean()
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

    for _ in range(3):
        one_step()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    dt = time.time() - t0
    return batch * steps / dt, dt / steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--rate", type=float, default=0.89, help="prix horaire du pod en dollars")
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()

    device = "cuda"
    print(f"GPU : {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} Go")
    print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
    print(f"coeurs CPU : {mp.cpu_count()}")

    # ---- 1. lecture du parquet et nature des labels
    pf = pq.ParquetFile(args.parquet)
    print(f"\nparquet : {pf.metadata.num_rows} lignes, {pf.metadata.num_row_groups} groupes")

    t0 = time.time()
    tbl = next(pf.iter_batches(batch_size=args.rows, columns=["fen", "wdl", "policy", "game_winner"]))
    df = tbl.to_pandas()
    print(f"lecture de {len(df)} lignes en {time.time()-t0:.1f}s")

    wdl = np.stack([np.asarray(w, dtype=np.float32) for w in df["wdl"]])
    gw = np.stack([np.asarray(g, dtype=np.float32) for g in df["game_winner"]])
    # Si wdl = résultat de partie, alors wdl est quasi binaire et suit game_winner.
    frac_binary = float(np.mean((wdl.max(1) > 0.95)))
    agree = float(np.mean(wdl.argmax(1) == gw.argmax(1)))
    print("\n--- nature de la colonne wdl ---")
    print(f"positions où wdl est quasi binaire (max > 0.95) : {frac_binary*100:.1f} %")
    print(f"accord argmax(wdl) == gagnant de la partie      : {agree*100:.1f} %")
    print(f"exemples wdl : {[list(np.round(x,3)) for x in wdl[:3]]}")
    print(f"exemples game_winner : {[list(x.astype(int)) for x in gw[:3]]}")

    # ---- 2. vitesse de conversion
    fens = df["fen"].tolist()[:5000]
    single, multi = bench_conversion(fens, workers=max(1, mp.cpu_count() - 2))
    print("\n--- conversion FEN -> plans ---")
    print(f"mono-coeur   : {single:,.0f} positions/s")
    print(f"multi-coeurs : {multi:,.0f} positions/s ({mp.cpu_count()-2} processus)")

    # ---- 3. débit d'entraînement
    print("\n--- débit d'entraînement (positions/s) ---")
    results = []
    configs = [(6, 64, 512, False), (6, 64, 512, True), (6, 64, 1024, True),
               (10, 128, 512, True), (12, 128, 512, True), (15, 256, 256, True)]
    for blocks, filters, batch, amp in configs:
        try:
            pps, sps = bench_model(blocks, filters, batch, amp, args.steps, device)
            nparams = sum(p.numel() for p in DebZeroNet(blocks=blocks, filters=filters).parameters())
            results.append((blocks, filters, batch, amp, pps, nparams))
            print(f"{blocks:2d}x{filters:<4d} lot {batch:4d} {'fp16' if amp else 'fp32'} : "
                  f"{pps:8,.0f} pos/s ({sps*1000:6.1f} ms/pas, {nparams/1e6:.2f} M params)")
        except torch.cuda.OutOfMemoryError:
            print(f"{blocks:2d}x{filters:<4d} lot {batch:4d} : VRAM insuffisante, ignoré")
            torch.cuda.empty_cache()

    # ---- 4. extrapolation
    print(f"\n--- extrapolation au tarif de {args.rate} $/h ---")
    print(f"{'config':12s} {'pos/s':>10s} {'10 M pos':>10s} {'100 M pos':>10s} {'732 M pos':>12s}")
    for blocks, filters, batch, amp, pps, nparams in results:
        def cost(n):
            h = n / pps / 3600
            return h * args.rate
        print(f"{blocks}x{filters:<9d} {pps:10,.0f} "
              f"{cost(10e6):9.2f} $ {cost(100e6):9.2f} $ {cost(732e6):11.2f} $")


if __name__ == "__main__":
    main()
