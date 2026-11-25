"""Second benchmark : réglage fin du chargement de données et du débit maximal."""
import argparse, time, multiprocessing as mp
import numpy as np
import torch
import torch.nn.functional as F
import pyarrow.parquet as pq

from bench_training import DebZeroNet, encode_fen


def _work(fen):
    return encode_fen(fen)


def bench_mp(fens, workers):
    t0 = time.time()
    if workers == 1:
        for f in fens:
            encode_fen(f)
    else:
        with mp.Pool(workers) as pool:
            for _ in pool.imap_unordered(_work, fens, chunksize=64):
                pass
    return len(fens) / (time.time() - t0)


def bench_train(blocks, filters, batch, amp, steps, device):
    model = DebZeroNet(blocks=blocks, filters=filters).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    x = torch.rand(batch, 8, 8, 17, device=device)
    m = (torch.rand(batch, 1858, device=device) > 0.98).float()
    m[:, 0] = 1
    tp = torch.rand(batch, 1858, device=device) * m
    tp = tp / tp.sum(1, keepdim=True).clamp(min=1e-9)
    tv = F.one_hot(torch.randint(0, 3, (batch,), device=device), 3).float()

    def step():
        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=amp):
            p, v = model(x, m)
            loss = -(tp * (p + 1e-9).log()).sum(1).mean() - (tv * (v + 1e-9).log()).sum(1).mean()
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()

    for _ in range(3):
        step()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    return batch * steps / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--rate", type=float, default=0.89)
    args = ap.parse_args()
    dev = "cuda"

    df = next(pq.ParquetFile(args.parquet).iter_batches(batch_size=20000,
                                                        columns=["fen"])).to_pandas()
    fens = df["fen"].tolist()

    print("--- conversion FEN -> plans selon le nombre de processus ---")
    best = 0
    for w in (1, 4, 8, 16, 32):
        r = bench_mp(fens, w)
        best = max(best, r)
        print(f"  {w:2d} processus : {r:8,.0f} positions/s")
    print(f"  meilleur : {best:,.0f} positions/s")

    print("\n--- débit maximal d'entraînement ---")
    rows = []
    for blocks, filters, batch, amp in [
        (6, 64, 1024, False), (6, 64, 2048, False), (6, 64, 1024, True),
        (6, 64, 2048, True), (8, 96, 1024, True), (10, 128, 1024, True),
    ]:
        try:
            r = bench_train(blocks, filters, batch, amp, 20, dev)
            rows.append((blocks, filters, batch, amp, r))
            print(f"  {blocks}x{filters:<4d} lot {batch:5d} {'fp16' if amp else 'fp32'} : {r:8,.0f} pos/s")
        except torch.cuda.OutOfMemoryError:
            print(f"  {blocks}x{filters:<4d} lot {batch:5d} : VRAM insuffisante")
            torch.cuda.empty_cache()

    blocks, filters, batch, amp, r = max(rows, key=lambda x: x[4])
    h100 = 100e6 / r / 3600
    h732 = 732e6 / r / 3600
    print(f"\nmeilleure config : {blocks}x{filters}, lot {batch}, "
          f"{'fp16' if amp else 'fp32'} -> {r:,.0f} pos/s")
    print(f"  100 M positions : {h100:.2f} h = {h100*args.rate:.2f} $")
    print(f"  732 M positions : {h732:.2f} h = {h732*args.rate:.2f} $")
    print(f"\nsi le chargement plafonne à {best:,.0f} pos/s, "
          f"le GPU est {'LE goulot' if best < r else 'bien alimenté'}")


if __name__ == "__main__":
    main()
