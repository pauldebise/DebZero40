"""
Pipeline d'entraînement DebZero — version améliorée (17 plans + masque de légalité).

Différences avec `dataset_generator.py` d'origine :
  * les plans 12..16 (droits de roque, case EP) sont produits dans les TFRecords ;
  * le masque de légalité (1858 booléens) est produit à partir des `-1` du dataset ;
  * les coups illégaux sont EXCLUS de la cible policy (renormalisation sur le légal),
    au lieu d'être remis à zéro après coup ;
  * un rapport de contrôle est imprimé : masse légale de la policy cible.

Usage :
    python -m src.training.dataset_processing_pipeline.dataset_generator_v2 --limit 2
"""
import argparse
import glob
import os
import time

import numpy as np
import pyarrow.parquet as pq

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')

import tensorflow as tf  # noqa: E402

from src.utils.board_encoder import board_to_planes_int8, TOTAL_PLANES  # noqa: E402
from src.utils.mapping_out_1858 import FlatMapper  # noqa: E402

INPUT_PARQUET_DIR = "data/parquet_files/"
OUTPUT_TFRECORD_DIR = "data/tfrecords_v2/"


def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(planes, policy, wdl, legal_mask):
    feature = {
        'input': _bytes_feature(tf.io.serialize_tensor(planes)),
        'policy': _bytes_feature(tf.io.serialize_tensor(policy)),
        'wdl': _bytes_feature(tf.io.serialize_tensor(wdl)),
        'legal_mask': _bytes_feature(tf.io.serialize_tensor(legal_mask)),
    }
    return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()


def process_parquet_file(args):
    parquet_path, output_dir, file_index, fm_moves = args

    import chess

    filename = os.path.basename(parquet_path).replace('.parquet', '')
    output_path = os.path.join(output_dir, f"{filename}.tfrecord.gz")
    options = tf.io.TFRecordOptions(compression_type='GZIP')

    move_to_index = {m: i for i, m in enumerate(fm_moves)}
    count = 0
    skipped = 0
    legal_mass_sum = 0.0

    try:
        parquet_file = pq.ParquetFile(parquet_path)
        with tf.io.TFRecordWriter(output_path, options=options) as writer:
            for batch in parquet_file.iter_batches(batch_size=10000):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    board = chess.Board(row['fen'])
                    planes = board_to_planes_int8(board, extended=True)

                    raw_policy = np.asarray(row['policy'], dtype=np.float32)
                    if raw_policy.shape[0] != 1858:
                        skipped += 1
                        continue

                    # Le dataset marque les coups illégaux avec -1 : c'est un masque gratuit.
                    legal = raw_policy >= 0
                    if not legal.any():
                        skipped += 1
                        continue

                    policy = np.where(legal, raw_policy, 0.0)
                    s = policy.sum()
                    if s <= 0:
                        skipped += 1
                        continue
                    legal_mass_sum += float(legal.sum() and s / max(raw_policy[legal].sum(), 1e-9))
                    policy /= s

                    wdl = np.asarray(row['wdl'], dtype=np.float32)
                    writer.write(serialize_example(planes, policy, wdl, legal.astype(np.int8)))
                    count += 1
                del df
        return (f"fichier {file_index} : {count} positions ({skipped} ignorées), "
                f"masse légale moyenne {legal_mass_sum / max(count, 1):.3f} -> {output_path}")
    except Exception as e:  # noqa: BLE001
        return f"ERREUR fichier {file_index} : {e}"


def compute_legal_mask_fn(board):
    """Masque booléen 1858 des coups légaux (utilisable côté inférence)."""
    from src.utils.mapping_out_1858 import FlatMapper
    fm = fm_singleton()
    mask = np.zeros(1858, dtype=np.int8)
    for move in board.legal_moves:
        idx = fm.get_move_index(move.uci())
        if idx is not None:
            mask[idx] = 1
    return mask


_FM = None


def fm_singleton():
    global _FM
    if _FM is None:
        _FM = FlatMapper()
    return _FM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="nombre de fichiers parquet à traiter")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--input", default=INPUT_PARQUET_DIR)
    ap.add_argument("--output", default=OUTPUT_TFRECORD_DIR)
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input, "*.parquet")))
    if not files:
        print(f"Aucun .parquet dans {args.input}")
        return
    if args.limit:
        files = files[:args.limit]

    # On sérialise la LISTE des 1858 coups pour que chaque worker construise le même mapping.
    from multiprocessing import Pool
    from src.utils.mapping_out_1858 import generate_moves_list
    moves = generate_moves_list()

    print(f"{len(files)} fichier(s), {args.workers} worker(s), encodage {TOTAL_PLANES} plans")
    t0 = time.time()
    tasks = [(f, args.output, i, moves) for i, f in enumerate(files)]
    with Pool(processes=args.workers) as pool:
        for res in pool.imap_unordered(process_parquet_file, tasks):
            print(res, flush=True)
    print(f"Terminé en {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
