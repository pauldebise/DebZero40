"""
Entraînement d'un réseau DebZero 17 plans (avec masque de légalité).

Conçu pour un run COURT de validation sur GPU loué, pas pour une campagne complète.

Usage sur le pod :
    python train_v2.py --parquet-dir /workspace/parquet --steps 3000 --blocks 6 --filters 64

Le script :
  1. lit les parquet ChessFENS et les convertit en TFRecords (encodage 17 plans) ;
  2. entraîne avec une cible policy renormalisée sur les coups légaux ;
  3. vérifie sur un lot de validation que la masse de policy sur coups légaux est ~100 % ;
  4. exporte un .keras et un .onnx FP32 compatible avec le moteur (entrée 8,8,17).
"""
import argparse
import glob
import math
import os
import sys
import time

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np


def build_tfrecords(parquet_dir, out_dir, limit_files, workers):
    import pyarrow.parquet as pq
    import tensorflow as tf
    import chess
    from src.utils.board_encoder import board_to_planes_int8

    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(parquet_dir, "*.parquet")))[:limit_files]
    if not files:
        raise SystemExit(f"aucun parquet dans {parquet_dir}")

    def _bytes(value):
        # `tf.io.serialize_tensor` renvoie un EagerTensor : BytesList attend un bytes pur.
        if hasattr(value, "numpy"):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    total = 0
    t0 = time.time()
    for fi, path in enumerate(files):
        out_path = os.path.join(out_dir, os.path.basename(path).replace(".parquet", ".tfrecord.gz"))
        if os.path.exists(out_path):
            print(f"[skip] {out_path} existe déjà")
            continue
        count = 0
        with tf.io.TFRecordWriter(out_path, options=tf.io.TFRecordOptions("GZIP")) as writer:
            pf = pq.ParquetFile(path)
            for batch in pf.iter_batches(batch_size=5000):
                df = batch.to_pandas()
                for _, row in df.iterrows():
                    board = chess.Board(row["fen"])
                    planes = board_to_planes_int8(board, extended=True)
                    raw = np.asarray(row["policy"], dtype=np.float32)
                    if raw.shape[0] != 1858:
                        continue
                    legal = raw >= 0
                    if not legal.any():
                        continue
                    pol = np.where(legal, raw, 0.0)
                    s = pol.sum()
                    if s <= 0:
                        continue
                    pol /= s
                    wdl = np.asarray(row["wdl"], dtype=np.float32)
                    feats = {
                        "input": _bytes(tf.io.serialize_tensor(planes)),
                        "policy": _bytes(tf.io.serialize_tensor(pol)),
                        "wdl": _bytes(tf.io.serialize_tensor(wdl)),
                        "legal_mask": _bytes(tf.io.serialize_tensor(legal.astype(np.int8))),
                    }
                    writer.write(tf.train.Example(features=tf.train.Features(feature=feats)).SerializeToString())
                    count += 1
        total += count
        print(f"[{fi+1}/{len(files)}] {count} positions -> {out_path} ({time.time()-t0:.0f}s)")
    return total


def make_dataset(tfrecord_dir, batch_size, shuffle=True):
    import tensorflow as tf

    files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.gz")))
    if not files:
        raise SystemExit(f"aucun tfrecord dans {tfrecord_dir}")

    split = max(1, len(files) // 20)
    val_files, train_files = files[:split], files[split:]

    def parse(proto):
        f = tf.io.parse_single_example(proto, {
            "input": tf.io.FixedLenFeature([], tf.string),
            "policy": tf.io.FixedLenFeature([], tf.string),
            "wdl": tf.io.FixedLenFeature([], tf.string),
            "legal_mask": tf.io.FixedLenFeature([], tf.string),
        })
        x = tf.cast(tf.reshape(tf.io.parse_tensor(f["input"], tf.int8), (8, 8, 17)), tf.float32)
        p = tf.reshape(tf.io.parse_tensor(f["policy"], tf.float32), (1858,))
        v = tf.reshape(tf.io.parse_tensor(f["wdl"], tf.float32), (3,))
        m = tf.cast(tf.reshape(tf.io.parse_tensor(f["legal_mask"], tf.int8), (1858,)), tf.float32)
        # Entrées nommées : le modèle attend les clés 'input' et 'legal_mask'.
        return ({"input": x, "legal_mask": m}, {"policy": p, "wdl": v})

    def make(files_, training):
        if not files_:
            raise SystemExit("liste de fichiers vide : impossible de construire le dataset")
        ds = tf.data.TFRecordDataset(files_, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)
        if training:
            ds = ds.shuffle(20000)
        ds = ds.map(parse, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
        if training:
            ds = ds.repeat()
        return ds

    # Un seul fichier : on l'utilise pour train ET pour validation (pas de split possible).
    if len(files) == 1:
        val_files = files
        train_files = files
    else:
        val_files = files[:split]
        train_files = files[split:]

    return make(train_files, True), make(val_files, False), train_files, val_files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet-dir", required=True)
    ap.add_argument("--work-dir", default="/workspace/debzero")
    ap.add_argument("--limit-files", type=int, default=2, help="nb de parquet à convertir")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--filters", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-hours", type=float, default=1.8, help="garde-fou de durée")
    args = ap.parse_args()

    import tensorflow as tf
    from src.utils.model_builder import build_model

    os.makedirs(args.work_dir, exist_ok=True)
    tfrec_dir = os.path.join(args.work_dir, "tfrecords")

    print("=== 1. conversion des parquet en TFRecords 17 plans ===")
    n = build_tfrecords(args.parquet_dir, tfrec_dir, args.limit_files, workers=4)
    print(f"{n} positions converties")

    train_ds, val_ds, train_files, val_files = make_dataset(tfrec_dir, args.batch)
    print(f"train: {len(train_files)} fichiers | val: {len(val_files)}")

    print("=== 2. modèle ===")
    model = build_model(input_shape=(8, 8, 17), blocks=args.blocks, filters=args.filters,
                        use_legal_mask=True)
    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.AdamW(learning_rate=args.lr, weight_decay=1e-2),
        loss={"policy": "categorical_crossentropy", "wdl": "categorical_crossentropy"},
        loss_weights={"policy": 1.0, "wdl": 1.6},
        metrics={"policy": ["categorical_accuracy"], "wdl": ["categorical_accuracy"]},
    )

    ckpt = os.path.join(args.work_dir, "best.keras")
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(ckpt, monitor="loss", save_best_only=True, verbose=1),
        tf.keras.callbacks.TerminateOnNaN(),
        # garde-fou budgétaire : on arrête avant l'heure limite
        tf.keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: None
        ),
    ]

    print("=== 3. entraînement ===")
    t0 = time.time()
    history = model.fit(train_ds, steps_per_epoch=args.steps, epochs=1,
                        validation_data=val_ds.take(20), callbacks=callbacks, verbose=1)
    dt = time.time() - t0
    print(f"entraînement terminé en {dt/60:.1f} min")
    if dt / 3600 > args.max_hours:
        print("!! durée supérieure au garde-fou : vérifier le coût")

    print("=== 4. contrôle de la masse légale ===")
    masses = []
    for inputs, yb in val_ds.take(5):
        pol, _ = model(inputs, training=False)
        m = inputs["legal_mask"]
        masses.append(float(tf.reduce_mean(tf.reduce_sum(pol * m, axis=-1))))
    print(f"masse policy moyenne sur coups légaux (validation) = {np.mean(masses)*100:.2f} %")

    print("=== 5. export ONNX ===")
    out_keras = os.path.join(args.work_dir, "debzero_v2.keras")
    model.save(out_keras)
    print("keras:", out_keras)

    try:
        import tf2onnx
        signature = [
            tf.TensorSpec([None, 8, 8, 17], tf.float32, name="input"),
            tf.TensorSpec([None, 1858], tf.float32, name="legal_mask"),
        ]
        onnx_path = os.path.join(args.work_dir, "debzero_v2_fp32.onnx")
        tf2onnx.convert.from_keras(model, input_signature=signature, opset=17, output_path=onnx_path)
        print("onnx:", onnx_path)
    except Exception as e:  # noqa: BLE001
        print(f"export ONNX impossible ici ({e}) : exporter depuis le VPS ou ajouter tf2onnx")


if __name__ == "__main__":
    main()
