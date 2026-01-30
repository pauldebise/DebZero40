import random

import tensorflow as tf
import os
from datetime import datetime

def parse_compressed_tfrecord(example_proto):
    feature_description = {
        'input': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.string),
        'wdl': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)


    x = tf.io.parse_tensor(parsed['input'], out_type=tf.int8)
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, (8, 8, 12))


    p = tf.io.parse_tensor(parsed['policy'], out_type=tf.float32)
    p = tf.reshape(p, (1858,))


    v = tf.io.parse_tensor(parsed['wdl'], out_type=tf.float32)
    v = tf.reshape(v, (3,))

    return x, {'policy': p, 'wdl': v}


def get_datasets(
        tfrecord_dir,
        batch_size=1024,
        validation_split=0.1,
        avg_samples_per_file=1_000_000
):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.gz"))

    random.shuffle(files)

    num_val = int(len(files) * validation_split)
    if num_val == 0 and len(files) > 1 and validation_split > 0:
        num_val = 1

    val_files = files[:num_val]
    train_files = files[num_val:]


    train_steps = int((len(train_files) * avg_samples_per_file * 0.95) // batch_size)
    val_steps = int((len(val_files) * avg_samples_per_file * 0.95) // batch_size)

    print(f"Total fichiers: {len(files)}")
    print(f" -> Train : {len(train_files)} fichiers | ~{train_steps} steps/epoch")
    print(f" -> Val   : {len(val_files)} fichiers | ~{val_steps} steps")


    def build_pipeline(file_list, is_training):
        if not file_list:
            return None

        ds = tf.data.TFRecordDataset(
            file_list,
            compression_type="GZIP",
            num_parallel_reads=tf.data.AUTOTUNE
        )

        ds = ds.map(parse_compressed_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            ds = ds.shuffle(buffer_size=50000)
            ds = ds.batch(batch_size, drop_remainder=True)
            ds = ds.repeat()
        else:
            ds = ds.batch(batch_size, drop_remainder=False)

        ds = ds.prefetch(tf.data.AUTOTUNE)
        return ds

    train_dataset = build_pipeline(train_files, is_training=True)
    val_dataset = build_pipeline(val_files, is_training=False)


    return train_dataset, val_dataset, train_steps, val_steps


def train_model(
        model,
        train_dataset,
        val_dataset,
        train_steps = None,
        validation_steps = None,
        epochs=100,
        initial_epoch=0,
        learning_rate=1e-3,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        use_mixed_precision=False,
        run_name=None,
        base_log_dir="logs",
        base_model_dir="saved_models",
        patience_stop=10,
        patience_lr=3,
        lr_factor=0.1,
        save_best_only=True
):
    """
    Training function for DebZero40
    """

    if use_mixed_precision:
        if tf.keras.mixed_precision.global_policy().name != 'mixed_float16':
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("Mixed Precision (FP16) activated")


    if run_name is None:
        run_name = datetime.now().strftime("%Y%m%d-%H%M%S")

    log_dir = os.path.join(base_log_dir, run_name)
    model_dir = os.path.join(base_model_dir, run_name)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Run : {run_name} | Epochs: {initial_epoch} -> {epochs}")
    print(f"Loss weights -> Policy: {policy_loss_weight} | WDL: {value_loss_weight}")


    callbacks = [

        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(model_dir, "best_model.keras"),
            monitor='val_loss',
            save_best_only=save_best_only,
            mode='min',
            verbose=1
        ),

        tf.keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            update_freq=100,
            profile_batch=0
        ),

        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=lr_factor,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=1
        ),

        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience_stop,
            restore_best_weights=True,
            verbose=1
        )
    ]


    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)




    model.compile(
        optimizer=optimizer,

        loss={
            'policy': 'categorical_crossentropy',
            'wdl': 'categorical_crossentropy',
        },

        loss_weights={
            'policy': policy_loss_weight,
            'wdl': value_loss_weight
        },
        metrics={
            'policy': 'categorical_accuracy',
            'wdl': ['categorical_accuracy', 'mae']
        }
    )


    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        initial_epoch=initial_epoch,
        callbacks=callbacks,
        verbose=1,
        steps_per_epoch=train_steps,
        validation_steps=validation_steps
    )


    final_path = os.path.join(model_dir, run_name+".keras")
    model.save(final_path)

    return history