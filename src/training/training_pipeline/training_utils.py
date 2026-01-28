import tensorflow as tf

def parse_compressed_tfrecord(example_proto):
    feature_description = {
        'input': tf.io.FixedLenFeature([], tf.string),
        'policy': tf.io.FixedLenFeature([], tf.string),
        'value': tf.io.FixedLenFeature([], tf.string),
    }
    parsed = tf.io.parse_single_example(example_proto, feature_description)


    x = tf.io.parse_tensor(parsed['input'], out_type=tf.int8)
    x = tf.cast(x, tf.float32)
    x = tf.reshape(x, (8, 8, 12))


    p = tf.io.parse_tensor(parsed['policy'], out_type=tf.float32)
    p = tf.reshape(p, (1858,))


    v = tf.io.parse_tensor(parsed['wdl'], out_type=tf.float32)

    return x, {'policy': p, 'wdl': v}


def get_dataset(tfrecord_dir, batch_size=1024):

    files = tf.io.gfile.glob(tfrecord_dir + "/*.gz")


    dataset = tf.data.TFRecordDataset(files, compression_type="GZIP", num_parallel_reads=tf.data.AUTOTUNE)

    dataset = dataset.map(parse_compressed_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(50000)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)

    return dataset