import os
import glob
import numpy as np
import pyarrow.parquet as pq
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from src.utils.board_encoder import fen_to_planes_int8
from multiprocessing import Pool, cpu_count
import time


INPUT_PARQUET_DIR = "../../../data/parquet_files/"
OUTPUT_TFRECORD_DIR = "../../../data/tfrecords_gzip/"
NUM_WORKERS = max(1, cpu_count() - 2)

#Disabling GPU to avoid any conflict
tf.config.set_visible_devices([], 'GPU')



def _bytes_feature(value):
    """Helper for binary datas"""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(planes, policy, wdl):
    """
    planes: (8, 8, 12) int8
    policy: (1858,) float32
    wdl: (3,) float32
    """
    feature = {
        'input': _bytes_feature(tf.io.serialize_tensor(planes)),
        'policy': _bytes_feature(tf.io.serialize_tensor(policy)),
        'wdl': _bytes_feature(tf.io.serialize_tensor(wdl)),
    }
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def process_parquet_file(args):

    """
    Workers read one parquet file each, with batches of 10000 lines to decrease RAM consumption.
    """

    parquet_path, output_dir, file_index = args

    filename = os.path.basename(parquet_path).replace('.parquet', '')
    output_path = os.path.join(output_dir, f"{filename}.tfrecord.gz")
    options = tf.io.TFRecordOptions(compression_type='GZIP')

    count = 0
    try:

        parquet_file = pq.ParquetFile(parquet_path)

        with tf.io.TFRecordWriter(output_path, options=options) as writer:

            for batch in parquet_file.iter_batches(batch_size=10000):

                df_batch = batch.to_pandas()

                for _, row in df_batch.iterrows():

                    input_tensor = fen_to_planes_int8(row['fen'])
                    policy_vec = np.array(row['policy'], dtype=np.float32)
                    policy_vec[policy_vec < 0] = 0.0
                    value_target = np.array(row['wdl'], dtype=np.float32)

                    serialized = serialize_example(input_tensor, policy_vec, value_target)
                    writer.write(serialized)
                    count += 1

                del df_batch

        return f"File {file_index} finished : {output_path} ({count} positions)"

    except Exception as e:
        return f"Error encountered for file {file_index} : {str(e)}"


def process_dataset(limit=None):

    if not os.path.exists(OUTPUT_TFRECORD_DIR):
        os.makedirs(OUTPUT_TFRECORD_DIR)


    parquet_files = sorted(glob.glob(os.path.join(INPUT_PARQUET_DIR, "*.parquet")))

    if not parquet_files:
        print(f"No files .parquet found in {INPUT_PARQUET_DIR}")
        return

    if limit is not None:
        original_count = len(parquet_files)
        parquet_files = parquet_files[:limit]
        print(f"LIMIT ACTIVATED : {len(parquet_files)} files processed out of the {original_count} availables.")

    print(f"Starting the conversion of {len(parquet_files)} files with {NUM_WORKERS} workers...")
    print(f"Config: Input INT8 | Policy/WDL FP32 | Compression GZIP")


    tasks = []
    for i, p_file in enumerate(parquet_files):
        tasks.append((p_file, OUTPUT_TFRECORD_DIR, i))

    start_time = time.time()


    with Pool(processes=NUM_WORKERS) as pool:

        for result in pool.imap_unordered(process_parquet_file, tasks):
            print(result)

    duration = time.time() - start_time
    print(f"\n Finished in {duration / 60:.2f} minutes.")




if __name__ == '__main__':
    process_dataset()