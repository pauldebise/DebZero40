import os
import tensorflow as tf
import tf2onnx
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_to_onnx_triplet(model_path, output_dir, input_shape):


    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(model_path))[0]

    path_fp32 = os.path.join(output_dir, f"{base_name}_fp32.onnx")
    path_fp16 = os.path.join(output_dir, f"{base_name}_fp16.onnx")
    path_int8 = os.path.join(output_dir, f"{base_name}_int8.onnx")

    tf.keras.mixed_precision.set_global_policy('float32')

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        return None

    input_signature = [tf.TensorSpec(input_shape, tf.float32, name='input_board')]

    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=path_fp32
    )

    model_onnx = onnx.load(path_fp32)
    model_fp16 = float16.convert_float_to_float16(model_onnx)
    onnx.save(model_fp16, path_fp16)

    quantize_dynamic(
        model_input=path_fp32,
        model_output=path_int8,
        weight_type=QuantType.QUInt8
    )


    return {
        "fp32": path_fp32,
        "fp16": path_fp16,
        "int8": path_int8
    }




if __name__ == "__main__":

    os.chdir("../..")

    MES_FICHIERS = export_to_onnx_triplet(
        model_path="nets/test3/best_model.keras",
        output_dir="nets/test3",
        input_shape=(None, 8, 8, 12)
    )