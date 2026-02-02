import os
import tensorflow as tf
import tf2onnx
import onnx
from onnxconverter_common import float16
from onnxruntime.quantization import quantize_dynamic, QuantType


def export_to_onnx_triplet(model_path, output_dir, input_shape):
    """
    Convertit un modèle Keras en 3 versions ONNX : FP32, FP16 et INT8.

    Args:
        model_path (str): Chemin vers le fichier .keras source.
        output_dir (str): Dossier de destination.
        input_shape (tuple): Forme d'entrée (ex: (None, 8, 8, 19)).

    Returns:
        dict: Les chemins des 3 fichiers générés.
    """

    # 1. Préparation des noms de fichiers
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(model_path))[0]

    path_fp32 = os.path.join(output_dir, f"{base_name}_fp32.onnx")
    path_fp16 = os.path.join(output_dir, f"{base_name}_fp16.onnx")
    path_int8 = os.path.join(output_dir, f"{base_name}_int8.onnx")

    print(f"🚀 Début de l'export pour : {base_name}")

    # 2. Force le mode Float32 (Crucial si entraîné en Mixed Precision)
    # Cela évite les bugs de graphe lors de la conversion
    tf.keras.mixed_precision.set_global_policy('float32')

    # 3. Chargement du modèle Keras
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception as e:
        print(f"❌ Erreur lors du chargement du modèle Keras : {e}")
        return None

    # Définition de la signature (Input Spec)
    input_signature = [tf.TensorSpec(input_shape, tf.float32, name='input_board')]

    # ---------------------------------------------------------
    # ÉTAPE A : Export FP32 (Master)
    # ---------------------------------------------------------
    print(f"🔹 Conversion FP32 en cours...")
    tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        opset=13,
        output_path=path_fp32
    )
    print(f"✅ FP32 sauvegardé : {path_fp32}")

    # ---------------------------------------------------------
    # ÉTAPE B : Conversion FP16
    # ---------------------------------------------------------
    print(f"🔹 Conversion FP16 en cours...")
    model_onnx = onnx.load(path_fp32)
    model_fp16 = float16.convert_float_to_float16(model_onnx)
    onnx.save(model_fp16, path_fp16)
    print(f"✅ FP16 sauvegardé : {path_fp16}")

    # ---------------------------------------------------------
    # ÉTAPE C : Quantisation INT8 (Dynamique)
    # ---------------------------------------------------------
    print(f"🔹 Quantisation INT8 en cours...")
    quantize_dynamic(
        model_input=path_fp32,
        model_output=path_int8,
        weight_type=QuantType.QUInt8
    )
    print(f"✅ INT8 sauvegardé : {path_int8}")

    print(f"🎉 Terminé avec succès.\n")

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