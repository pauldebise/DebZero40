from keras import models
from tensorflow.keras import layers

from src.utils.board_encoder import PIECE_PLANES, TOTAL_PLANES


def se_block(input_tensor, ratio=8):
    """Bloc Squeeze-and-Excitation (attention par canal)."""
    filters = input_tensor.shape[-1]
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape((1, 1, filters))(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.Multiply()([input_tensor, se])


def res_block(x, filters, se_ratio=8):
    shortcut = x
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = se_block(x, se_ratio)
    return layers.Add()([x, shortcut])


def _masked_softmax(tensors):
    logits, mask = tensors
    return layers.Activation('softmax', dtype='float32', name='policy')(logits + (1.0 - mask) * -1e9)


def policy_block(x, filters, mask_tensor=None):
    """
    Tête de policy, encodage de sortie 1858 coups.

    Si `mask_tensor` est fourni (entrée booléenne 1858), les coups illégaux reçoivent une
    probabilité nulle avant le softmax. Sans masque, le réseau observé (N1_10x64) dépense
    jusqu'à 36 % de sa probabilité sur des coups impossibles et ne donne que 0,02 % au roque.
    """
    x = layers.Conv2D(32, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(filters * 2, activation='swish')(x)

    if mask_tensor is None:
        return layers.Dense(1858, activation='softmax', dtype='float32', name='policy')(x)

    logits = layers.Dense(1858, name='policy_logits')(x)
    return _masked_softmax([logits, mask_tensor])


def value_block(x):
    x = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='swish')(x)
    return layers.Dense(3, activation='softmax', name='wdl')(x)


def build_model(input_shape=(8, 8, PIECE_PLANES), blocks=6, filters=64, se_ratio=8,
                use_legal_mask=False):
    """Construit le réseau.

    input_shape=(8, 8, 12) -> architecture historique, compatible avec les nets existants.
    input_shape=(8, 8, 17) -> ajoute les droits de roque et la case de prise en passant.
    use_legal_mask=True    -> ajoute une entrée `legal_mask` (1858) qui met à -inf les
                              coups illégaux avant le softmax de policy.
    """
    inputs = layers.Input(shape=input_shape, name='input')

    mask_tensor = None
    model_inputs = inputs
    if use_legal_mask:
        mask_tensor = layers.Input(shape=(1858,), name='legal_mask')
        model_inputs = [inputs, mask_tensor]

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    for _ in range(blocks):
        x = res_block(x, filters, se_ratio)

    policy_out = policy_block(x, filters, mask_tensor=mask_tensor)
    value_out = value_block(x)

    name = "ChessNet_SE_ResNet_Masked" if use_legal_mask else "ChessNet_SE_ResNet"
    return models.Model(inputs=model_inputs, outputs=[policy_out, value_out], name=name)


if __name__ == "__main__":
    model = build_model(input_shape=(8, 8, TOTAL_PLANES), blocks=12, filters=128,
                        se_ratio=8, use_legal_mask=True)
    model.summary()
