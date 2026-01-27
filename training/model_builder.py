from keras import models
from tensorflow.keras import layers


def se_block(input_tensor, ratio=8):

    filters = input_tensor.shape[-1]

    se = layers.GlobalAveragePooling2D()(input_tensor)

    se = layers.Reshape((1, 1, filters))(se)

    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)

    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)

    x = layers.Multiply()([input_tensor, se])

    return x


def res_block(x, filters, se_ratio=8):
    shortcut = x

    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)
    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)

    x = se_block(x, se_ratio)

    x = layers.Add()([x, shortcut])

    return x


def policy_block(x, filters):

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    x = layers.Conv2D(73, 1, padding='same', use_bias=True)(x)

    x = layers.Flatten()(x)

    output_policy = layers.Activation('softmax', name='policy')(x)

    return output_policy


def value_block(x):

    x = layers.Conv2D(1, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)


    x = layers.Flatten()(x)

    x = layers.Dense(256, activation='swish')(x)

    output_value = layers.Dense(1, activation='tanh', name='value')(x)

    return output_value


def build_model(input_shape=(8, 8, 112), blocks=10, filters=128, se_ratio=8):

    inputs = layers.Input(shape=input_shape, name='input')

    x = layers.Conv2D(filters, 3, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('swish')(x)

    for i in range(blocks):
        x = res_block(x, filters, se_ratio)

    policy_out = policy_block(x, filters)
    value_out = value_block(x)

    output_model = models.Model(inputs=inputs, outputs=[policy_out, value_out], name="ChessNet_SE_ResNet")

    return output_model



if __name__ == "__main__":
    model = build_model(input_shape=(8, 8, 12), blocks=15, filters=256, se_ratio=8)

    model.summary()