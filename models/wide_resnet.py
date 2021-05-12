"""
References:
    * http://arxiv.org/abs/1605.07146
    * https://github.com/asmith26/wide_resnets_keras/blob/master/main.py
Majority of the code comes from here:
https://github.com/asmith26/wide_resnets_keras/blob/master/main.py
"""

# Imports
from tensorflow.keras import regularizers
from tensorflow.keras import layers
import tensorflow as tf

WEIGHT_DECAY = 1e-6
INIT = "he_normal"
DEPTH = 28
WIDTH_MULT = 2


def wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        conv_params = [[3, 3, stride, "same"], [3, 3, (1, 1), "same"]]

        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(net)
                    net = layers.Activation("relu")(net)
                    convs = net
                else:
                    convs = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(net)
                    convs = layers.Activation("relu")(convs)
                convs = layers.Conv2D(
                    n_bottleneck_plane,
                    (v[0], v[1]),
                    strides=v[2],
                    padding=v[3],
                    kernel_initializer=INIT,
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                    use_bias=False,
                )(convs)
            else:
                convs = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(convs)
                convs = layers.Activation("relu")(convs)
                convs = layers.Conv2D(
                    n_bottleneck_plane,
                    (v[0], v[1]),
                    strides=v[2],
                    padding=v[3],
                    kernel_initializer=INIT,
                    kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                    use_bias=False,
                )(convs)

        # Shortcut connection: identity function or 1x1
        # convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in
        #   each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = layers.Conv2D(
                n_output_plane,
                (1, 1),
                strides=stride,
                padding="same",
                kernel_initializer=INIT,
                kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
                use_bias=False,
            )(net)
        else:
            shortcut = net

        return layers.Add()([convs, shortcut])

    return f


# Stacking residual Units on the same stage
def layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2, int(count + 1)):
            net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
        return net

    return f


def projection_head(x, hidden_dim=128):
    """Constructs the projection head."""
    for i in range(2):
        x = layers.Dense(
            hidden_dim,
            use_bias=False,
            name=f"projection_layer_{i}",
            kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        )(x)
        x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
        x = layers.Activation("relu")(x)
    outputs = layers.Dense(hidden_dim, use_bias=False, name="projection_output")(x)
    return outputs


def prediction_head(x, hidden_dim=128, mx=4):
    """Constructs the prediction head."""
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = layers.Dense(
        hidden_dim // mx,
        use_bias=False,
        name=f"prediction_layer_0",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)
    x = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
    x = layers.Activation("relu")(x)
    x = layers.Dense(
        hidden_dim,
        use_bias=False,
        name="prediction_output",
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
    )(x)
    return x


def get_network(hidden_dim=128, use_pred=False, return_before_head=True):
    n = (DEPTH - 4) / 6
    n_stages = [16, 16 * WIDTH_MULT, 32 * WIDTH_MULT, 64 * WIDTH_MULT]

    inputs = layers.Input(shape=(None, None, 3))
    x = layers.experimental.preprocessing.Rescaling(scale=1.0 / 255)(inputs)
    x = layers.experimental.preprocessing.Normalization(
        mean=[0.4914, 0.4822, 0.4465],
        variance=[i ** 2 for i in [0.2023, 0.1994, 0.2010]],
    )(x)

    conv1 = layers.Conv2D(
        n_stages[0],
        (3, 3),
        strides=1,
        padding="same",
        kernel_initializer=INIT,
        kernel_regularizer=regularizers.l2(WEIGHT_DECAY),
        use_bias=False,
    )(x)

    # Add wide residual blocks
    block_fn = wide_basic
    conv2 = layer(
        block_fn,
        n_input_plane=n_stages[0],
        n_output_plane=n_stages[1],
        count=n,
        stride=(1, 1),
    )(
        conv1
    )  # Stage 1
    conv3 = layer(
        block_fn,
        n_input_plane=n_stages[1],
        n_output_plane=n_stages[2],
        count=n,
        stride=(2, 2),
    )(
        conv2
    )  # Stage 2
    conv4 = layer(
        block_fn,
        n_input_plane=n_stages[2],
        n_output_plane=n_stages[3],
        count=n,
        stride=(2, 2),
    )(
        conv3
    )  # Stage 3

    batch_norm = layers.BatchNormalization(epsilon=1e-05, momentum=0.1)(conv4)
    relu = layers.Activation("relu")(batch_norm)

    # Trunk outputs
    trunk_outputs = layers.GlobalAveragePooling2D()(relu)

    # Projections
    projection_outputs = projection_head(trunk_outputs, hidden_dim=hidden_dim)
    if return_before_head:
        model = tf.keras.Model(inputs, [trunk_outputs, projection_outputs])
    else:
        model = tf.keras.Model(inputs, projection_outputs)

    # Predictions
    if use_pred:
        prediction_outputs = prediction_head(projection_outputs)
        if return_before_head:
            model = tf.keras.Model(inputs, [projection_outputs, prediction_outputs])
        else:
            model = tf.keras.Model(inputs, prediction_outputs)

    return model
