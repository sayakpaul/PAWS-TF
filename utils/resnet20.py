# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# ResNet20, 56, 110, 164, 1001 version 2 for CIFAR-10
# Paper: https://arxiv.org/pdf/1603.05027.pdf

# Modified from:
# https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10_v2.py

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import (
    Conv2D,
    Dense,
    BatchNormalization,
    ReLU,
    Add,
    Activation,
)
from tensorflow.keras.layers import AveragePooling2D, GlobalAvgPool2D, Dropout
from tensorflow.keras.layers import experimental
from tensorflow.keras.regularizers import l2

WEIGHT_DECAY = 1e-6


def stem(inputs):
    """Construct Stem Convolutional Group
    inputs : the input vector
    """
    x = Conv2D(
        16,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    return x


def learner(x, n_blocks):
    """Construct the Learner
    x          : input to the learner
    n_blocks   : number of blocks in a group
    """
    # First Residual Block Group of 16 filters (Stage 1)
    # Quadruple (4X) the size of filters to fit the next Residual Group
    x = residual_group(x, 16, n_blocks, strides=(1, 1), n=4)

    # Second Residual Block Group of 64 filters (Stage 2)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 64, n_blocks, n=2)

    # Third Residual Block Group of 64 filters (Stage 3)
    # Double (2X) the size of filters and reduce feature maps by 75% (strides=2) to fit the next Residual Group
    x = residual_group(x, 128, n_blocks, n=2)
    return x


def residual_group(x, n_filters, n_blocks, strides=(2, 2), n=2):
    """Construct a Residual Group
    x         : input into the group
    n_filters : number of filters for the group
    n_blocks  : number of residual blocks with identity link
    strides   : whether the projection block is a strided convolution
    n         : multiplier for the number of filters out
    """
    # Double the size of filters to fit the first Residual Group
    x = projection_block(x, n_filters, strides=strides, n=n)

    # Identity residual blocks
    for _ in range(n_blocks):
        x = identity_block(x, n_filters, n)
    return x


def identity_block(x, n_filters, n=2):
    """Construct a Bottleneck Residual Block of Convolutions
    x        : input into the block
    n_filters: number of filters
    n        : multiplier for filters out
    """
    # Save input vector (feature maps) for the identity link
    shortcut = x

    ## Construct the 1x1, 3x3, 1x1 residual block (fig 3c)

    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Bottleneck layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (3, 3),
        strides=(1, 1),
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Dimensionality restoration - increase the number of output filters by 2X or 4X
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters * n,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Add the identity link (input) to the output of the residual block
    x = Add()([x, shortcut])
    return x


def projection_block(x, n_filters, strides=(2, 2), n=2):
    """Construct a Bottleneck Residual Block with Projection Shortcut
    Increase the number of filters by 2X (or 4X on first stage)
    x        : input into the block
    n_filters: number of filters
    strides  : whether the first convolution is strided
    n        : multiplier for number of filters out
    """
    # Construct the projection shortcut
    # Increase filters by 2X (or 4X) to match shape when added to output of block
    shortcut = Conv2D(
        n_filters * n,
        (1, 1),
        strides=strides,
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    ## Construct the 1x1, 3x3, 1x1 convolution block

    # Dimensionality reduction
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Bottleneck layer - feature pooling when strides=(2, 2)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters,
        (3, 3),
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Dimensionality restoration - increase the number of filters by 2X (or 4X)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(
        n_filters * n,
        (1, 1),
        strides=(1, 1),
        use_bias=False,
        kernel_regularizer=l2(WEIGHT_DECAY),
    )(x)

    # Add the projection shortcut to the output of the residual block
    x = Add()([shortcut, x])
    return x


def projection_head(x, hidden_dim=128):
    """Constructs the projection head."""
    for i in range(2):
        x = Dense(hidden_dim, use_bias=False,
                  name=f"projection_layer_{i}",
                  kernel_regularizer=l2(WEIGHT_DECAY))(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)
    outputs = Dense(hidden_dim, use_bias=False,
                    name="projection_output")(x)
    return outputs

def prediction_head(x, hidden_dim=128, mx=4):
    """Constructs the prediction head."""
    x = BatchNormalization()(x)
    x = Dense(hidden_dim//mx, use_bias=False,
                  name=f"prediction_layer_0",
                  kernel_regularizer=l2(WEIGHT_DECAY))(x)
    x = BatchNormalization()(x)
    x =  Activation("relu")(x)
    x = Dense(hidden_dim, use_bias=False,
              name="prediction_output",
              kernel_regularizer=l2(WEIGHT_DECAY))(x)
    return x

# -------------------
# Model      | n   |
# ResNet20   | 2   |
# ResNet56   | 6   |
# ResNet110  | 12  |
# ResNet164  | 18  |
# ResNet1001 | 111 |

def get_network(n=2, hidden_dim=128, use_pred=False):
    depth = n * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    # The input tensor
    inputs = Input(shape=(32, 32, 3))
    x = experimental.preprocessing.Rescaling(scale=1./127.5,
                                             offset=-1)(inputs)

    # The Stem Convolution Group
    x = stem(x)

    # The learner
    x = learner(x, n_blocks)

    # Projections
    x = GlobalAvgPool2D()(x)
    outputs = projection_head(x, hidden_dim=hidden_dim)

    # Predictions
    if use_pred:
        outputs = prediction_head(x)

    return Model(inputs, outputs)

