"""
References:
    * https://github.com/facebookresearch/suncet/blob/master/src/snn_fine_tune.py
"""
from . import losses
import tensorflow as tf


suncet_loss = losses.get_suncet_loss()


def train_step(sup_loader, encoder: tf.keras.Model):
    """
    One step of fine-tuning after PAWS pre-training.

    :param sup_loader: data loader for the labeled support set
    :param encoder: trunk with projection head (with batchnorm layers frozen)
    :return: loss and gradients
    """
    # Unpack the data
    images, labels = sup_loader

    with tf.GradientTape() as tape:
        # Forward pass (z: projection output)
        _, z = encoder(images)
        loss = suncet_loss(z, labels)

    # Compute gradients
    gradients = tape.gradient(loss, encoder.trainable_variables)
    return loss, gradients
