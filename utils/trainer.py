"""
References:
    * https://github.com/facebookresearch/suncet/blob/master/src/paws_train.py
"""
from . import losses
import tensorflow as tf


paws_loss = losses.get_paws_loss(multicrop=6, tau=0.1, T=0.25, me_max=True)


def train_step(unsup_images, sup_loader, encoder: tf.keras.Model):
    """
    One step of PAWS training.

    :param unsup_images: unsupervised images (nb_crops, batch_size, h, w, nb_channels)
    :param sup_loader: data loader for the labeled support set
    :param encoder: trunk with projection head
    :return: loss and gradients
    """
    # Get batch size for unsupervised images
    u_batch_size = tf.shape(unsup_images[0])[0]

    # Unsupervised imgs (2 views)
    imgs = tf.concat([u for u in unsup_images[:2]], axis=0)
    # Unsupervised multicrop img views (6 views)
    mc_imgs = tf.concat([u for u in unsup_images[2:]], axis=0)
    # Segregate images and labels from support set
    simgs, labels = sup_loader
    # Concatenate unlabeled images and labeled support images
    imgs, simgs = tf.cast(imgs, tf.float32), tf.cast(simgs, tf.float32)
    imgs = tf.concat([imgs, simgs], axis=0)

    with tf.GradientTape() as tape:
        # Pass through the global views (including images from the
        # support set) and multicrop views.
        # h: trunk output, z, z_mc: projection output
        h, z = encoder(imgs)
        _, z_mc = encoder(mc_imgs)

        # Determine anchor views / supports and their  corresponding
        # target views/supports (we are not using prediction head)
        h = z
        target_supports = h[2 * u_batch_size :]
        target_views = h[: 2 * u_batch_size]
        target_views = tf.concat(
            [target_views[u_batch_size:], target_views[:u_batch_size]], axis=0
        )
        anchor_supports = z[2 * u_batch_size :]
        anchor_views = z[: 2 * u_batch_size]
        anchor_views = tf.concat([anchor_views, z_mc], axis=0)

        # Compute paws loss with me-max regularization
        (ploss, me_max) = paws_loss(
            anchor_views=anchor_views,
            anchor_supports=anchor_supports,
            anchor_support_labels=labels,
            target_views=target_views,
            target_supports=target_supports,
            target_support_labels=labels,
        )
        loss = ploss + me_max
    # Compute gradients
    gradients = tape.gradient(loss, encoder.trainable_variables)
    return ploss, me_max, gradients