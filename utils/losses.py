"""
References:
    * https://arxiv.org/abs/2104.13963
    * https://github.com/facebookresearch/suncet/blob/master/src/losses.py
Majority of the code comes from here:
 https://github.com/facebookresearch/suncet/blob/master/src/losses.py
"""

from copy import deepcopy
import tensorflow as tf


def get_paws_loss(multicrop=6, tau=0.1, T=0.25, me_max=True):
    """
    Computes PAWS loss

    :param multicrop: number of small views
    :param tau: cosine temperature
    :param T: sharpening temperature
    :param me_max: mean entropy maximization flag
    :return: PAWS loss
    """

    def sharpen(proba):
        """ Target sharpening function """
        sharp_p = proba ** (1.0 / T)
        sharp_p /= tf.reduce_sum(sharp_p, axis=1, keepdims=True)
        return sharp_p

    def snn(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: Normalize embeddings
        query = tf.math.l2_normalize(query)
        supports = tf.math.l2_normalize(supports)

        # Step 2: Compute similarity
        return tf.nn.softmax(query @ tf.transpose(supports) / tau, axis=1) @ labels

    def loss(
        anchor_views,
        anchor_supports,
        anchor_support_labels,
        target_views,
        target_supports,
        target_support_labels,
        sharpen=sharpen,
        snn=snn,
    ):
        # -- NOTE: num views of each unlabeled instance = 2+multicrop
        # 2 global views and 6 local views
        batch_size = len(anchor_views) // (2 + multicrop)

        # Step 1: Compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)

        # Step 2: Compute targets for anchor predictions
        targets = tf.stop_gradient(
            snn(target_views, target_supports, target_support_labels)
        )
        targets = sharpen(targets)
        if multicrop > 0:
            mc_target = 0.5 * (targets[:batch_size] + targets[batch_size:])
            targets = tf.concat(
                [targets, *[mc_target for _ in range(multicrop)]], axis=0
            )
        # For numerical stability
        mask = tf.math.less(targets, 1e-4)
        mask = tf.cast(mask, dtype=targets.dtype)
        targets *= mask

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.log(probs ** (-targets)), axis=1))

        # Step 4: compute me-max regularizer
        rloss = 0.0
        if me_max:
            avg_probs = tf.reduce_mean(sharpen(probs), axis=0)
            rloss -= tf.reduce_sum(tf.math.log(avg_probs ** (-avg_probs)))

        return loss, rloss

    return loss

def get_suncet_loss(num_classes=10,
    batch_size=64,
    temperature=0.1,
    rank=0):
    """
    Computes supervised noise contrastive estimation loss (refer
    https://arxiv.org/abs/2006.10803)

    :param num_classes: number of image classes
    :param batch_size: number of images per class per batch
    :param temperature: cosine temperature
    :param rank: denotes single-GPU
    :return: SUNCET loss
    """
    local_images = batch_size * num_classes
    total_images = deepcopy((local_images))
    diag_mask = tf.ones((local_images, total_images))
    offset = rank * local_images
    for i in range(local_images):
        diag_mask[i, offset + i] = 0.

    def contrastive_loss(z, labels):

        # Step 1: normalize embeddings
        z = tf.math.l2_normalize(z)

        # Step 2: compute class predictions
        exp_cs = tf.math.exp(z @ tf.transpose(z) / temperature) * diag_mask
        exp_cs_sum = tf.reduce_sum(exp_cs, axis=1, keepdims=True)
        probs = tf.math.divide(exp_cs, exp_cs_sum) @ labels

        # Step 3: compute loss for predictions
        targets = labels[offset : offset+local_images]
        overlap = probs ** (-targets)
        loss = tf.reduce_mean(tf.reduce_sum(tf.math.log(overlap), axis=1))
        return loss

    return contrastive_loss
