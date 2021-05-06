"""
References:
    * https://arxiv.org/abs/2104.13963
    * https://github.com/facebookresearch/suncet/blob/master/src/losses.py
"""


import tensorflow as tf


def get_paws_loss(multicrop=6,
    tau=0.1,
    T=0.25,
    me_max=True):
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
        sharp_p = proba ** (1. / T)
        sharp_p /= tf.reduce_sum(sharp_p, axis=1, keepdims=True)
        return sharp_p

    def snn(query, supports, labels):
        """ Soft Nearest Neighbours similarity classifier """
        # Step 1: Normalize embeddings
        query = tf.math.l2_normalize(query)
        supports =tf.math.l2_normalize(supports)

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
            snn=snn
    ):
        # -- NOTE: num views of each unlabeled instance = 2+multicrop
        # 2 global views and 6 local views
        batch_size = len(anchor_views) // (2 + multicrop)

        # Step 1: Compute anchor predictions
        probs = snn(anchor_views, anchor_supports, anchor_support_labels)

        # Step 2: Compute targets for anchor predictions
        targets = tf.stop_gradient(snn(target_views, target_supports,
                      target_support_labels))
        targets = sharpen(targets)
        if multicrop > 0:
            mc_target = 0.5*(targets[:batch_size]+targets[batch_size:])
            targets = tf.concat(
                [targets, *[mc_target for _ in range(multicrop)]],
                axis=0)
        # For numerical stability
        mask = tf.math.less(targets, 1e-4)
        mask = tf.cast(mask, dtype=targets.dtype)
        targets *= mask

        # Step 3: compute cross-entropy loss H(targets, queries)
        loss = tf.reduce_mean(
            tf.reduce_sum(tf.math.log(probs ** (-targets)), axis=1))

        # Step 4: compute me-max regularizer
        rloss = 0.
        if me_max:
            avg_probs = tf.reduce_mean(sharpen(probs), axis=0)
            rloss -= tf.reduce_sum(tf.math.log(avg_probs ** (-avg_probs)))

        return loss, rloss

    return loss