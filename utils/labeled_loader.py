from . import multicrop_loader, config
import tensorflow as tf
import numpy as np

GLOBAL_SCALE = [0.75, 1.0]
AUTO = tf.data.AUTOTUNE


def support_sampler(support_ds):
    """
    Samples indices from the label array with a uniform distribution.

    :param support_ds: TensorFlow dataset (each entry should have (
    image, label) pair)
    :return: a list of datasets for each unique label of CiFAR-10
    """
    ds = []
    for i in np.arange(0, 10):
        ds_label = support_ds.filter(lambda image, label: label == i).repeat()
        ds.append(ds_label)
    return ds


def onehot_encode(labels, label_smoothing=0.1):
    """
    One-hot encode label with label smoothing.

    :param labels: (batch_size, )
    return: one-hot encoded labels with optional label smoothing
    """
    labels = tf.one_hot(labels, depth=10)
    # Reference: https://t.ly/CSYO)
    labels *= 1.0 - label_smoothing
    labels += label_smoothing / labels.shape[1]
    return labels


def get_support_ds(ds, bs, aug=True):
    """
    Prepares TensorFlow dataset with sampling as suggested in:
    https://arxiv.org/abs/2104.13963 (See Appendix C)

    :param ds: TensorFlow dataset (each entry should have (image,
    label) pair)
    :param bs: batch size (int)
    :return: a multi-crop dataset
    """
    # Since at each iteration the support dataset should have equal
    # number
    # of images per class we assign uniform weights for sampling.
    listed_ds = support_sampler(ds)
    balanced_ds = tf.data.experimental.sample_from_datasets(
        listed_ds, [0.1] * 10, seed=42
    )

    # As per Appendix C, for CIFAR10 2x views are needed for making
    # the network better at instance discrimination.
    loaders = tuple()
    for _ in range(config.SUP_VIEWS):
        if aug:
            balanced_ds = balanced_ds.map(
                lambda x, y: (
                    multicrop_loader.random_resize_distort_crop(x, GLOBAL_SCALE, 32),
                    y,
                ),
                num_parallel_calls=AUTO,
                deterministic=True,
            )
        loaders += (balanced_ds,)

    zipped_loaders = tf.data.Dataset.zip(loaders)
    return zipped_loaders
