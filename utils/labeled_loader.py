from . import multicrop_loader
import tensorflow as tf
import numpy as np


AUTO = tf.data.AUTOTUNE


@tf.function
def aug_for_labeled(image, label):
    image = tf.image.random_crop(image, (32, 32, 3))
    distorted_image = multicrop_loader.custom_augment(image)
    return distorted_image, label


def support_sampler(sampled_labels, bs):
    """
    Samples indices from the label array with a uniform distribution.

    :param sampled_labels: labels (batch_size, num_classes)
    :param bs: batch size (int)
    :return: sampled indices
    """
    # Since the labels are one-hot encoded we first need to get them
    # in their original form to do the sampling.
    sampled_labels = np.argmax(sampled_labels, axis=1).squeeze()
    idxs = []
    for class_id in np.arange(0, 10):
        subset_labels = sampled_labels[sampled_labels == class_id]
        random_sampled = np.random.choice(len(subset_labels), bs//10)
        idxs.append(random_sampled)
    return np.array(np.concatenate(idxs))


def get_support_ds(sampled_train, sampled_labels, bs=160):
    """
    Prepares TensorFlow dataset with sampling as suggested in:
    https://arxiv.org/abs/2104.13963 (See Appendix C)

    :param sampled_train: images (batch_size, h, w, nb_channels)
    :param sampled_labels: labels (batch_size, num_classes)
    :param bs: batch size (int)
    :return: TensorFlow dataset object
    """
    random_balanced_idx = support_sampler(sampled_labels, bs)
    temp_train, temp_labels = (
        sampled_train[random_balanced_idx],
        sampled_labels[random_balanced_idx],
    )
    support_ds = tf.data.Dataset.from_tensor_slices((temp_train, temp_labels))
    support_ds = (
        support_ds.shuffle(bs * 100)
        .map(aug_for_labeled, num_parallel_calls=AUTO)
        .batch(bs)
    )
    return support_ds
