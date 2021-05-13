"""
References:
	* https://github.com/facebookresearch/suncet/blob/master/snn_eval.py
"""

# Imports
from utils import config
import tensorflow as tf
import numpy as np


def snn(query, h_train, h_labs, temp=0.1):
    """
    Soft nearest neighbors similarity classifier (for details,
    refer to Appendix A).

    :param query: query embeddings (batch_size, 128)
    :param h_train: embeddings of the labeled set (batch_size, 128)
    :param h_labs: labels of the labeled set (batch_size, 10)
    :param temp: temperature hyperparameter
    :return: similarity score
    """
    # Normalize embeddings
    query = tf.math.l2_normalize(query, axis=1)
    h_train = tf.math.l2_normalize(h_train, axis=1)

    # Compute similarity
    return tf.nn.softmax(query @ tf.transpose(h_train) / temp, axis=1) @ h_labs


# Constants
AUTO = tf.data.AUTOTUNE

# Load dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# Prepare Dataset object for the test set
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test))
test_ds = test_ds.map(
    lambda x, y: (tf.image.central_crop(x, 0.5), y), num_parallel_calls=AUTO
).batch(config.MULTICROP_BS)

# Prepare Dataset object for the same labeled samples we have been using
# Note - no augmentation except center crop
# (https://github.com/facebookresearch/suncet/blob/master/snn_eval.py#L125)
sampled_idx = np.load(config.SUPPORT_IDX)
sampled_train, sampled_labels = x_train[sampled_idx], y_train[sampled_idx].squeeze()
labeled_ds = tf.data.Dataset.from_tensor_slices((sampled_train, sampled_labels))
labeled_ds = labeled_ds.map(
    lambda x, y: (tf.image.central_crop(x, 0.5), y), num_parallel_calls=AUTO
).batch(16)
print("Data loaders prepared.")

# Load the fine-tuned encoders
wide_resnet_enc = tf.keras.models.load_model(config.FINETUNED_MODEL)
print("Fine-tuned encoder loaded.")

############## Compute embeddings ##############
labeled_embeddings = wide_resnet_enc.predict(labeled_ds)[1]
print("Embeddings computed from the available labeled samples.")
support_labels = []
for _, batch_labels in labeled_ds:
    support_labels.append(batch_labels)

support_labels = tf.concat(support_labels, axis=0)
support_labels = tf.one_hot(support_labels, depth=10)

############## Evaluate embeddings ##############
mean_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
for i, (batch_images, batch_labels) in enumerate(test_ds):
    batch_emb = wide_resnet_enc.predict(batch_images)[1]
    batch_probs = snn(batch_emb, labeled_embeddings, support_labels)

    mean_accuracy.update_state(batch_labels, batch_probs)

    if i % 5 == 0:
        batch_top1_acc = tf.keras.metrics.sparse_categorical_accuracy(
            batch_labels, batch_probs
        )
        batch_acc = batch_top1_acc.numpy().sum() / len(batch_top1_acc)
        print(f"Batch {i} Top-1 Accuracy: {batch_acc * 100:.3f}%")

print("Top-1 accuracy: {:3f}%".format(100 * mean_accuracy.result()))
