# Imports
from utils import labeled_loader, suncet_fine_tune, config
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

# Constants
AUTO = tf.data.AUTOTUNE

# Load dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Load the random indices prepared during pre-training
sampled_idx = np.load(config.SUPPORT_IDX)

# Prepare support samples
sampled_train, sampled_labels = x_train[sampled_idx], y_train[sampled_idx].squeeze()
sampled_labels = tf.one_hot(
    sampled_labels, depth=len(np.unique(sampled_labels))
).numpy()

# Note: no label-smoothing (https://github.com/facebookresearch/suncet/blob/master/configs/paws/cifar10_snn.yaml#L10)

# Prepare dataset object for the support samples
support_ds = labeled_loader.get_support_ds(aug=False, bs=config.SUPPORT_BS)
print("Data loaders prepared.")

# Initialize encoder and optimizer
resnet20_enc = tf.keras.models.load_model(config.PRETRAINED_MODEL)
for layer in resnet20_enc.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
print("Model and optimizer initialized.")

# Loss tracker
epoch_suncet_losses = []

############## Training ##############
for e in range(config.FINETUNING_EPOCHS):
    print(f"=======Starting epoch: {e}=======")
    batch_suncet_losses = []
    start_time = time.time()

    for images, labels in support_ds:
        # As per Appendix C, for CIFAR10 2x views are needed for making
        # the network better at instance discrimination.
        support_images = tf.concat([images for _ in range(config.SUP_VIEWS)], axis=0)
        support_labels = tf.concat([labels for _ in range(config.SUP_VIEWS)], axis=0)

        # Perform training step
        batch_suncet_loss, gradients = suncet_fine_tune.train_step(
            (support_images, support_labels), resnet20_enc
        )
        batch_suncet_losses.append(batch_suncet_loss.numpy())

        # Update the parameters of the encoder
        optimizer.apply_gradients(zip(gradients, resnet20_enc.trainable_variables))

    print(
        f"Epoch: {e} SUNCET Loss: {np.mean(batch_suncet_losses):.2f}"
        f" Time elapsed: {time.time()-start_time:.2f} secs"
    )
    print("")
    epoch_suncet_losses.append(np.mean(batch_suncet_losses))

# Serialize model
resnet20_enc.save(config.FINETUNED_MODEL)
print(f"Encoder serialized to : {config.FINETUNED_MODEL}")
