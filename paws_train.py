# Imports
from utils import multicrop_loader, labeled_loader, paws_trainer, config
from models import resnet20
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

# Constants
AUTO = tf.data.AUTOTUNE

# Load dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Prepare Dataset object for multicrop
train_ds = tf.data.Dataset.from_tensor_slices(x_train)
multicrop_ds = multicrop_loader.get_multicrop_loader(train_ds)
multicrop_ds = multicrop_ds.shuffle(config.MULTICROP_BS * 100).batch(64).prefetch(AUTO)

# Prepare support samples
sampled_idx = np.random.choice(len(x_train), config.SUPPORT_SAMPLES)
sampled_train, sampled_labels = x_train[sampled_idx], y_train[sampled_idx].squeeze()
sampled_labels = tf.one_hot(
    sampled_labels, depth=len(np.unique(sampled_labels))
).numpy()

# Label-smoothing (reference: https://t.ly/CSYO)
sampled_labels *= 1 - config.LABEL_SMOOTHING
sampled_labels += config.LABEL_SMOOTHING / sampled_labels.shape[1]

# Prepare dataset object for the support samples
support_ds = labeled_loader.get_support_ds(sampled_train, sampled_labels, bs=config.SUPPORT_BS)
print("Data loaders prepared.")

# Initialize encoder and optimizer
resnet20_enc = resnet20.get_network(n=2, hidden_dim=128)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
print("Model and optimizer initialized.")

# Loss tracker
epoch_ce_losses = []
epoch_me_losses = []

############## Training ##############
for e in range(config.PRETRAINING_EPOCHS):
    print(f"=======Starting epoch: {e}=======")
    batch_ce_losses = []
    batch_me_losses = []
    start_time = time.time()

    for unsup_imgs in multicrop_ds:
        # Sample support images
        # As per Appendix C, for CIFAR10 2x views are needed for making
        # the network better at instance discrimination.
        support_images, support_labels = next(iter(support_ds))
        support_images = tf.concat([support_images for _ in range(config.SUP_VIEWS)], axis=0)
        support_labels = tf.concat([support_labels for _ in range(config.SUP_VIEWS)], axis=0)

        # Perform training step
        batch_ce_loss, batch_me_loss, gradients = paws_trainer.train_step(
            unsup_imgs, (support_images, support_labels), resnet20_enc
        )
        batch_ce_losses.append(batch_ce_loss.numpy())
        batch_me_losses.append(batch_me_loss.numpy())

        # Update the parameters of the encoder
        optimizer.apply_gradients(zip(gradients, resnet20_enc.trainable_variables))

    print(
        f"Epoch: {e} CE Loss: {np.mean(batch_ce_losses):.2f}"
        f" ME-MAX Loss: {np.mean(batch_me_losses):.2f}"
        f" Time elapsed: {time.time()-start_time:.2f} secs"
    )
    print("")
    epoch_ce_losses.append(np.mean(batch_ce_losses))
    epoch_me_losses.append(np.mean(batch_me_losses))

# Create a plot to see the cross-entropy losses
plt.figure(figsize=(8, 8))
plt.plot(epoch_ce_losses)
plt.title("Cross-entropy losses during pre-training", fontsize=12)
plt.grid()
plt.savefig(config.PRETRAINING_PLOT, dpi=300)

# Serialize model
resnet20_enc.save(config.PRETRAINED_MODEL)
print(f"Encoder serialized to : {config.PRETRAINED_MODEL}")

# Serialize other artifacts
np.save(config.SUPPORT_IDX, sampled_idx)
