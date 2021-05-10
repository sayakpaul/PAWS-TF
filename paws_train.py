# Imports
from utils import (
    multicrop_loader,
    labeled_loader,
    paws_trainer,
    config,
    lr_scheduler,
    lars_optimizer,
)
from models import resnet20, wide_resnet
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time


# Load dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Constants
AUTO = tf.data.AUTOTUNE
STEPS_PER_EPOCH = int(len(x_train) // config.MULTICROP_BS)
WARMUP_EPOCHS = int(config.PRETRAINING_EPOCHS * 0.1)
WARMUP_STEPS = int(WARMUP_EPOCHS * STEPS_PER_EPOCH)

# Prepare Dataset object for multicrop
train_ds = tf.data.Dataset.from_tensor_slices(x_train)
multicrop_ds = multicrop_loader.get_multicrop_loader(train_ds)
multicrop_ds = (
    multicrop_ds.shuffle(config.MULTICROP_BS * 10)
    .batch(config.MULTICROP_BS)
    .prefetch(AUTO)
)

# Prepare support samples
sampled_idx = np.random.choice(len(x_train), config.SUPPORT_SAMPLES)
sampled_train, sampled_labels = x_train[sampled_idx], y_train[sampled_idx].squeeze()
initial_supp_ds = tf.data.Dataset.from_tensor_slices((sampled_train, sampled_labels))

# Prepare dataset object for the support samples
support_ds = labeled_loader.get_support_ds(initial_supp_ds, bs=config.SUPPORT_BS)
support_ds = (
    support_ds.shuffle(config.SUPPORT_BS * 5).batch(config.SUPPORT_BS).prefetch(AUTO)
)
print("Data loaders prepared.")

# Initialize encoder and optimizer
wide_resnet_enc = wide_resnet.get_network()
scheduled_lrs = lr_scheduler.WarmUpCosine(
    learning_rate_base=config.WARMUP_LR,
    total_steps=config.PRETRAINING_EPOCHS * STEPS_PER_EPOCH,
    warmup_learning_rate=config.START_LR,
    warmup_steps=WARMUP_STEPS,
)
optimizer = lars_optimizer.LARS(
    learning_rate=scheduled_lrs,
    momentum=0.9,
    exclude_from_weight_decay=["batch_normalization", "bias"],
    exclude_from_layer_adaptation=["batch_normalization", "bias"],
)
print("Model and optimizer initialized.")

# Loss trackers
epoch_ce_losses = []
epoch_me_losses = []

############## Training ##############
for e in range(config.PRETRAINING_EPOCHS):
    print(f"=======Starting epoch: {e}=======")
    start_time = time.time()
    epoch_ce_loss_avg = tf.keras.metrics.Mean()
    epoch_me_loss_avg = tf.keras.metrics.Mean()

    for i, unsup_imgs in enumerate(multicrop_ds):
        # Sample support images, concat the images and labels, and
        # then apply label-smoothing.
        support_images_one, support_images_two = next(iter(support_ds))
        support_images = tf.concat(
            [support_images_one[0], support_images_two[0]], axis=0
        )
        support_labels = tf.concat(
            [support_images_one[1], support_images_two[1]], axis=0
        )
        support_labels = labeled_loader.onehot_encode(
            support_labels, config.LABEL_SMOOTHING
        )

        # Perform training step
        batch_ce_loss, batch_me_loss, gradients = paws_trainer.train_step(
            unsup_imgs, (support_images, support_labels), wide_resnet_enc
        )
        epoch_ce_loss_avg.update_state(batch_ce_loss)
        epoch_me_loss_avg.update_state(batch_me_loss)

        # Update the parameters of the encoder
        optimizer.apply_gradients(zip(gradients, wide_resnet_enc.trainable_variables))

    print(
        f"Epoch: {e} CE Loss: {epoch_ce_loss_avg.result():.3f}"
        f" ME-MAX Loss: {epoch_me_loss_avg.result():.3f}"
        f" Time elapsed: {time.time()-start_time:.2f} secs"
    )
    print("")
    epoch_ce_losses.append(epoch_ce_loss_avg.result())
    epoch_me_losses.append(epoch_me_loss_avg.result())

# Create a plot to see the cross-entropy losses
plt.figure(figsize=(8, 8))
plt.plot(epoch_ce_losses)
plt.title("PAWS Training Cross-Entropy Loss", fontsize=12)
plt.grid()
plt.savefig(config.PRETRAINING_PLOT, dpi=300)

# Serialize model
wide_resnet_enc.save(config.PRETRAINED_MODEL)
print(f"Encoder serialized to : {config.PRETRAINED_MODEL}")

# Serialize other artifacts
np.save(config.SUPPORT_IDX, sampled_idx)
