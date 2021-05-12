# Imports
from utils import labeled_loader, suncet_fine_tune, config
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time

# Constants
STEPS_PER_EPOCH = int(config.SUPPORT_SAMPLES // config.SUPPORT_BS)
TOTAL_STEPS = config.FINETUNING_EPOCHS * STEPS_PER_EPOCH

# Prepare dataset object for the support samples
# Note - no augmentation
support_ds = labeled_loader.get_support_ds(aug=False, bs=config.SUPPORT_BS)
print("Data loaders prepared.")

# Initialize encoder and optimizer
wide_resnet_enc = tf.keras.models.load_model(config.PRETRAINED_MODEL)
for layer in wide_resnet_enc.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False
    else:
        layer.trainable = True

scheduled_lr = tf.keras.experimental.CosineDecay(
    initial_learning_rate=0.1, decay_steps=TOTAL_STEPS
)
optimizer = tf.keras.optimizers.SGD(learning_rate=scheduled_lr, momentum=0.9)
print("Model and optimizer initialized.")

############## Training ##############
for e in range(config.FINETUNING_EPOCHS):
    print(f"=======Starting epoch: {e}=======")
    epoch_suncet_loss_avg = tf.keras.metrics.Mean()
    start_time = time.time()

    for i, (set_one, set_two) in enumerate(support_ds):
        if i == STEPS_PER_EPOCH:
            break

        #  Concat the 2x views from the support set.
        support_images = tf.concat([set_one[0], set_two[0]], axis=0)
        support_labels = tf.concat([set_one[1], set_two[1]], axis=0)
        # Note: no label-smoothing: https://git.io/Jskgu
        support_labels = tf.one_hot(support_labels, depth=10)

        # Perform training step
        batch_suncet_loss, gradients = suncet_fine_tune.train_step(
            (support_images, support_labels), wide_resnet_enc
        )

        # Update the parameters of the encoder
        optimizer.apply_gradients(zip(gradients, wide_resnet_enc.trainable_variables))
        epoch_suncet_loss_avg.update_state(batch_suncet_loss)

    print(
        f"Epoch: {e} SUNCET Loss: "
        f"{epoch_suncet_loss_avg.result():.3f}"
        f" Time elapsed: {time.time() - start_time:.2f} secs"
    )
    print("")

# Serialize model
wide_resnet_enc.save(config.FINETUNED_MODEL)
print(f"Encoder serialized to : {config.FINETUNED_MODEL}")
