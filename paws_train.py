from utils import resnet20, multicrop_loader, labeled_loader, trainer
import tensorflow as tf
import numpy as np
import time

# Constants
MULTICROP_BS = 64
SUPPORT_BS = 160
SUPPORT_SAMPLES = 4000
SUP_VIEWS = 2
LABEL_SMOOTHING = 0.1
EPOCHS = 10
AUTO = tf.data.AUTOTUNE
SAVE_PATH = "paws_encoder"

# Load dataset
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()

# Prepare Dataset object for multicrop
train_ds = tf.data.Dataset.from_tensor_slices(x_train)
multicrop_ds = multicrop_loader.get_multicrop_loader(train_ds)
multicrop_ds = (
    multicrop_ds
    .shuffle(MULTICROP_BS * 100)
    .batch(64)
    .prefetch(AUTO)
)

# Prepare support samples
sampled_idx = np.random.choice(len(x_train), SUPPORT_SAMPLES)
sampled_train, sampled_labels = x_train[sampled_idx],\
								y_train[sampled_idx].squeeze()
sampled_labels = tf.one_hot(sampled_labels, depth=np.unique(sampled_labels))

# Label-smoothing (reference: https://t.ly/CSYO)
sampled_labels *= (1 - LABEL_SMOOTHING)
sampled_labels += (LABEL_SMOOTHING / sampled_labels.shape[1])

# Prepare dataset object for the support samples
support_ds = labeled_loader.get_support_ds(sampled_train,
										   sampled_labels, bs=SUPPORT_BS)

# Initialize encoder and optimizer
resnet20_enc = resnet20.get_network(n=2, hidden_dim=128)
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

# Loss tracker
epoch_losses = []

############## Training ##############
for e in range(EPOCHS):
	print(f"=======Starting epoch: {e}=======")
	batch_wise_losses = []
	start_time = time.time()
	for unsup_imgs in multicrop_ds:
		# Sample support images
		# As per Appendix C, for CIFAR10 2x views are needed for making
		# the network better at instance discrimination
		support_images, support_labels = next(iter(support_ds))
		support_images = tf.concat(
			[support_images for _ in range(SUP_VIEWS)], axis=0)
		support_labels = tf.concat(
			[support_labels for _ in range(SUP_VIEWS)], axis=0)

		# Perform training step
		batch_loss, gradients = trainer.train_step(unsup_imgs,
									(support_images, support_labels),
									resnet20_enc)
		batch_wise_losses.append(batch_loss.numpy())
		# Update the parameters of the encoder
		optimizer.apply_gradients(zip(gradients,
									  resnet20_enc.trainable_variables))

	print(f"Epoch: {e} Loss: {np.mean(batch_wise_losses)}" 
		f"Time elapsed: {time.time()-start_time:.2f} secs")
	print("")
	epoch_losses.append(np.mean(batch_wise_losses))

# Serialize model
resnet20_enc.save(SAVE_PATH)
print(f"Encoder serialized to : {SAVE_PATH}")