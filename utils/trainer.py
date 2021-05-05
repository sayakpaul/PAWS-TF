"""
References:
	* https://github.com/facebookresearch/suncet/blob/master/src/paws_train.py
"""
import losses

import tensorflow as tf
import copy

paws_loss = losses.get_paws_loss(
        multicrop=6,
        tau=0.1,
        T=0.25,
        me_max=True)


def train_step(unsup_images, sup_loader, encoder):
	# Get batch size for unsupervised images
	u_batch_size = tf.shape(unsup_images)[0]

	# Unsupervised imgs (2 views)
	imgs = [u for u in unsup_images[:2]]
	# Unsupervised multicrop img views (6 views)
	mc_imgs = tf.concat([u for u in unsup_images[2:-1]], axis=0)
	# Segregate images and labels from support set
	simgs, labels = sup_loader
	# Concatenate unlabeled images and labeled support images
	imgs = tf.concat(imgs + simgs, axis=0)

	# Pass through the global views (including images from the
	# support set) and multicrop views
	z = encoder(imgs)
	z_mc = encoder(mc_imgs)

	# Determine anchor views / supports and their  corresponding
	# target views/supports (we are not using prediction head)
	h = copy.deepcopy(z)
	target_supports = h[2 * u_batch_size:].detach()
	target_views = h[:2 * u_batch_size].detach()
	target_views = tf.concat([
		target_views[u_batch_size:],
		target_views[:u_batch_size]], axis=0)
	anchor_supports = z[2 * u_batch_size:]
	anchor_views = z[:2 * u_batch_size]
	anchor_views = tf.concat([anchor_views, z_mc], dim=1)

	# Compute paws loss with me-max regularization
	(ploss, me_max) = paws_loss(
		anchor_views=anchor_views,
		anchor_supports=anchor_supports,
		anchor_support_labels=labels,
		target_views=target_views,
		target_supports=target_supports,
		target_support_labels=labels)
	loss = ploss + me_max
	return loss

