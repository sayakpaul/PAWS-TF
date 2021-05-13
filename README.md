# PAWS-TF 🐾
TensorFlow 2 implementation of [Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples](https://arxiv.org/abs/2104.13963)
 (PAWS) in TensorFlow. 

PAWS introduces a simple way to combine a very small fraction of labeled data with a comparatively larger corpus of unlabeled data _during_ pre-training. With its approach it sets the state-of-the-art in semi-supervised learning beating methods like [SimCLRV2](https://arxiv.org/abs/2006.10029), [Meta Pseudo Labels](https://arxiv.org/abs/2003.10580) that too with fewer parameters and a smaller pre-training schedule. For details, I recommend checking out the original paper as well as [this blog post](https://ai.facebook.com/blog/dino-paws-computer-vision-with-self-supervised-transformers-and-10x-more-efficient-training/) by the authors. 

This repository implements and includes all the major bits proposed in PAWS in TensorFlow. The current code works with CIFAR10. The only major difference is that the pre-training and subsequent fine-tuning weren't run for the original number of epochs (600 and 30 respectively) to save compute. I have reused the utility components for PAWS loss from the [original implementation](https://github.com/facebookresearch/suncet/).

## Features ✨

- [x] Multi-crop augmentation strategy (originally introduced in [SwAV](https://arxiv.org/abs/2006.09882))
- [x] Class stratified sampler (common in few-shot classification problems)
- [x] WarmUpCosine learning rate schedule (which is typical for self-supervised and semi-supervised pre-training schedules)
- [x] LARS optimizer (comes from [TensorFlow Model Garden](https://github.com/tensorflow/models/blob/master/official/modeling/optimization/lars_optimizer.py))

The trunk portion (all, except the last classification layer) of a WideResNet-28-2 is used inside the encoder for CIFAR10. All the experimental configurations were followed from the Appendix C of the paper. 

## Setup and code structure 💻

A GCP VM ([`n1-standard-8`](https://cloud.google.com/compute/docs/machine-types)) with a single V100 GPU was used for executing the code. 

* `paws_train.py` runs the pre-training as introduced in PAWS.
* `fine_tune.py` runs the fine-tuning part as suggested in Appendix C. Note that this is only required for CIFAR10.
* `nn_eval.py` runs the soft nearest neighbor classification on CIFAR10 test set.

Pre-training and fine-tuning total take **1.4 hours** to complete. All the logs are available in [`misc/logs.txt`](https://github.com/sayakpaul/PAWS-TF/blob/main/misc/logs.txt). Additionally, the indices that were used to sample the labeled examples from the CIFAR10 training set are available [here](https://github.com/sayakpaul/PAWS-TF/blob/main/misc/random_idx.npy).

## Results 📊

PAWS minimizes the cross-entropy loss (as well as maximizes mean-entropy) during pre-training. This is what the training plot indicates too:

<div align="center">
<img src="https://i.ibb.co/4MwCb9J/pretraining-ce-loss.png" width=450/>
</div>

To evaluate the effectivity of the pre-training, PAWS performs soft nearest neighbor classification to report the top-1 accuracy score on a given test set.

This repository gets to **73.46%** top-1 accuracy on the CIFAR10 test set. Again, **note** that I only pre-trained for 50 epochs (as opposed to 600) and fine-tuned for 10 epochs (as opposed to 30). With the original schedule this score should be around 96.0%. 

In the following PCA projection plot, we see that the embeddings of images (computed after fine-tuning) of PAWS are starting to be well separated:

<div align="center">
<img src="https://i.ibb.co/y0XB6pL/projections-viz.png" width=450/>
</div>

## Notebooks 📘

There are two Colab Notebooks:

* [`colabs/data_prep.ipynb`](https://github.com/sayakpaul/PAWS-TF/blob/main/colabs/data_prep.ipynb): It walks through the process of constructing a multi-crop dataset with CIFAR10.
* [`colabs/visualization_paws_projections.ipynb`](https://github.com/sayakpaul/PAWS-TF/blob/main/colabs/visualization_paws_projections.ipynb): Visualizes the PCA projections of pre-computed embeddings. 

## Misc ⺟

* Model weights are available [here](https://github.com/sayakpaul/PAWS-TF/releases/tag/v1.0.0) for reproducibility. 
* With mixed-precision training, the performance can further be improved. I am open to accepting contributions that would implement mixed-precision training in the current code.

## Acknowledgements

* Huge amount of thanks to Mahmoud Assran (first author of PAWS) for patiently resolving my doubts.
* [ML-GDE program](https://developers.google.com/programs/experts/) for providing GCP credit support. 

## Paper Citation

```
@misc{assran2021semisupervised,
      title={Semi-Supervised Learning of Visual Features by Non-Parametrically Predicting View Assignments with Support Samples}, 
      author={Mahmoud Assran and Mathilde Caron and Ishan Misra and Piotr Bojanowski and Armand Joulin and Nicolas Ballas and Michael Rabbat},
      year={2021},
      eprint={2104.13963},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```