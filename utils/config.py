## Reference: https://github.com/facebookresearch/suncet/blob/master/configs/paws/cifar10_train.yaml

############ Dataset ############
MULTICROP_BS = 256
SUPPORT_BS = 640
SUPPORT_SAMPLES = 4000
SUP_VIEWS = 2
SUPPORT_IDX = "random_idx.npy"

############ Pre-training ############
LABEL_SMOOTHING = 0.1
PRETRAINING_EPOCHS = 50
START_LR = 0.8
WARMUP_LR = 3.2
PRETRAINING_PLOT = "pretraining_ce_loss.png"
PRETRAINED_MODEL = "paws_encoder"

############ Fine-tuning ############
FINETUNING_EPOCHS = 10
FINETUNED_MODEL = "paws_finetuned"
