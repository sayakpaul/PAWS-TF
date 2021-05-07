############ Dataset ############
MULTICROP_BS = 64
SUPPORT_BS = 160
SUPPORT_SAMPLES = 4000
SUP_VIEWS = 2
SUPPORT_IDX = "random_idx.npy"

############ Pre-training ############
LABEL_SMOOTHING = 0.1
EPOCHS = 10
PRETRAINING_PLOT = "pretraining_ce_loss.png"
PRETRAINED_MODEL = "paws_encoder"