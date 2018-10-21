#!/bin/bash

set -o pipefail

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/projects/ashok/yueguo/20bn/checkpoints/Fusion_LSTM/model.ckpt

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=kinetics

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/projects/ashok/yueguo/kinetics/${MODEL_NAME}_ALL_0_1_adam

# Where the raw dataset is saved to.
DATASET_DIR=/projects/ashok/yueguo/${MODEL_NAME}

# Where the tfRecords dataset is saved to.
DATASET_OUTPUT=/projects/ashok/yueguo/${MODEL_NAME}/${MODEL_NAME}-tfRecords

# Where the results are saved to.
EVAL_DIR=/projects/ashok/yueguo/20bn/results${MODEL_NAME}_ALL

{
# Convert the dataset
python download_and_convert_data.py \
  --dataset_name=${MODEL_NAME} \
  --dataset_output=${DATASET_OUTPUT} \
  --dataset_dir=${DATASET_DIR} &&

# Fine-tune
# python train_image_classifier_all.py \
#   --train_dir=${TRAIN_DIR} \
#   --dataset_name=${MODEL_NAME} \
#   --dataset_split_name=train \
#   --dataset_dir=${DATASET_OUTPUT} \
#   --model_name=${MODEL_NAME} \
#   --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
#   --checkpoint_exclude_scopes=Fusion \
#   --batch_size=6 \
#   --learning_rate=0.001 \
#   --learning_rate_decay_type=fixed \
#   --save_interval_secs=300 \
#   --save_summaries_secs=300 \
#   --log_every_n_steps=50 \
#   --optimizer=adam &&

# Run evaluation.
# python eval_image_classifier_all.py \
#   --checkpoint_path=${TRAIN_DIR} \
#   --eval_dir=${EVAL_DIR} \
#   --dataset_name=${MODEL_NAME} \
#   --dataset_split_name=validation \
#   --dataset_dir=${DATASET_OUTPUT} \
#   --batch_size=1 \
#   --model_name=${MODEL_NAME} &&

echo $(hostname) | mail -s Job_completed yueguo3211@gmail.com

} || {

echo $(hostname) | mail -s Job_failed yueguo3211@gmail.com

}
