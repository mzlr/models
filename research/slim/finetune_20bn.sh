#!/bin/bash

set -o pipefail

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
PRETRAINED_CHECKPOINT_DIR=/projects/ashok/yueguo/kinetics_400_rgb_imgnet_flownets_ckpt/model.ckpt

# Where the pre-trained Inception Resnet V2 checkpoint is saved to.
MODEL_NAME=20bn

# Where the training (fine-tuned) checkpoint and logs will be saved to.
TRAIN_DIR=/projects/ashok/yueguo/20bnV2/${MODEL_NAME}_rgb

# Where the raw dataset is saved to.
DATASET_DIR=/projects/ashok/yueguo/20bnV2/raw/20bn-something-something-v2

# Where the tfRecords dataset is saved to.
DATASET_OUTPUT=/projects/ashok/yueguo/20bnV2/${MODEL_NAME}-tfRecords

# Where the results are saved to.
EVAL_DIR=/projects/ashok/yueguo/20bnV2/results_${MODEL_NAME}_rgb

{
# Convert the dataset
#python download_and_convert_data.py \
#  --dataset_name=${MODEL_NAME} \
#  --dataset_output=${DATASET_OUTPUT} \
#  --dataset_dir=${DATASET_DIR} &&

# Fine-tune
python train_image_classifier.py \
    --train_dir=${TRAIN_DIR} \
    --dataset_name=${MODEL_NAME} \
    --dataset_split_name=train \
    --dataset_dir=${DATASET_OUTPUT} \
    --model_name=${MODEL_NAME} \
    --checkpoint_path=${PRETRAINED_CHECKPOINT_DIR} \
    --checkpoint_exclude_scopes=inception_i3d/Logits \
    --batch_size=10 \
    --learning_rate=0.01 \
    --learning_rate_decay_type=fixed \
    --save_interval_secs=300 \
    --save_summaries_secs=300 \
    --log_every_n_steps=50 \
    --optimizer=sgd &&

# Run evaluation.
 python eval_image_classifier.py \
   --checkpoint_path=${TRAIN_DIR} \
   --eval_dir=${EVAL_DIR} \
   --dataset_name=${MODEL_NAME} \
   --dataset_split_name=validation \
   --dataset_dir=${DATASET_OUTPUT} \
   --batch_size=1 \
   --model_name=${MODEL_NAME} | tail -n 50 | tee -a ${EVAL_DIR}/log.txt &&

echo $(hostname) | mail -s Job_completed yueguo3211@gmail.com

} || {

echo $(hostname) | mail -s Job_failed yueguo3211@gmail.com

}