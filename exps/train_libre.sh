#!/bin/bash

export DATA_DIR="./sam_dataset/sam_img"
export SAVE_DIR="./save_dir"

if [ -z $1 ]; then
  echo "Usage: bash train.sh <exp group>"
  exit 1
fi

if [ -z $DATA_DIR ]; then
  echo "DATA_DIR not set, source env.sh?"
  exit 1
fi

result=($(split_string_with_slash "$1"))
EXP_GROUP=${result[0]}
EXP_ID=${result[1]}
#echo "a: '$a'"  # Output: a: 'test'
#echo "b: '$b'"  # Output: b: ''
if [[ $EXP_ID == "" ]]; then
  EXP_ID=`date +%s`
fi

EXP_NAME=${EXP_GROUP}/${EXP_ID}
OUTPUT_DIR=${SAVE_DIR}/${EXP_NAME}

echo "-------------------------------"
echo "Exp group        :  " $EXP_GROUP
echo "Exp id           :  " $EXP_ID
echo "Experiment name  :  " $EXP_NAME
echo "Output directory :  " $OUTPUT_DIR
echo "Data dir         :  " $DATA_DIR
echo "-------------------------------"

cd ..


VALID_STEPS=1000
CKPT_STEPS=1000

YOLO_VERBOSE=0 python train_flux.py \
  --offload_gpu 0 \
  --task                          "AB" \
  --use_8bit_adam \
  --mixed_precision                bf16 \
  --quantize                       \
  --pretrained_model_name_or_path "jimmycarter/LibreFlux-SimpleTuner" \
  --resume_from_checkpoint        "latest" \
  --output_dir                    ${OUTPUT_DIR} \
  --dataset_dir                   ${DATA_DIR} \
  --dataset_py_file               "dataset.py" \
  --train_batch_size              1 \
  --gradient_accumulation_steps   1 \
  --checkpointing_steps           $CKPT_STEPS \
  --logging_steps                 1000 \
  --validation_steps              $VALID_STEPS \
  --learning_rate                 1e-5 \
  --max_sequence_length           512 \
  --weighting_scheme              "logit_normal" \
  --validation_path               "validation_images_cc" \
  --validation_resolution         512 \
  --resolution                    512 \
  --guidance_scale                4.0 \
  --num_workers                   1 \
  --num_train_epochs              100
