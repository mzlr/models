# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Downloads and converts 20BNV2 data to TFRecords of TF-Example protos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import cv2
import json
import numpy as np
import tensorflow as tf

from datasets import dataset_utils

_NUM_PER_SHARDS = 100
LABELMAP_PATH = '/projects/ashok/yueguo/20bnV2/label/something-something-v2-labels.json'
LABEL_PATH ={
    'train': '/projects/ashok/yueguo/20bnV2/label/something-something-v2-train.json',
    'validation': '/projects/ashok/yueguo/20bnV2/label/something-something-v2-validation.json',
    'test': '/projects/ashok/yueguo/20bnV2/label/something-something-v2-test.json',
}
    

def _get_dataset_filename(dataset_output, split_name, shard_id, total_shard_num):
    output_filename = '20bn_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, total_shard_num)
    return os.path.join(dataset_output, output_filename)


def _get_filenames(split_name, LABEL_PATH=LABEL_PATH, LABELMAP_PATH=LABELMAP_PATH):
    with open(LABEL_PATH[split_name]) as f:
        label = json.load(f)
    with open(LABELMAP_PATH) as f:
        labelmap = json.load(f)
    filenames = np.zeros((len(label), 2), dtype=int)
    for i, l in enumerate(label):
        filenames[i, 0] = l['id']
        if split_name == 'test':
            filenames[i, 1] = -1
        else:
            label_str = l['template'].replace('[', '')
            label_str = label_str.replace(']', '')
            filenames[i, 1] = labelmap[label_str]
    return filenames


def _convert_dataset(split_name, filenames, dataset_dir, dataset_output):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'validation'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'validation', 'test']

    total_shard_num = int(math.ceil(filenames.shape[0] / float(_NUM_PER_SHARDS)))

    for shard_id in range(total_shard_num):
        output_filename = _get_dataset_filename(
            dataset_output, split_name, shard_id + 1, total_shard_num)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * _NUM_PER_SHARDS
            end_ndx = min((shard_id + 1) * _NUM_PER_SHARDS, filenames.shape[0])
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting video %d/%d shard %d' % (
                    i + 1, filenames.shape[0], shard_id))
                sys.stdout.flush()

                name_id = str(filenames[i, 0])
                class_id = filenames[i, 1]

                # read the video
                vid_path = os.path.join(dataset_dir, name_id+'.webm')
                vid = cv2.VideoCapture(vid_path)

                if vid.isOpened():
                    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
                    imgs = []
                    while True:
                        ret, frame = vid.read()
                        if ret:
                            imgs.append(cv2.imencode('.jpg', frame)[1].tobytes())
                        else:
                            break
                    vid.release()
                else:
                    raise Exception('Video#{} cannot be opened.'.format(name_id))

                example = dataset_utils.images_to_tfexample(
                    imgs,
                    'jpg',
                    class_id,
                    height,
                    width,
                    name_id)
                tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def run(dataset_dir, dataset_output):
    """Runs the conversion operation.

    Args:
      dataset_dir: The dataset directory where the dataset is stored.
      dataset_output: The directory where the TFRecord dataset is stored.
    """
    if not tf.gfile.Exists(dataset_output):
        tf.gfile.MakeDirs(dataset_output)

    # Convert the training set.
    filenames = _get_filenames('train')
    _convert_dataset('train', filenames, dataset_dir, dataset_output)

    # # Convert the validation set.
    # filenames = _get_filenames('validation')
    # _convert_dataset('validation', filenames, dataset_dir, dataset_output)

    # # Convert the test set.
    # filenames = _get_filenames('test')
    # _convert_dataset('test', filenames, dataset_dir, dataset_output)

    print('\nFinished converting the 20BN dataset!')
