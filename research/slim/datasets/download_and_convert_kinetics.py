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
r"""Downloads and converts Kinetics data to TFRecords of TF-Example protos.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import csv
import cv2
import tensorflow as tf

from datasets import dataset_utils

_NUM_PER_SHARDS = 100


def _get_rows(split, labels, dataset_dir):
    filename = dataset_dir + '/labels/' + split + '.csv'
    rows = []
    trim_format = '%06d'
    with open(filename, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\n')        
        for row in spamreader:
            tmp = row[0].split(',')
            if split == 'test':
                if tmp[4] == 'Downloaded':
                    basename = '%s_%s_%s.mp4' % (tmp[0],
                                                trim_format % int(tmp[1]),
                                                trim_format % int(tmp[2]))
                    rows.append(('test/' + basename, tmp[0], -1))
            else:
                if tmp[6] == 'Downloaded':
                    basename = '%s_%s_%s.mp4' % (tmp[1],
                                                trim_format % int(tmp[2]),
                                                trim_format % int(tmp[3]))
                    rows.append((tmp[0] + '/' + basename, tmp[1], labels.index(tmp[0])))
    return rows

def _get_dataset_filename(dataset_output, split_name, shard_id, total_shard_num):
    output_filename = 'kinetics_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, total_shard_num)
    return os.path.join(dataset_output, output_filename)


def _convert_dataset(split_name, filenames, dataset_dir, dataset_output):
    """Converts the given filenames to a TFRecord dataset.

    Args:
      split_name: The name of the dataset, either 'train' or 'val'.
      filenames: A list of absolute paths to png or jpg images.
      class_names_to_ids: A dictionary from class names (strings) to ids
        (integers).
      dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'val', 'test']

    total_shard_num = int(math.ceil(len(filenames) / float(_NUM_PER_SHARDS)))

    for shard_id in range(total_shard_num):
        output_filename = _get_dataset_filename(
            dataset_output, split_name, shard_id + 1, total_shard_num)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * _NUM_PER_SHARDS
            end_ndx = min((shard_id + 1) * _NUM_PER_SHARDS, len(filenames))
            for i in range(start_ndx, end_ndx):
                sys.stdout.write('\r>> Converting video %d/%d shard %d' % (
                    i + 1, len(filenames), shard_id))
                sys.stdout.flush()

                name_id = filenames[i][1]
                class_id = filenames[i][2]

                # read the video
                vid_path = os.path.join(dataset_dir, filenames[i][0])
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
        
    csv.field_size_limit(sys.maxsize)

    f = open( dataset_dir + '/labels/' + 'labels.txt', "r" )
    labels = []
    for line in f:
        labels.append(line[:-1])

    split = 'train'
    rows = _get_rows(split, labels, dataset_dir)
    _convert_dataset(split, rows, dataset_dir + '/raw_data', dataset_output)

    # split = 'val'
    # rows = _get_rows(split, labels, dataset_dir)
    # _convert_dataset(split, rows, dataset_dir + '/raw_data', dataset_output)

    # split = 'test'
    # rows = _get_rows(split, labels, dataset_dir)
    # _convert_dataset(split, rows, dataset_dir + '/raw_data', dataset_output)

    print('\nFinished converting the Kinetics dataset!')

