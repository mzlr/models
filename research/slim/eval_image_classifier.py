# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic evaluation script that evaluates a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import tensorflow as tf

from datasets import dataset_factory
from nets import nets_factory
from preprocessing import preprocessing_factory

from nets import i3d
from nets import i3d_last
from nets import lstm
from preprocessing import i3d_preprocessing

slim = tf.contrib.slim

tf.app.flags.DEFINE_integer(
    'batch_size', 100, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'max_num_batches', None,
    'Max number of batches to evaluate by default use all.')

tf.app.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.app.flags.DEFINE_string(
    'checkpoint_path', '/tmp/tfmodel/',
    'The directory where the model was written to or an absolute path to a '
    'checkpoint file.')

tf.app.flags.DEFINE_string(
    'eval_dir', '/tmp/tfmodel/', 'Directory where the results are saved to.')

tf.app.flags.DEFINE_integer(
    'num_preprocessing_threads', 4,
    'The number of threads used to create the batches.')

tf.app.flags.DEFINE_string(
    'dataset_name', 'imagenet', 'The name of the dataset to load.')

tf.app.flags.DEFINE_string(
    'dataset_split_name', 'test', 'The name of the train/test split.')

tf.app.flags.DEFINE_string(
    'dataset_dir', None, 'The directory where the dataset files are stored.')

tf.app.flags.DEFINE_integer(
    'labels_offset', 0,
    'An offset for the labels in the dataset. This flag is primarily used to '
    'evaluate the VGG and ResNet architectures which do not use a background '
    'class for the ImageNet dataset.')

tf.app.flags.DEFINE_string(
    'model_name', 'inception_v3', 'The name of the architecture to evaluate.')

tf.app.flags.DEFINE_string(
    'preprocessing_name', None, 'The name of the preprocessing to use. If left '
    'as `None`, then the model_name flag is used.')

tf.app.flags.DEFINE_float(
    'moving_average_decay', None,
    'The decay to use for the moving average.'
    'If left as None, then moving averages are not used.')

tf.app.flags.DEFINE_integer(
    'eval_image_size', None, 'Eval image size')

tf.app.flags.DEFINE_bool(
    'quantize', False, 'whether to use quantized graph or not.')

FLAGS = tf.app.flags.FLAGS


def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    # tf_global_step = slim.get_or_create_global_step()

    ######################
    # Select the dataset #
    ######################
    dataset = dataset_factory.get_dataset(
        FLAGS.dataset_name, FLAGS.dataset_split_name, FLAGS.dataset_dir)

    ####################
    # Select the model #
    ####################
    network_fn_rgb = i3d.InceptionI3d(
      dataset.num_classes, spatial_squeeze=True,
      final_endpoint='Mixed_5c', name='RGB/inception_i3d')

    network_fn_opt = i3d.InceptionI3d(
      dataset.num_classes, spatial_squeeze=True,
      final_endpoint='Mixed_5c', name='OPT/inception_i3d')

    network_fn_fusion = i3d_last.InceptionI3d(
      dataset.num_classes, spatial_squeeze=True,
      name='Fusion')

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=20 * FLAGS.batch_size,
        common_queue_min=FLAGS.batch_size)

    [rgb, opt, label, height, width] = provider.get(
      ['RGB', 'OPT', 'label', 'height', 'width'])
    opt = tf.map_fn(lambda x: tf.reshape(x, tf.stack([height, width, 2])),
                    opt, dtype=tf.float32)

    label -= FLAGS.labels_offset

    #####################################
    # Select the preprocessing function #
    #####################################
    rgb, opt = i3d_preprocessing.preprocess_rgb_opt(rgb, opt,
                                                    is_training=False)

    rgbs, opts, labels = tf.train.batch(
        [rgb, opt, label],
        shapes=[[None, 224, 224, 3], [None, 224, 224, 2], []],
        dynamic_pad=True,
        batch_size=FLAGS.batch_size,
        num_threads=FLAGS.num_preprocessing_threads,
        capacity=20 * FLAGS.batch_size)

    ####################
    # Define the model #
    ####################
    logits_rgb, _ = network_fn_rgb(
      rgbs, is_training=False)

    logits_opt, _ = network_fn_opt(
      opts, is_training=False)

    logits_lstm, logits = lstm.lstm(logits_rgb, logits_opt, is_training=False)
    logits_lstm = tf.expand_dims(logits_lstm, axis=0)
    logits_lstm.set_shape([1, dataset.num_classes])
    logits_lstm_sm = tf.nn.softmax(-1 * logits_lstm)

    logits_fused, _ = network_fn_fusion(logits, is_training=False)
    logits_fused_sm = tf.nn.softmax(logits_fused)

    variables_to_restore = slim.get_variables_to_restore()
    # if FLAGS.moving_average_decay:
    #   variable_averages = tf.train.ExponentialMovingAverage(
    #       FLAGS.moving_average_decay, tf_global_step)
    #   variables_to_restore = variable_averages.variables_to_restore(
    #       slim.get_model_variables())
    #   variables_to_restore[tf_global_step.op.name] = tf_global_step
    # else:
    #   variables_to_restore = slim.get_variables_to_restore()

    predictions_c = tf.argmax(logits_fused, axis=1)
    predictions_l = tf.argmin(logits_lstm, axis=1)
    predictions_f = tf.argmax(logits_lstm_sm + logits_fused_sm, axis=1)
    logits, _ = network_fn(images)

    if FLAGS.quantize:
      tf.contrib.quantize.create_eval_graph()

    if FLAGS.moving_average_decay:
      variable_averages = tf.train.ExponentialMovingAverage(
          FLAGS.moving_average_decay, tf_global_step)
      variables_to_restore = variable_averages.variables_to_restore(
          slim.get_model_variables())
      variables_to_restore[tf_global_step.op.name] = tf_global_step
    else:
      variables_to_restore = slim.get_variables_to_restore()

    predictions = tf.argmax(logits, 1)
    labels = tf.squeeze(labels)

    # Define the metrics:
    names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
      'Accuracy_Video': slim.metrics.streaming_accuracy(predictions_c, labels),
      'Accuracy_LSTM': slim.metrics.streaming_accuracy(predictions_l, labels),
      'Accuracy_Fused': slim.metrics.streaming_accuracy(predictions_f, labels),
      # 'Recall_5': slim.metrics.streaming_recall_at_k(
      #     logits_fused, labels, 5),
      # 'LSTM': slim.metrics.streaming_concat(logits_lstm),
      # 'Fused': slim.metrics.streaming_concat(logits_fused),
      # 'Labels': slim.metrics.streaming_concat(labels),
    })

    # Print the summaries to screen.
    for name, value in names_to_values.items():
      summary_name = 'eval/%s' % name
      if name == 'Accuracy' or name == 'Recall_5':
        op = tf.summary.scalar(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
      else:
        op = tf.summary.tensor_summary(summary_name, value, collections=[])
        op = tf.Print(op, [value], summary_name)
      tf.add_to_collection(tf.GraphKeys.SUMMARIES, op)

    # TODO(sguada) use num_epochs=1
    if FLAGS.max_num_batches:
      num_batches = FLAGS.max_num_batches
    else:
      # This ensures that we make a single pass over all of the data.
      num_batches = math.ceil(dataset.num_samples / float(FLAGS.batch_size))

    if tf.gfile.IsDirectory(FLAGS.checkpoint_path):
      checkpoint_path = tf.train.latest_checkpoint(FLAGS.checkpoint_path)
    else:
      checkpoint_path = FLAGS.checkpoint_path

    tf.logging.info('Evaluating %s' % checkpoint_path)

    slim.evaluation.evaluate_once(
        master=FLAGS.master,
        checkpoint_path=checkpoint_path,
        logdir=FLAGS.eval_dir,
        num_evals=num_batches + 1,
        eval_op=list(names_to_updates.values()),
        session_config=tf.ConfigProto(allow_soft_placement=True),
        variables_to_restore=variables_to_restore)


if __name__ == '__main__':
  tf.app.run()
