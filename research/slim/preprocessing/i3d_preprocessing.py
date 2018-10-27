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
"""Provides utilities to preprocess images for the Inception networks."""

from __future__ import absolute_import

import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def loop_video(images, frame_num, vid_len=50):
    images = tf.concat([tf.tile(images, [vid_len/frame_num, 1, 1, 1]),
                        images[:vid_len - (vid_len/frame_num) * frame_num, :, :, :]], 0)
    return images


def crop_video(images, frame_num, vid_len=50):
    ind = tf.random_uniform([], maxval=frame_num-vid_len+1, dtype=tf.int32)
    return images[ind:ind+vid_len, :, :, :]


def crop_video_fixed(images, frame_num, vid_len=50):
    return images[frame_num/2-vid_len/2:frame_num/2+vid_len/2, :, :, :]


def preprocess_for_train(images, scope=None):
    """Distort one image for training a network.

    Distorting images provides a useful technique for augmenting the data
    set during training in order to make the network invariant to aspects
    of the image that do not effect the label.

    Additionally it would create image_summaries to display the different
    transformations applied to the image.

    Args:
      images: 4-D Tensor of images.
      scope: optional scope for name_scope.
    Returns:
      3-D float Tensor of distorted image used for training with range [-1, 1].
    """
    with tf.name_scope(scope, 'train_image', [images]):

        img_shape = array_ops.shape(images)
        frame_num = img_shape[0]
        channel_num = img_shape[3]

        # Randomly crop images
        resized_images = tf.random_crop(images, [frame_num, 224, 224, channel_num])
        # resized_images = tf.image.resize_bilinear(cropped_images, [224, 224])

        # Crop or loop the video
        final_images = tf.cond(tf.greater(frame_num, 50),
                               lambda: crop_video(resized_images, frame_num),
                               lambda: loop_video(resized_images, frame_num))
        return final_images


def preprocess_for_eval(images, central_fraction=0.875, scope=None):
    """Prepare one image for evaluation.

    If central_fraction is specified it would crop the central fraction of the
    input image.

    Args:
      images: 4-D Tensor of images.
      central_fraction: optional Float, fraction of the image to crop.
      scope: optional scope for name_scope.
    Returns:
      3-D float Tensor of prepared image.
    """
    with tf.name_scope(scope, 'eval_image', [images]):

        # Crop the central region of the image with an area containing 87.5% of
        # the original image.
        img_shape = array_ops.shape(images)

        # Loop the video with too few frames
        frame_num = img_shape[0]
        images = tf.cond(tf.greater(frame_num, 10),
                         true_fn=lambda: images,
                         false_fn=lambda: loop_video(images, frame_num, vid_len=10))

        min_size = tf.minimum(img_shape[1], img_shape[2])

        images = tf.image.resize_image_with_crop_or_pad(images, min_size, min_size)
        images = tf.image.resize_bilinear(images, [256, 256])
        images = tf.map_fn(lambda x: tf.image.central_crop(x, central_fraction=central_fraction),
                           images, dtype=tf.float32)

        return images


def preprocess_rgb(rgb, is_training=False):
    """Pre-process one image for training or evaluation.

    Args:
      rgb: 4-D Tensor [frames, height, width, channels] with RGB images.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.

    Returns:
      4-D float Tensor containing an appropriately scaled images
    """

    rgb = tf.map_fn(lambda x: tf.image.convert_image_dtype(x, dtype=tf.float32),
                    rgb, dtype=tf.float32)
    rgb = tf.stack(rgb)
    # rescale to [-1,1]
    rgb = (rgb - 0.5) * 2.0

    if is_training:
        return preprocess_for_train(rgb)
    else:
        return preprocess_for_eval(rgb)


def preprocess_opt(opt, is_training=False):
    """Pre-process one image for training or evaluation.

    Args:
      opt: 4-D Tensor [frames, height, width, channels] with OPT images.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.

    Returns:
      4-D float Tensor containing an appropriately scaled images
    """

    opt = tf.stack(opt)

    # truncate to [-20, 20]
    opt = tf.minimum(opt, 20.0)
    opt = tf.maximum(opt, -20.0)
    # rescale to [-1, 1]
    opt = opt / 20.0

    if is_training:
        return preprocess_for_train(opt)
    else:
        return preprocess_for_eval(opt)


def preprocess_rgb_opt(rgb, opt, is_training=False):
    """Pre-process one image for training or evaluation.

    Args:
      rgb: 4-D Tensor [frames, height, width, channels] with RGB images.
      opt: 4-D Tensor [frames, height, width, channels] with OPT images.
      is_training: Boolean. If true it would transform an image for train,
        otherwise it would transform it for evaluation.

    Returns:
      4-D float Tensor containing an appropriately scaled images
    """

    opt = tf.stack(opt)

    # truncate to [-20, 20]
    opt = tf.minimum(opt, 20.0)
    opt = tf.maximum(opt, -20.0)
    # rescale to[-1, 1]
    opt = opt / 20.0

    rgb = tf.map_fn(lambda x: tf.image.convert_image_dtype(x, dtype=tf.float32),
                    rgb, dtype=tf.float32)
    rgb = tf.stack(rgb)
    # rescale to [-1,1]
    rgb = (rgb - 0.5) * 2.0

    if is_training:
        images = preprocess_for_train(tf.concat([rgb, opt], -1))
        return images[:, :, :, :3], images[:, :, :, 3:]
    else:
        return preprocess_for_eval(rgb), preprocess_for_eval(opt)
