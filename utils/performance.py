# Lint as: python3
# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Functions and classes related to training performance."""

import tensorflow as tf
from absl import flags

import os
import multiprocessing

FLAGS = flags.FLAGS

def configure_optimizer(optimizer,
                        use_float16=False,
                        use_graph_rewrite=False,
                        loss_scale="dynamic"):
  """Configures optimizer object with performance options."""
  if use_float16:
    # Wraps optimizer with a LossScaleOptimizer. This is done automatically
    # in compile() with the "mixed_float16" policy, but since we do not call
    # compile(), we must wrap the optimizer manually.
    optimizer = (
        tf.keras.mixed_precision.LossScaleOptimizer(
            optimizer, loss_scale=loss_scale))
  if use_graph_rewrite:
    # Note: the model dtype must be 'float32' before doing the graph rewrite
    optimizer = tf.train.enable_mixed_precision_graph_rewrite(optimizer)
  return optimizer


def set_mixed_precision_policy(dtype):
  """Sets mix precision policy."""
  if dtype == tf.float16:
    policy = tf.keras.mixed_precision.set_global_policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)
  elif dtype == tf.bfloat16:
    policy = tf.keras.mixed_precision.set_global_policy('mixed_bfloat16')
    tf.keras.mixed_precision.set_global_policy(policy)
  elif dtype == tf.float32:
     tf.keras.mixed_precision.set_global_policy('float32')
  else:
    raise ValueError("Unexpected dtype: %s" % dtype)


def set_cudnn_batchnorm_mode():
  """Set CuDNN batchnorm mode for better performance.

     Note: Spatial Persistent mode may lead to accuracy losses for certain
     models.
  """
  if FLAGS.batchnorm_spatial_persistent:
    os.environ['TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT'] = '1'
  else:
    os.environ.pop('TF_USE_CUDNN_BATCHNORM_SPATIAL_PERSISTENT', None)


def set_gpu_thread_mode_and_count(gpu_thread_mode,
                                  datasets_num_private_threads,
                                  num_gpus, per_gpu_thread_count):
  """Set GPU thread mode and count, and adjust dataset threads count."""
  cpu_count = multiprocessing.cpu_count()
  logging.info('Logical CPU cores: %s', cpu_count)

  # Allocate private thread pool for each GPU to schedule and launch kernels
  per_gpu_thread_count = per_gpu_thread_count or 2
  os.environ['TF_GPU_THREAD_MODE'] = gpu_thread_mode
  os.environ['TF_GPU_THREAD_COUNT'] = str(per_gpu_thread_count)
  logging.info('TF_GPU_THREAD_COUNT: %s',
               os.environ['TF_GPU_THREAD_COUNT'])
  logging.info('TF_GPU_THREAD_MODE: %s',
               os.environ['TF_GPU_THREAD_MODE'])

  # Limit data preprocessing threadpool to CPU cores minus number of total GPU
  # private threads and memory copy threads.
  total_gpu_thread_count = per_gpu_thread_count * num_gpus
  num_runtime_threads = num_gpus
  if not datasets_num_private_threads:
    datasets_num_private_threads = min(
        cpu_count - total_gpu_thread_count - num_runtime_threads,
        num_gpus * 8)
    logging.info('Set datasets_num_private_threads to %s',
                 datasets_num_private_threads)
