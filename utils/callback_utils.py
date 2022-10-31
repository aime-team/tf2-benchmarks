# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
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
"""Callback util functions"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from absl import logging
import tensorflow.compat.v2 as tf

class BenchmarkCallbacks(tf.keras.callbacks.Callback):
  """Callback for Keras models."""

  def __init__(self, batch_size, log_steps, mean_img_per_sec_file_dest):
    """Callback for logging performance.

    Args:
      batch_size: Total batch size.
      log_steps: Interval of steps between logging of batch level stats.
    """
    self.batch_size = batch_size
    super(BenchmarkCallbacks, self).__init__()
    self.log_steps = log_steps
    self.mean_img_per_sec_file_dest = mean_img_per_sec_file_dest
    self.last_log_step = 0
    self.steps_before_epoch = 0
    self.steps_in_epoch = 0
    self.start_time = None
    self.examples_per_second_list = []

  @property
  def global_steps(self):
    """The current 1-indexed global step."""
    return self.steps_before_epoch + self.steps_in_epoch

  def on_train_end(self, logs=None):
    self.train_finish_time = time.time()

  def on_epoch_begin(self, epoch, logs=None):
    print("-- Training Epoch: %i" % (epoch))
    self.epoch_start = time.time()

  def on_batch_begin(self, batch, logs=None):
    if not self.start_time:
      self.start_time = time.time()

  def on_batch_end(self, batch, logs=None):
    """Records elapse time of the batch and calculates examples per second."""
    self.steps_in_epoch = batch + 1
    steps_since_last_log = self.global_steps - self.last_log_step
    if steps_since_last_log >= self.log_steps:
      now = time.time()
      elapsed_time = now - self.start_time
      steps_per_second = steps_since_last_log / elapsed_time
      examples_per_second = steps_per_second * self.batch_size
      self.examples_per_second_list.append(examples_per_second)
    
      print(
          'Step %d, Images per second: %.1f, Loss: %0.3f' % (self.global_steps, examples_per_second, logs['loss']
          ))
      
      self.last_log_step = self.global_steps
      self.start_time = None

  def on_epoch_end(self, epoch, logs=None):
    epoch_run_time = time.time() - self.epoch_start
    print("\n-- Epoch Runtime: %.1fs\n" % (epoch_run_time))

    self.steps_before_epoch += self.steps_in_epoch
    self.steps_in_epoch = 0
    del self.examples_per_second_list[0]
    mean_img_per_sec = sum(self.examples_per_second_list)/len(self.examples_per_second_list)
    print('Mean images per second: ', mean_img_per_sec)
    if self.mean_img_per_sec_file_dest:
       with open(self.mean_img_per_sec_file_dest, 'w') as log_file:
            log_file.write(f'Mean images per sec: {mean_img_per_sec}, Batch size: {self.batch_size}')


