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
"""Runs a ResNet model on the ImageNet dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

 
import sys

from absl import app
from absl import flags
from absl import logging 
import os
  
import logging

logging.getLogger('tensorflow').disabled = True
os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(2)  # Selection of the verbosity level

import tensorflow as tf

from utils import performance
from utils.flags import core as flags_core
from utils.logs import logger
from utils import distribution_utils
import common
import imagenet_preprocessing
import resnet_model  


def run(flags_obj):
  
  """Run ResNet ImageNet training and eval loop using native Keras APIs.

  Args:
    flags_obj: An object containing parsed flag values.

  Raises:
    ValueError: If fp16 is passed as it is not currently supported.
    NotImplementedError: If some features are not currently supported.

  Returns:
    Dictionary of training and eval stats.
  """

  if flags_obj.enable_xla:
    print("--- Enable XLA")   
    tf.config.optimizer.set_jit(True)

  # Execute flag override logic for better model performance
  if flags_obj.gpu_thread_private:
    print("--- Enable GPU Private Thread Mode")   
    performance.set_gpu_thread_mode_and_count(
        per_gpu_thread_count=flags_obj.per_gpu_thread_count,
        gpu_thread_mode="gpu_private",
        num_gpus=flags_obj.num_gpus,
        datasets_num_private_threads=flags_obj.datasets_num_private_threads)

  performance.set_cudnn_batchnorm_mode()

  dtype = flags_core.get_tf_dtype(flags_obj)
  performance.set_mixed_precision_policy(
      flags_core.get_tf_dtype(flags_obj),
      flags_core.get_loss_scale(flags_obj, default_for_fp16=128))

  data_format = flags_obj.data_format
  if data_format is None:
    data_format = ('channels_first'
                   if tf.test.is_built_with_cuda() else 'channels_last')
  tf.keras.backend.set_image_data_format(data_format)

  # Configures cluster spec for distribution strategy.
  _ = distribution_utils.configure_cluster(flags_obj.worker_hosts,
                                           flags_obj.task_index)

  strategy = distribution_utils.get_distribution_strategy(
      distribution_strategy=flags_obj.distribution_strategy,
      num_gpus=flags_obj.num_gpus,
      all_reduce_alg=flags_obj.all_reduce_alg,
      num_packs=flags_obj.num_packs)#,
      #tpu_address=flags_obj.tpu)

  if strategy:
    # flags_obj.enable_get_next_as_optional controls whether enabling
    # get_next_as_optional behavior in DistributedIterator. If true, last
    # partial batch can be supported.
    strategy.extended.experimental_enable_get_next_as_optional = (
        flags_obj.enable_get_next_as_optional
    )

  strategy_scope = distribution_utils.get_strategy_scope(strategy)

  # pylint: disable=protected-access
  if flags_obj.use_synthetic_data:
    print("--- Using Synthetic Data")
    distribution_utils.set_up_synthetic_data()
    input_fn = common.get_synth_input_fn(
        height=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        width=imagenet_preprocessing.DEFAULT_IMAGE_SIZE,
        num_channels=imagenet_preprocessing.NUM_CHANNELS,
        num_classes=imagenet_preprocessing.NUM_CLASSES,
        dtype=dtype,
        drop_remainder=True)
  else:
    distribution_utils.undo_set_up_synthetic_data()
    input_fn = imagenet_preprocessing.input_fn

  # When `enable_xla` is True, we always drop the remainder of the batches
  # in the dataset, as XLA-GPU doesn't support dynamic shapes.
  drop_remainder = flags_obj.enable_xla

  # Current resnet_model.resnet50 input format is always channel-last.
  # We use keras_application mobilenet model which input format is depends on
  # the keras beckend image data format.
  # This use_keras_image_data_format flags indicates whether image preprocessor
  # output format should be same as the keras backend image data format or just
  # channel-last format.
  #  use_keras_image_data_format = (flags_obj.model == 'mobilenet')
  use_keras_image_data_format = False
  train_input_dataset = input_fn(
      is_training=True,
      data_dir=flags_obj.data_dir,
      batch_size=flags_obj.batch_size,
      parse_record_fn=imagenet_preprocessing.get_parse_record_fn(
          use_keras_image_data_format=use_keras_image_data_format),
      datasets_num_private_threads=flags_obj.datasets_num_private_threads,
      dtype=dtype,
      drop_remainder=drop_remainder,
      tf_data_experimental_slack=flags_obj.tf_data_experimental_slack,
      training_dataset_cache=flags_obj.training_dataset_cache,
  )

  eval_input_dataset = None
  if not flags_obj.skip_eval:
    eval_input_dataset = input_fn(
        is_training=False,
        data_dir=flags_obj.data_dir,
        batch_size=flags_obj.batch_size,
        parse_record_fn=imagenet_preprocessing.get_parse_record_fn(
            use_keras_image_data_format=use_keras_image_data_format),
        dtype=dtype,
        drop_remainder=drop_remainder)

  lr_schedule = common.PiecewiseConstantDecayWithWarmup(
      batch_size=flags_obj.batch_size,
      epoch_size=imagenet_preprocessing.NUM_IMAGES['train'],
      warmup_epochs=common.LR_SCHEDULE[0][1],
      boundaries=list(p[1] for p in common.LR_SCHEDULE[1:]),
      multipliers=list(p[0] for p in common.LR_SCHEDULE),
      compute_lr_on_cpu=True)
  steps_per_epoch = (
      imagenet_preprocessing.NUM_IMAGES['train'] // flags_obj.batch_size)

  with strategy_scope:
    if flags_obj.optimizer == 'resnet50_default':
      optimizer = common.get_optimizer(lr_schedule)
    elif flags_obj.optimizer == 'mobilenet_default':
      initial_learning_rate = \
          flags_obj.initial_learning_rate_per_sample * flags_obj.batch_size
      optimizer = tf.keras.optimizers.SGD(
          learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
              initial_learning_rate,
              decay_steps=steps_per_epoch * flags_obj.num_epochs_per_decay,
              decay_rate=flags_obj.lr_decay_factor,
              staircase=True),
          momentum=0.9)
    if flags_obj.fp16_implementation == 'graph_rewrite':
      # Note: when flags_obj.fp16_implementation == "graph_rewrite", dtype as
      # determined by flags_core.get_tf_dtype(flags_obj) would be 'float32'
      # which will ensure tf.compat.v2.keras.mixed_precision and
      # tf.train.experimental.enable_mixed_precision_graph_rewrite do not double
      # up.
      optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
          optimizer)

    print("--- Image Data Format: " + tf.keras.backend.image_data_format())
    if tf.keras.backend.image_data_format() == 'channels_first':
      input_shape = [3, 224, 224]
    else:
      input_shape = [224, 224, 3]

    print("--- Batch Size: " + str(flags_obj.batch_size))

    img_input = tf.keras.layers.Input(shape=input_shape, batch_size=flags_obj.batch_size)

    model_name = "unknown" 
    if flags_obj.model == 'resnet50':
      model_name = "ResNet50 (v1.5)"
      model = resnet_model.resnet50v1_5(input_shape=input_shape, num_classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.model == 'resnet50_v1.0':
      model_name = "ResNet50 (v1.0)"
      model = tf.keras.applications.ResNet50(input_shape=input_shape, weights=None, classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.model == 'resnet152':
      model_name = "ResNet152"
      model = tf.keras.applications.ResNet152(input_shape=input_shape, weights=None, classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.model == 'vgg19':
      model_name = "VGG19"
      model = tf.keras.applications.VGG19(input_shape=input_shape, weights=None, classes=imagenet_preprocessing.NUM_CLASSES)
    elif flags_obj.model == 'mobilenet':
      model_name = "MobileNet"
      model = tf.keras.applications.MobileNetV2(input_shape=input_shape, weights=None, classes=imagenet_preprocessing.NUM_CLASSES)
    else:
      sys.exit("! unknown model: " + flags_obj.model)

    print("--- Model: %s" % (model_name))

    if flags_obj.pretrained_filepath:
      model.load_weights(flags_obj.pretrained_filepath)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=optimizer,
        metrics=(['sparse_categorical_accuracy']
                 if flags_obj.report_accuracy_metrics else None),
        run_eagerly=flags_obj.run_eagerly)

  train_epochs = flags_obj.train_epochs

  callbacks = common.get_callbacks(
      steps_per_epoch=steps_per_epoch,
      enable_checkpoint_and_export=flags_obj.enable_checkpoint_and_export,
      model_dir=flags_obj.model_dir)

  # if multiple epochs, ignore the train_steps flag.
  if train_epochs <= 1 and flags_obj.train_steps:
    steps_per_epoch = min(flags_obj.train_steps, steps_per_epoch)
    train_epochs = 1

  num_eval_steps = (
      imagenet_preprocessing.NUM_IMAGES['validation'] // flags_obj.batch_size)

  validation_data = eval_input_dataset
  if flags_obj.skip_eval:
    # Only build the training graph. This reduces memory usage introduced by
    # control flow ops in layers that have different implementations for
    # training and inference (e.g., batch norm).
    num_eval_steps = None
    validation_data = None

  if not strategy and flags_obj.explicit_gpu_placement:
    # TODO(b/135607227): Add device scope automatically in Keras training loop
    # when not using distribution strategy.
    no_dist_strat_device = tf.device('/device:GPU:0')
    no_dist_strat_device.__enter__()
 
  history = model.fit(train_input_dataset,
                      epochs=train_epochs,
                      batch_size=flags_obj.batch_size,
                      steps_per_epoch=steps_per_epoch,
                      callbacks=callbacks,
                      validation_steps=num_eval_steps,
                      validation_data=validation_data,
                      validation_freq=flags_obj.epochs_between_evals,
                      verbose=2)

  eval_output = None
  if not flags_obj.skip_eval:
    eval_output = model.evaluate(eval_input_dataset,
                                 steps=num_eval_steps,
                                 verbose=2)

  if flags_obj.enable_checkpoint_and_export:
    if dtype == tf.bfloat16:
      logging.warning('Keras model.save does not support bfloat16 dtype.')
    else:
      # Keras model.save assumes a float32 input designature.
      export_path = os.path.join(flags_obj.model_dir, 'saved_model')
      model.save(export_path, include_optimizer=False)

  if not strategy and flags_obj.explicit_gpu_placement:
    no_dist_strat_device.__exit__()
    
  stats = common.build_stats(history, eval_output)
  return stats


def define_imagenet_keras_flags():
  common.define_keras_flags(
      model=True,
      optimizer=True,
      pretrained_filepath=True)
  flags_core.set_defaults()
  flags.adopt_module_key_flags(common)


def main(_):
  with logger.benchmark_context(flags.FLAGS):
    stats = run(flags.FLAGS)
  logging.info('\n--- Run stats:\n%s', stats)

if __name__ == '__main__':
  define_imagenet_keras_flags()  
  app.run(main)
