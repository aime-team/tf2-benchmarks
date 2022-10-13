# TensorFlow 2.x Benchmarks

ImageNet (ResNet50) benchmarks for Tensorflow 2.x


# About ResNet-50 v1.5 model:

Residual neural networks, ResNet for short, were first introduced in 2015 to classify images. ResNet is known to be one of the first deep learning networks solving the vanishing/exploding gradient problem that occurs in previously used perceptron network structures when the number of intermediate layers is increased, see [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385). The characteristic feature of residual networks is the use of "jump connections" between different layers, where certain layers can be skipped. This allows to build much deeper networks and solves the vanishing/exploding gradient problem. The ResNet-50 model used for this experiment consists of 48 convolutional layers, as well as a MaxPool and an Average Pool layer (48+1+1=50 layers). With the deeper network structure, better detection rates are achieved indeed than with the flatter network structures previously used.

A version of the ResNet model pre-trained with the ImageNet dataset can be downloaded from the PyTorch library. However, we used an untrained ResNet50 model because we wanted to investigate the optimization of training with ImageNet.

# ImageNet ILSVRC2012 dataset (download and preprocessing):

The [ImageNet dataset](https://image-net.org/) consists of around 14 million annotated images that have been assigned to 1000 different classes. Since 2010, the dataset has been used in the ImageNet Large Scale Visual Recognition Challenge ([ILSVRC](https://image-net.org/challenges/LSVRC/index.php)) to study image classification and object recognition. Because of its size, quality, and accessibility, the ImageNet dataset is well suited to study the training of models for image classification. It can be downloaded for free from [kaggle}(https://www.kaggle.com/c/imagenet-object-localization-challenge/overview/description) too.
you can visualize the dataset [here](https://knowyourdata-tfds.withgoogle.com/#tab=STATS&dataset=imagenet2012).

After the download, you have 3 files, from which only 2, containing the images for training (138GB) and for validation (6.3GB) will be used:
- ILSVRC2012_devkit_t12.tar.gz
- ILSVRC2012_img_train.tar
- ILSVRC2012_img_val.tar

Additionally the [synset labels](https://image-net.org/challenges/LSVRC/2012/browse-synsets.php), synonym set or categories, are needed. You can find them [here](https://raw.githubusercontent.com/tensorflow/models/master/research/slim/datasets/imagenet_2012_validation_synset_labels.txt) and rename as synset_labels.txt. 

After you extract the content of both files, 1000 tar files, which should be extracted too, are located within the training folder, with the format n_categorie_number.tar (e.g. n09256479.tar), and 50000 JPEG files located in the validation folder.

The next step is to generate the [TFrecords](https://www.tensorflow.org/tutorials/load_data/tfrecord) of the training and validation JPEG files. For that, there is available the script imagenet_to_gcs.py, which can be downloaded using the following command:
```  
wget https://raw.githubusercontent.com/tensorflow/tpu/master/tools/datasets/imagenet_to_gcs.py
```
Take care about the relative path of synset_labels.txt (/Imagenet_folder/synset_labels.txt), training (/Imagenet_folder/train/n09256479/n09256479_1.JPEG) and validation JPEG files (/Imagenet_folder/validation/ILSVRC2012_val_00000001.JPEG).

More information about how the script works and the compulsory location of the different files can be found inside. 

Now you are prepared to generate the TFrecords using:
```
 python imagenet_to_gcs.py 
  --raw_data_dir=/Imagenet_folder
  --local_scratch_dir=/where_you_want_the_tf_records
  --nogcs_upload
```
The result is a folder with 2 subfolders containing 1024 (trainig) and 128 (validation) files, whose format is train-01023-of-01024 and validation-00000-of-00128, respectively and are around 140 MB each. Put together all tfrecord files in a folder, which will be provided as flag later.

## Usage

To start the training, move to the folder where you have cloned the repo and use the following command adapting it for you setup:

```
python tf2-benchmarks.py  --data_dir=/location_of_your_train_and_val_tfrecords --model_dir=/folder_to_save_the_checkpoints --log_dir=/folder_to_save_log_files --num_gpus 1 --batch_size=640 --train_epochs=1 --xla  --synth=False --enable_checkpoint_and_export=True --output_verbosity=2
```
where:
--data_dir: location of the tfrecords (see above)
--model_dir: folder where the checkpoints will be saved
--log_dir: where the log files will be saved
--num_gpus: number of GPUs used for training 
--batch_size: batch size 
--train_epochs: number of epochs
--[xla](https://www.tensorflow.org/xla): enable XLA auto jit compilation 
--synth: False for training
--enable_checkpoint_and_export= 
--output_verbosity= output verbosity level (0:all messages are logged (default behavior), 1: only WARNINGS and ERRORS, 2: only ERRORS, 3: no verbosity)

