# Tensorflow 2 Benchmarks

ImageNet (ResNet50) benchmarks for Tensorflow 2.9 and later

## Usage

For Tensorflow 2.x float32 benchmarking use:

```
python tf2-benchmarks.py --model resnet50 --xla --batch_size 64 --num_gpus 1
```

For Tensorflow 2.x float16 (mixed precision) benchmarking use:

```
python tf2-benchmarks.py --model resnet50 --xla --batch_size 128 --dtype fp16 --num_gpus 1
```

## Results

Benchmarks meassured with this scripts are available here:

[AIME Deep Learning Benchmarks 2022](https://www.aime.info/blog/en/deep-learning-gpu-benchmarks-2022/)
