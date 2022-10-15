# Tensorflow 2.x Benchmarks

ImageNet (ResNet50) benchmarks for Tensorflow 2.x

## Usage

For Tensorflow 2.x float32 benchmarking use:

```
python tf2-benchmarks.py --model resnet50 --xla --batch_size 64 --num_gpus 1
```

For Tensorflow 2.x float16 (mixed precision) benchmarking use:

```
python tf2-benchmarks.py --model resnet50 --xla --batch_size 128 --dtype fp16 --num_gpus 1
```
