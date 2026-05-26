# CycleGAN Training/Test Tips

## Dataset

Use unpaired folders:

```text
trainA/
trainB/
testA/
testB/
```

Run training with:

```bash
python train.py --dataroot ./datasets/your_dataset --name your_experiment --model cycle_gan
```

## Resume Training

Use:

```bash
python train.py --dataroot ./datasets/your_dataset --name your_experiment --model cycle_gan --continue_train
```

Set `--epoch_count` if you need a custom starting epoch number.

## Testing

Full CycleGAN test:

```bash
python test.py --dataroot ./datasets/your_dataset --name your_experiment --model cycle_gan
```

Single-direction test:

```bash
python test.py --dataroot ./datasets/your_dataset/testA --name your_experiment --model test --model_suffix _A --no_dropout
```

Use `_A` for generator `G_A` and `_B` for generator `G_B`.

## Image Size

The default ResNet generator expects image width and height to be divisible by 4. If you use `--preprocess none`, the loader still adjusts dimensions to a valid multiple.

## High Resolution

CycleGAN loads two generators and two discriminators during training, so memory use is high. For large images, train with crops:

```bash
python train.py --dataroot ./datasets/your_dataset --name your_experiment --model cycle_gan --preprocess scale_width_and_crop --load_size 1024 --crop_size 360
```

At test time, you can use larger images because only inference is needed.

## Loss Curves

GAN loss curves are not enough to judge quality. Periodically inspect generated images in:

```text
checkpoints/your_experiment/web/index.html
```
