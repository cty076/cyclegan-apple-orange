# CycleGAN FAQ

## Can I resume training?

Yes. Use `--continue_train`:

```bash
python train.py --dataroot ./datasets/your_dataset --name your_experiment --model cycle_gan --continue_train
```

Use `--epoch_count` if you need to control the starting epoch number.

## What dataset layout do I need?

Use unpaired domain folders:

```text
datasets/your_dataset/
  trainA/
  trainB/
  testA/
  testB/
```

Images in A and B do not need matching filenames.

## Why does training loss not converge smoothly?

GAN losses often oscillate. For CycleGAN, inspect generated images in `checkpoints/<experiment>/web/index.html` instead of relying only on loss curves.

## How do I test only one direction?

Use `--model test` and select the generator suffix:

```bash
python test.py --dataroot ./datasets/your_dataset/testA --name your_experiment --model test --model_suffix _A --no_dropout
```

`_A` loads `latest_net_G_A.pth`; `_B` loads `latest_net_G_B.pth`.

## What should I do if checkpoint loading fails?

Make sure the testing options match training options, especially:

- `--netG`
- `--norm`
- `--input_nc`
- `--output_nc`
- `--no_dropout`

The default CycleGAN setup is `resnet_9blocks`, instance normalization, and no dropout.

## Can I run on CPU?

Yes, if your local options and environment support it. Expect inference and training to be much slower than CUDA.

## Why is CycleGAN memory intensive?

During training, CycleGAN uses two generators and two discriminators. For large images, train on crops and test at higher resolution later.

## What is identity loss?

Identity loss encourages a generator to preserve images that are already in the target domain. It can reduce unnecessary color or tone changes. Use `--lambda_identity` to enable it.
