# CycleGAN Code Overview

`train.py` is the training entry point. It parses options, creates an unpaired dataset, creates the CycleGAN model, then runs the training loop and saves checkpoints.

`test.py` is the inference entry point. It loads a saved checkpoint, runs generated images through the model, and writes results to an HTML page.

## data

- `data/unaligned_dataset.py`: loads unpaired domain A/domain B images from `trainA/trainB` or `testA/testB`.
- `data/single_dataset.py`: loads one image folder for single-direction inference with `--model test`.
- `data/base_dataset.py`: shared image transforms and preprocessing helpers.
- `data/image_folder.py`: recursive image file discovery.

## models

- `models/cycle_gan_model.py`: CycleGAN generators, discriminators, cycle consistency loss, identity loss, GAN loss, and optimizer steps.
- `models/test_model.py`: loads one generator for single-direction CycleGAN inference.
- `models/base_model.py`: shared model setup, checkpoint save/load, scheduler, and device handling.
- `models/networks.py`: generator and discriminator architectures plus GAN loss and initialization helpers.

## options

- `options/base_options.py`: shared training/testing options.
- `options/train_options.py`: training-only options such as epoch count, learning rate, and checkpoint frequency.
- `options/test_options.py`: test-only options such as result directory and number of test images.

## util

- `util/visualizer.py`: saves images, HTML previews, logs, and optional W&B records.
- `util/html.py`: writes HTML result pages.
- `util/image_pool.py`: stores previously generated images for discriminator updates.
- `util/util.py`: tensor/image conversion, directory helpers, and DDP setup.
