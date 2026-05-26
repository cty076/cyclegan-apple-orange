# CycleGAN PyTorch Project

This local project keeps the CycleGAN parts of the original `junyanz/pytorch-CycleGAN-and-pix2pix` implementation and removes the pix2pix/colorization code path.

The research method used here is:

> Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

## Key Files

- `train.py`: train a CycleGAN model.
- `test.py`: run inference and save generated images to an HTML page.
- `main.py`: local menu workflow for the apple/orange CycleGAN experiment.
- `models/cycle_gan_model.py`: CycleGAN losses and optimization logic.
- `models/networks.py`: generators, discriminators, GAN loss, and schedulers.
- `models/test_model.py`: single-direction generator inference for CycleGAN.
- `data/unaligned_dataset.py`: unpaired `trainA/trainB` dataset loader.
- `data/single_dataset.py`: single-folder test dataset loader.
- `options/`: command-line options for training and testing.
- `util/`: HTML, image saving, logging, and helper utilities.

## Dataset Format

CycleGAN uses unpaired image folders:

```text
datasets/your_dataset/
  trainA/
  trainB/
  testA/
  testB/
```

`trainA` and `trainB` do not need one-to-one matching images.

Datasets, checkpoints, logs, generated inference inputs, and result images are intentionally ignored by Git. Prepare or download datasets locally before training.

For the apple/orange experiment used in this project, place the source image folders next to this repository workspace and use `main.py` or `scripts/prepare_apple_orange_dataset.py` to build:

```text
datasets/apple_orange_user/
  trainA/
  trainB/
  testA/
  testB/
```

## Train

```bash
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
```

For the local apple/orange workflow, you can run:

```bash
python main.py
```

## Test

Test both directions with the full CycleGAN model:

```bash
python test.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
```

Run a single direction with one generator:

```bash
python test.py --dataroot ./datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --model_suffix _A --no_dropout
```

Results are saved under `results/`.

## Citation

If you use this project for a report, cite the CycleGAN paper:

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  booktitle={ICCV},
  year={2017}
}
```
