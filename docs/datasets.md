# CycleGAN Datasets

CycleGAN trains on unpaired data. Create a dataset directory with:

```text
dataset_name/
  trainA/
  trainB/
  testA/
  testB/
```

`trainA` and `trainB` are two image domains. Images do not need to be paired by filename or content.

Example:

```bash
python train.py --dataroot ./datasets/horse2zebra --name horse2zebra_cyclegan --model cycle_gan
```

You can download official CycleGAN datasets with:

```bash
bash ./datasets/download_cyclegan_dataset.sh horse2zebra
```

Useful dataset choices include `horse2zebra`, `apple2orange`, `summer2winter_yosemite`, `monet2photo`, and `iphone2dslr_flower`.

The method works best when both domains share structure, such as horse/zebra, apple/orange, summer/winter landscapes, or painting/photo landscapes.
