# 苹果橘子 CycleGAN 图像转换项目

本仓库是基于 `junyanz/pytorch-CycleGAN-and-pix2pix` 整理出的 CycleGAN 研究代码版本。当前版本只保留与 CycleGAN 相关的训练、推理、数据读取和模型网络代码，已经移除 pix2pix、colorization 等另一篇论文对应的代码路径。

本项目对应的核心论文方法是：

> Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks

也就是 CycleGAN。它适合做非成对图像转换，例如苹果和橘子之间的风格/外观转换，不要求 `trainA` 和 `trainB` 中的图片一一对应。

## 仓库内容

- `train.py`：训练 CycleGAN 模型。
- `test.py`：使用训练好的生成器进行推理，并保存图片和 HTML 页面。
- `main.py`：本地苹果/橘子实验菜单入口。
- `models/cycle_gan_model.py`：CycleGAN 的生成器、判别器、循环一致性损失和训练逻辑。
- `models/networks.py`：ResNet 生成器、PatchGAN 判别器、GAN loss 和学习率调度。
- `models/test_model.py`：单方向生成器推理。
- `data/unaligned_dataset.py`：非成对训练数据加载器。
- `data/single_dataset.py`：单文件夹推理数据加载器。
- `options/`：训练和测试参数。
- `util/`：图片保存、HTML 输出、日志和可视化工具。
- `scripts/prepare_apple_orange_dataset.py`：将苹果/橘子原图整理成 CycleGAN 数据集格式。

## 数据集格式

CycleGAN 使用非成对数据集，目录结构如下：

```text
datasets/your_dataset/
  trainA/
  trainB/
  testA/
  testB/
```

其中：

- `trainA`：A 域训练图片，例如苹果。
- `trainB`：B 域训练图片，例如橘子。
- `testA`：A 域测试图片。
- `testB`：B 域测试图片。

`trainA` 和 `trainB` 不需要成对，也不需要文件名对应。

数据集、训练权重、日志、推理输入和推理结果都不会直接提交到 GitHub，相关目录已经在 `.gitignore` 中排除。

本项目使用的苹果/橘子数据集已经作为 Release 附件上传：

```text
https://github.com/cty076/cyclegan-apple-orange/releases/tag/apple-orange-bs1-e200
```

数据集压缩包名称：

```text
apple_orange_user_dataset.zip
```

解压后请放到：

```text
datasets/apple_orange_user/
```

数据集划分为：

- `trainA`：365 张苹果训练图。
- `trainB`：361 张橘子训练图。
- `testA`：20 张苹果测试图。
- `testB`：20 张橘子测试图。

## 苹果橘子实验

如果使用本项目中的本地菜单流程，需要将原始图片目录放在仓库上级工作区中：

```text
final_work/
  苹果/
  橘子/
  pytorch-CycleGAN-and-pix2pix-local/
    repo/
```

然后运行：

```bash
python main.py
```

菜单中可以查看配置、准备数据集、开始训练和运行测试推理。

也可以直接运行数据整理脚本：

```bash
python scripts/prepare_apple_orange_dataset.py \
  --apple-src ../苹果 \
  --orange-src ../橘子 \
  --dataset-root ./datasets/apple_orange_user \
  --test-count 20
```

## 训练

基础训练命令：

```bash
python train.py \
  --dataroot ./datasets/apple_orange_user \
  --name apple_orange_bs1_e200 \
  --model cycle_gan \
  --batch_size 1 \
  --n_epochs 100 \
  --n_epochs_decay 100
```

这里的 `100+100 epoch` 表示：

- 前 100 个 epoch 保持初始学习率。
- 后 100 个 epoch 线性衰减学习率到 0。

这也是 CycleGAN 官方常用的训练设置。

## 推理

苹果转橘子方向使用 `G_A`：

```bash
python test.py \
  --dataroot ./datasets/apple_orange_user/testA \
  --name apple_orange_bs1_e200 \
  --model test \
  --model_suffix _A \
  --no_dropout
```

橘子转苹果方向使用 `G_B`：

```bash
python test.py \
  --dataroot ./datasets/apple_orange_user/testB \
  --name apple_orange_bs1_e200 \
  --model test \
  --model_suffix _B \
  --no_dropout
```

推理结果默认保存在 `results/` 目录下。

## 已训练权重

本项目已上传一份苹果/橘子实验的最终生成器权重：

- 训练设置：`batch_size=1`
- 训练轮数：`100+100 epoch`
- 模型名称：`apple_orange_bs1_e200`
- 权重文件：`latest_net_G_A.pth` 和 `latest_net_G_B.pth`

下载地址在 GitHub Releases：

```text
https://github.com/cty076/cyclegan-apple-orange/releases
```

下载后请解压到：

```text
checkpoints/apple_orange_bs1_e200/
```

目录中至少需要包含：

```text
checkpoints/apple_orange_bs1_e200/
  latest_net_G_A.pth
  latest_net_G_B.pth
```

## 说明

本仓库不是 CycleGAN 论文作者的官方仓库，而是在官方 PyTorch 实现基础上，为苹果/橘子 CycleGAN 实验整理出的课程研究版本。原始官方仓库为：

```text
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
```

## 引用

如果在课程报告或论文中使用本项目，请引用 CycleGAN 原论文：

```bibtex
@inproceedings{CycleGAN2017,
  title={Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks},
  author={Zhu, Jun-Yan and Park, Taesung and Isola, Phillip and Efros, Alexei A.},
  booktitle={ICCV},
  year={2017}
}
```
