from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parent.parent
APPLE_SRC = WORKSPACE / "苹果"
ORANGE_SRC = WORKSPACE / "橘子"
DATASET_ROOT = ROOT / "datasets" / "apple_orange_user"
CHECKPOINT_NAME = "apple_orange_user_v1"
RESULTS_A2B = ROOT / "results" / "apple_orange_AtoB"
RESULTS_B2A = ROOT / "results" / "apple_orange_BtoA"
PYTHON_EXE = Path(sys.executable)
TEST_COUNT = 20


def run_command(args: list[str]) -> int:
    print("\n[运行命令]")
    print(" ".join(f'"{a}"' if " " in a else a for a in args))
    print()
    proc = subprocess.Popen(args, cwd=str(ROOT))
    return proc.wait()


def ensure_sources() -> None:
    if not APPLE_SRC.exists():
        raise FileNotFoundError(f"未找到苹果数据目录: {APPLE_SRC}")
    if not ORANGE_SRC.exists():
        raise FileNotFoundError(f"未找到橘子数据目录: {ORANGE_SRC}")


def count_jpgs(folder: Path) -> int:
    return len([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])


def count_dataset_splits() -> dict[str, int]:
    result = {}
    for split in ["trainA", "trainB", "testA", "testB"]:
        folder = DATASET_ROOT / split
        result[split] = len(list(folder.glob("*.jpg"))) if folder.exists() else 0
    return result


def expected_split_counts() -> dict[str, int]:
    ensure_sources()
    apple_count = count_jpgs(APPLE_SRC)
    orange_count = count_jpgs(ORANGE_SRC)
    if apple_count <= TEST_COUNT:
        raise ValueError(f"苹果图片数量不足，当前 {apple_count} 张，至少需要大于 {TEST_COUNT}")
    if orange_count <= TEST_COUNT:
        raise ValueError(f"橘子图片数量不足，当前 {orange_count} 张，至少需要大于 {TEST_COUNT}")
    return {
        "trainA": apple_count - TEST_COUNT,
        "trainB": orange_count - TEST_COUNT,
        "testA": TEST_COUNT,
        "testB": TEST_COUNT,
    }


def prepare_dataset(force: bool = False) -> None:
    ensure_sources()
    args = [
        str(PYTHON_EXE),
        str(ROOT / "scripts" / "prepare_apple_orange_dataset.py"),
        "--apple-src",
        str(APPLE_SRC),
        "--orange-src",
        str(ORANGE_SRC),
        "--dataset-root",
        str(DATASET_ROOT),
        "--test-count",
        str(TEST_COUNT),
    ]
    if force:
        args.append("--force")
    code = run_command(args)
    if code != 0:
        raise RuntimeError("数据集准备失败")


def has_existing_checkpoint() -> bool:
    checkpoint_dir = ROOT / "checkpoints" / CHECKPOINT_NAME
    return checkpoint_dir.exists() and any(checkpoint_dir.glob("latest_net_*.pth"))


def latest_epoch_count() -> int:
    loss_log = ROOT / "checkpoints" / CHECKPOINT_NAME / "loss_log.txt"
    if not loss_log.exists():
        return 1

    last_epoch = 1
    with loss_log.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "(epoch:" in line:
                try:
                    frag = line.split("(epoch:")[1].split(",")[0].strip()
                    last_epoch = max(last_epoch, int(frag))
                except Exception:
                    continue
    return last_epoch + 1


def show_config() -> None:
    ensure_sources()
    apple_count = count_jpgs(APPLE_SRC)
    orange_count = count_jpgs(ORANGE_SRC)
    expected = expected_split_counts()
    actual = count_dataset_splits()
    checkpoint_dir = ROOT / "checkpoints" / CHECKPOINT_NAME

    print("\n=== 当前训练配置 ===")
    print(f"Python 环境: {PYTHON_EXE}")
    print(f"苹果原始目录: {APPLE_SRC}")
    print(f"橘子原始目录: {ORANGE_SRC}")
    print(f"苹果图片数量: {apple_count}")
    print(f"橘子图片数量: {orange_count}")
    print(f"生成数据集目录: {DATASET_ROOT}")
    print(f"预期划分: trainA={expected['trainA']} trainB={expected['trainB']} testA={expected['testA']} testB={expected['testB']}")
    print(f"当前数据集: trainA={actual['trainA']} trainB={actual['trainB']} testA={actual['testA']} testB={actual['testB']}")
    print(f"实验名: {CHECKPOINT_NAME}")
    print(f"checkpoint 目录: {checkpoint_dir}")
    print("模型: cycle_gan")
    print("batch_size: 1")
    print("num_threads: 0")
    print("n_epochs: 100")
    print("n_epochs_decay: 100")
    print(f"可续训: {'是' if has_existing_checkpoint() else '否'}")
    if has_existing_checkpoint():
        print(f"建议续训 epoch_count: {latest_epoch_count()}")
    print("=====================\n")


def start_training() -> None:
    prepare_dataset(force=False)
    args = [
        str(PYTHON_EXE),
        str(ROOT / "train.py"),
        "--dataroot",
        str(DATASET_ROOT),
        "--name",
        CHECKPOINT_NAME,
        "--model",
        "cycle_gan",
        "--batch_size",
        "1",
        "--num_threads",
        "0",
        "--save_latest_freq",
        "1000",
        "--save_epoch_freq",
        "5",
        "--print_freq",
        "50",
    ]

    if has_existing_checkpoint():
        args.extend(["--continue_train", "--epoch_count", str(latest_epoch_count())])
        print(f"检测到已有 checkpoint，将从 epoch_count={latest_epoch_count()} 继续训练。")

    run_command(args)


def run_test() -> None:
    checkpoint_dir = ROOT / "checkpoints" / CHECKPOINT_NAME
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"未找到训练结果目录: {checkpoint_dir}")

    expected = expected_split_counts()
    actual = count_dataset_splits()
    if actual != expected:
        print("检测到当前数据集划分不完整，先自动准备数据集。")
        prepare_dataset(force=False)

    args_a2b = [
        str(PYTHON_EXE),
        str(ROOT / "test.py"),
        "--dataroot",
        str(DATASET_ROOT / "testA"),
        "--name",
        CHECKPOINT_NAME,
        "--model",
        "test",
        "--model_suffix",
        "_A",
        "--no_dropout",
        "--num_test",
        str(TEST_COUNT),
        "--results_dir",
        str(RESULTS_A2B),
    ]
    args_b2a = [
        str(PYTHON_EXE),
        str(ROOT / "test.py"),
        "--dataroot",
        str(DATASET_ROOT / "testB"),
        "--name",
        CHECKPOINT_NAME,
        "--model",
        "test",
        "--model_suffix",
        "_B",
        "--no_dropout",
        "--num_test",
        str(TEST_COUNT),
        "--results_dir",
        str(RESULTS_B2A),
    ]
    code1 = run_command(args_a2b)
    if code1 != 0:
        raise RuntimeError("A->B 测试失败")
    code2 = run_command(args_b2a)
    if code2 != 0:
        raise RuntimeError("B->A 测试失败")
    print("\n测试完成：")
    print(f"A->B 结果: {RESULTS_A2B / CHECKPOINT_NAME / 'test_latest' / 'index.html'}")
    print(f"B->A 结果: {RESULTS_B2A / CHECKPOINT_NAME / 'test_latest' / 'index.html'}\n")


def show_menu() -> None:
    print("=== 苹果橘子 CycleGAN 菜单 ===")
    print("1. 查看训练配置")
    print("2. 开始训练")
    print("3. 运行测试推理")
    print("4. 强制重建数据集")
    print("5. 退出")


def main() -> None:
    while True:
        try:
            show_menu()
            choice = input("请输入选项编号: ").strip()
            if choice == "1":
                show_config()
            elif choice == "2":
                show_config()
                input("按回车开始训练...")
                start_training()
            elif choice == "3":
                show_config()
                input("按回车开始测试推理...")
                run_test()
            elif choice == "4":
                show_config()
                input("按回车强制重建数据集...")
                prepare_dataset(force=True)
            elif choice == "5":
                print("已退出。")
                break
            else:
                print("无效选项，请重新输入。\n")
        except Exception as e:
            print(f"\n发生错误: {e}\n")


if __name__ == "__main__":
    main()
