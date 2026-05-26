import subprocess
from pathlib import Path

import pytest


class TestBeforePush:
    """Basic CycleGAN smoke tests."""

    @pytest.fixture(autouse=True)
    def setup_datasets(self):
        if not Path("./datasets/mini").exists():
            subprocess.run(["bash", "./datasets/download_cyclegan_dataset.sh", "mini"], check=True)

    def test_pretrained_cyclegan_model(self):
        """Test pretrained CycleGAN model download and single-direction inference."""
        if not Path("./checkpoints/horse2zebra_pretrained/latest_net_G.pth").exists():
            subprocess.run(["bash", "./scripts/download_cyclegan_model.sh", "horse2zebra"], check=True)

        result = subprocess.run(
            [
                "python",
                "test.py",
                "--model",
                "test",
                "--dataroot",
                "./datasets/mini",
                "--name",
                "horse2zebra_pretrained",
                "--no_dropout",
                "--num_test",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"CycleGAN test failed: {result.stderr}"

    def test_cyclegan_train_test(self):
        """Test CycleGAN training and testing pipeline."""
        train_result = subprocess.run(
            [
                "python",
                "train.py",
                "--model",
                "cycle_gan",
                "--name",
                "temp_cyclegan",
                "--dataroot",
                "./datasets/mini",
                "--n_epochs",
                "1",
                "--n_epochs_decay",
                "0",
                "--save_latest_freq",
                "10",
                "--print_freq",
                "1",
            ],
            capture_output=True,
            text=True,
        )

        assert train_result.returncode == 0, f"CycleGAN training failed: {train_result.stderr}"

        test_result = subprocess.run(
            [
                "python",
                "test.py",
                "--model",
                "test",
                "--name",
                "temp_cyclegan",
                "--dataroot",
                "./datasets/mini",
                "--num_test",
                "1",
                "--model_suffix",
                "_A",
                "--no_dropout",
            ],
            capture_output=True,
            text=True,
        )

        assert test_result.returncode == 0, f"CycleGAN testing failed: {test_result.stderr}"
