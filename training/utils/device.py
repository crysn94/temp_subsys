"""
======================================================================
Device Utilities
======================================================================
"""

import torch


class DeviceManager:

    @staticmethod
    def get_device() -> str:
        """
        Returns the best available device for training.
        """

        if torch.cuda.is_available():
            return "0"          # First CUDA GPU

        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return "mps"

        return "cpu"

    ##################################################################

    @staticmethod
    def print_info():

        print("\n" + "=" * 70)
        print("DEVICE INFORMATION")
        print("=" * 70)

        print(f"PyTorch Version : {torch.__version__}")
        print(f"CUDA Available  : {torch.cuda.is_available()}")
        print(f"CUDA Devices    : {torch.cuda.device_count()}")

        if torch.cuda.is_available():

            print(
                "GPU Name        :",
                torch.cuda.get_device_name(0)
            )

        print("Selected Device :", DeviceManager.get_device())

        print("=" * 70)