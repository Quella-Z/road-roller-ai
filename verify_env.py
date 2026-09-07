"""verify_env.py

Verifies that the project-local environment is complete and runnable.

Exits with code 0 only when every check passes. Used by the setup scripts as
the success/failure gate and by users/IT for diagnostics.

Checks:
  - Python version
  - torch version
  - torchvision import
  - torch.cuda.is_available()
  - CUDA device name when applicable
  - Ultralytics import
  - SAM2VideoPredictor import
  - EasyOCR import
  - OpenCV version
  - cv2.imshow availability
  - cv2.selectROI availability
  - NumPy version
  - SciPy version
  - openpyxl import
  - tkinter availability
"""

import argparse
import sys

REQUIRED_PYTHON = (3, 11)
REQUIRED_TORCH = "2.3.1"
REQUIRED_TORCHVISION = "0.18.1"
REQUIRED_ULTRALYTICS = "8.3.94"
REQUIRED_NUMPY = "1.26.4"
REQUIRED_SCIPY = "1.11.4"
REQUIRED_OPENCV = "4.10.0"


class Checker:
    def __init__(self):
        self.checks = []
        self.failed = False

    def report(self, name, ok, detail=""):
        status = "PASS" if ok else "FAIL"
        if not ok:
            self.failed = True
        msg = f"[{status}] {name}"
        if detail:
            msg += f" -> {detail}"
        print(msg)
        self.checks.append((name, ok, detail))


def main():
    parser = argparse.ArgumentParser(description="Verify the Road Roller environment.")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Fail if torch.cuda.is_available() is False (used by the GPU installer).",
    )
    args = parser.parse_args()

    c = Checker()
    python_ok = sys.version_info[:2] == REQUIRED_PYTHON
    c.report(
        f"Python version {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]}",
        python_ok,
        f"running {sys.version.split()[0]}",
    )

    try:
        import torch

        torch_ok = torch.__version__.startswith(REQUIRED_TORCH)
        c.report(f"torch {REQUIRED_TORCH}", torch_ok, f"installed {torch.__version__}")

        cuda = bool(torch.cuda.is_available())
        if args.require_cuda:
            c.report("torch.cuda.is_available() == True", cuda, str(cuda))
        else:
            c.report("torch.cuda.is_available() reported", True, str(cuda))
        if cuda:
            try:
                name = torch.cuda.get_device_name(0)
                c.report("CUDA device name", True, name)
            except Exception as exc:  # noqa: BLE001
                c.report("CUDA device name", False, str(exc))
    except Exception as exc:  # noqa: BLE001
        c.report(f"torch {REQUIRED_TORCH}", False, str(exc))

    try:
        import torchvision

        tv_ok = torchvision.__version__.startswith(REQUIRED_TORCHVISION)
        c.report(
            f"torchvision {REQUIRED_TORCHVISION}",
            tv_ok,
            f"installed {torchvision.__version__}",
        )
    except Exception as exc:  # noqa: BLE001
        c.report(f"torchvision {REQUIRED_TORCHVISION}", False, str(exc))

    try:
        import ultralytics

        ul_ok = ultralytics.__version__ == REQUIRED_ULTRALYTICS
        c.report(
            f"ultralytics {REQUIRED_ULTRALYTICS}",
            ul_ok,
            f"installed {ultralytics.__version__}",
        )
    except Exception as exc:  # noqa: BLE001
        c.report(f"ultralytics {REQUIRED_ULTRALYTICS}", False, str(exc))

    try:
        from ultralytics.models.sam import SAM2VideoPredictor  # noqa: F401

        c.report("SAM2VideoPredictor import", True)
    except Exception as exc:  # noqa: BLE001
        c.report("SAM2VideoPredictor import", False, str(exc))

    try:
        import easyocr  # noqa: F401

        c.report("EasyOCR import", True, getattr(easyocr, "__version__", "n/a"))
    except Exception as exc:  # noqa: BLE001
        c.report("EasyOCR import", False, str(exc))

    try:
        import cv2

        cv_ok = cv2.__version__.startswith(REQUIRED_OPENCV)
        c.report(f"OpenCV {REQUIRED_OPENCV}", cv_ok, f"installed {cv2.__version__}")
        c.report("cv2.imshow available", hasattr(cv2, "imshow"))
        c.report("cv2.selectROI available", hasattr(cv2, "selectROI"))
    except Exception as exc:  # noqa: BLE001
        c.report(f"OpenCV {REQUIRED_OPENCV}", False, str(exc))

    try:
        import numpy

        np_ok = numpy.__version__ == REQUIRED_NUMPY
        c.report(f"numpy {REQUIRED_NUMPY}", np_ok, f"installed {numpy.__version__}")
    except Exception as exc:  # noqa: BLE001
        c.report(f"numpy {REQUIRED_NUMPY}", False, str(exc))

    try:
        import scipy

        sp_ok = scipy.__version__ == REQUIRED_SCIPY
        c.report(f"scipy {REQUIRED_SCIPY}", sp_ok, f"installed {scipy.__version__}")
    except Exception as exc:  # noqa: BLE001
        c.report(f"scipy {REQUIRED_SCIPY}", False, str(exc))

    try:
        import openpyxl  # noqa: F401

        c.report("openpyxl import", True)
    except Exception as exc:  # noqa: BLE001
        c.report("openpyxl import", False, str(exc))

    try:
        import tkinter  # noqa: F401

        c.report("tkinter availability", True)
    except Exception as exc:  # noqa: BLE001
        c.report("tkinter availability", False, str(exc))

    print("")
    print("=" * 60)
    if c.failed:
        print("VERIFICATION FAILED")
        print("See the failed checks above and re-run setup.")
    else:
        print("VERIFICATION PASSED")
    print("=" * 60)
    return 1 if c.failed else 0


if __name__ == "__main__":
    sys.exit(main())