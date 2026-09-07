# Road Roller Compaction Analyzer

Desktop application that tracks a road roller in a video using SAM 2.1 Tiny,
then produces compaction section, speed, and effective-pass reports (Excel,
CSV, and a QA overlay image).

- Version: see `VERSION` (initial release `v1.0.0`)
- Runtime profiles: **CPU** and **NVIDIA GPU (CUDA 12.1)**

---

## End User

### CPU

1. Download `RoadRollerAI-v1.0.0-CPU.zip`.
2. Extract the ZIP anywhere you have write access (no `D:` drive required).
3. Double-click **`SETUP.cmd`** once. It installs a private environment with no
   administrator rights (Python 3.11, PyTorch CPU, OpenCV, EasyOCR, etc.).
4. For normal use, double-click **`RUN_ROLLER_AI.cmd`**.

> On the very first tracking run the SAM 2.1 Tiny model (~78 MB) is downloaded
> automatically by Ultralytics. This is a one-time step and requires internet.
> (In the released packages the model is bundled, so no download is needed.)

### GPU

1. Download `RoadRollerAI-v1.0.0-GPU.zip`.
2. Extract the ZIP anywhere you have write access.
3. Double-click **`SETUP.cmd`** once. It checks your NVIDIA driver first
   (CUDA 12.1 requires driver **>= 527.41**), then installs the CUDA 12.1
   PyTorch runtime. It never installs or modifies system drivers.
4. For normal use, double-click **`RUN_ROLLER_AI.cmd`**.

> If the driver is too old, setup stops and asks you to use the CPU package or
> contact IT. It will never fall back to CPU silently.

### Using the application

- **Start Analysis** — choose a video, number of sections, each section length
  in meters, and the section order. The existing interactive production
  workflow appears: pick the tracking start time on the timeline, then draw a
  box around the road roller.
- **Analyze Existing Tracking** — re-run the analysis (without re-tracking) on
  any output folder that already contains `trajectory.csv`. Useful on CPU
  machines.
- **Open Output Folder** — opens the output folder in Windows Explorer.

---

## Developer

### Source structure

```
track_roller.py          # SAM 2.1 tracking (master implementation, unchanged)
analyze_compaction.py    # section/speed/OCR analysis (master implementation, unchanged)
launcher.py              # tkinter desktop wrapper around the two scripts above
verify_env.py            # environment verification (used as the setup gate)
VERSION                  # single source of truth for the release version
requirements/
  common.lock            # shared app dependencies (pinned)
  cpu.lock               # PyTorch CPU wheel index + pinned torch/torchvision
  gpu.lock               # PyTorch CUDA 12.1 wheel index + pinned torch/torchvision
deployment/
  SETUP_CPU.cmd          # runs setup_cpu.ps1
  SETUP_GPU.cmd          # runs setup_gpu.ps1
  setup_cpu.ps1          # creates .venv, installs CPU runtime, verifies
  setup_gpu.ps1          # driver check + CUDA 12.1 runtime + verification
  RUN_ROLLER_AI.cmd      # canonical double-click launcher
SETUP.cmd                # package-level entry (CPU by default)
RUN_ROLLER_AI.cmd        # package-level convenience launcher
build_release.ps1        # builds the two release ZIPs
```

### CPU / GPU environment difference

Both profiles use the **same** `track_roller.py`, `analyze_compaction.py`,
`launcher.py`, and reporting logic. The only difference is the PyTorch runtime:

- CPU: `torch==2.3.1` + `torchvision==0.18.1` from
  `https://download.pytorch.org/whl/cpu`.
- GPU: the same versions from `https://download.pytorch.org/whl/cu121`.

### Dependency locking

Versions are pinned in `requirements/*.lock`:

- `numpy==1.26.4`, `scipy==1.11.4`, `opencv-python==4.10.0.84`,
  `ultralytics==8.3.94`, `easyocr==1.7.2`, `openpyxl==3.1.5`.

EasyOCR depends on `opencv-python-headless`, but the tracker needs the GUI
build (`cv2.imshow`, `cv2.selectROI`). The setup scripts therefore remove both
OpenCV distributions and install only `opencv-python` after resolution.
`verify_env.py` gates success on `hasattr(cv2, "imshow")` and
`hasattr(cv2, "selectROI")`.

### How to test without re-running SAM2

`analyze_compaction.py` can be re-run against any existing run folder:

```
python analyze_compaction.py --run-dir <folder-containing-trajectory.csv>
```

### Release process

1. Bump `VERSION`.
2. Commit and tag (`git tag v<version>`).
3. Run `powershell -File build_release.ps1` to produce
   `dist/RoadRollerAI-v<version>-CPU.zip` and `...-GPU.zip`.
4. Create a GitHub release and attach both ZIPs.

### Versioning

Production builds correspond to a Git tag/release (e.g. `v1.0.0`). The main
branch is not used to auto-update installed copies.
