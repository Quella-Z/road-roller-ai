"""launcher.py

Desktop launcher for the Road Roller Compaction Analyzer.

This is a thin tkinter/ttk wrapper around the approved production CLI scripts
(track_roller.py and analyze_compaction.py). It does not contain or modify any
tracking or analysis logic; it only builds the command lines and streams the
production output into a log panel.

The long-running SAM2 tracking operation is always launched as a subprocess so
the UI remains responsive and the interactive production workflow (timeline
selection, road-roller ROI selection, OpenCV windows) is preserved unchanged.
"""

import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

PROJECT_ROOT = Path(__file__).resolve().parent
APP_TITLE = "Road Roller Compaction Analyzer"

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv")
SUPPORTED_SECTION_ORDERS = ["left-to-right", "right-to-left"]

CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def read_version():
    try:
        return (PROJECT_ROOT / "VERSION").read_text(encoding="utf-8").strip()
    except Exception:  # noqa: BLE001
        return "unknown"


APP_VERSION = read_version()


def detect_environment():
    env = {
        "python": sys.version.split()[0],
        "mode": "Unknown",
        "device": "",
        "torch": "",
        "torchvision": "",
        "cuda_available": False,
        "ultralytics": "",
        "sam2": False,
        "easyocr": False,
        "opencv": "",
        "imshow": False,
        "selectROI": False,
        "numpy": "",
        "scipy": "",
        "openpyxl": False,
        "errors": [],
    }

    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda_available"] = bool(torch.cuda.is_available())
        if env["cuda_available"]:
            env["mode"] = "GPU"
            try:
                env["device"] = torch.cuda.get_device_name(0)
            except Exception:  # noqa: BLE001
                env["device"] = "CUDA device"
        else:
            env["mode"] = "CPU"
            env["device"] = "CPU"
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"torch: {exc}")

    try:
        import torchvision

        env["torchvision"] = torchvision.__version__
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"torchvision: {exc}")

    try:
        import ultralytics

        env["ultralytics"] = ultralytics.__version__
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"ultralytics: {exc}")

    try:
        from ultralytics.models.sam import SAM2VideoPredictor  # noqa: F401

        env["sam2"] = True
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"SAM2VideoPredictor: {exc}")

    try:
        import easyocr  # noqa: F401

        env["easyocr"] = True
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"easyocr: {exc}")

    try:
        import cv2

        env["opencv"] = cv2.__version__
        env["imshow"] = hasattr(cv2, "imshow")
        env["selectROI"] = hasattr(cv2, "selectROI")
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"opencv: {exc}")

    try:
        import numpy

        env["numpy"] = numpy.__version__
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"numpy: {exc}")

    try:
        import scipy

        env["scipy"] = scipy.__version__
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"scipy: {exc}")

    try:
        import openpyxl  # noqa: F401

        env["openpyxl"] = True
    except Exception as exc:  # noqa: BLE001
        env["errors"].append(f"openpyxl: {exc}")

    return env


class LogPanel(ttk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.text = tk.Text(self, wrap="word", height=16, state="disabled")
        scroll = ttk.Scrollbar(self, command=self.text.yview)
        self.text.configure(yscrollcommand=scroll.set)
        self.text.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def append(self, line):
        self.text.configure(state="normal")
        self.text.insert("end", line)
        self.text.see("end")
        self.text.configure(state="disabled")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(f"{APP_TITLE} {APP_VERSION}")
        self.geometry("820x760")
        self.minsize(760, 680)

        self.env = None
        self.proc = None
        self.running = False
        self.output_queue = queue.Queue()
        self.last_run_dir = None

        self._build_variables()
        self._build_ui()
        self._start_env_detection()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ----- variables ---------------------------------------------------------
    def _build_variables(self):
        self.video_var = tk.StringVar()
        self.sections_var = tk.IntVar(value=3)
        self.order_var = tk.StringVar(value="left-to-right")
        self.start_enabled_var = tk.BooleanVar(value=False)
        self.start_var = tk.StringVar()
        self.length_vars = []
        self.length_entries = []

        self.status_var = tk.StringVar(value="Detecting environment...")
        self.mode_var = tk.StringVar(value="Runtime: ...")
        self.device_var = tk.StringVar(value="Device: ...")
        self.python_var = tk.StringVar(value="Python: ...")

    # ----- UI -----------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        header = ttk.Frame(self)
        header.pack(fill="x", **pad)
        ttk.Label(header, text=APP_TITLE, font=("Segoe UI", 15, "bold")).pack(
            side="left"
        )
        ttk.Label(header, text=APP_VERSION, font=("Segoe UI", 10)).pack(
            side="left", padx=(6, 0), pady=(6, 0)
        )

        info = ttk.LabelFrame(self, text="Environment")
        info.pack(fill="x", **pad)
        for var in (self.mode_var, self.device_var, self.python_var):
            ttk.Label(info, textvariable=var).pack(anchor="w", padx=8, pady=1)
        ttk.Label(info, textvariable=self.status_var, foreground="#B00020").pack(
            anchor="w", padx=8, pady=1
        )

        inputs = ttk.LabelFrame(self, text="Tracking Inputs")
        inputs.pack(fill="x", **pad)

        row = ttk.Frame(inputs)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text="Video file").pack(side="left")
        ttk.Entry(row, textvariable=self.video_var).pack(
            side="left", fill="x", expand=True, padx=6
        )
        ttk.Button(row, text="Browse...", command=self._browse_video).pack(side="left")

        row2 = ttk.Frame(inputs)
        row2.pack(fill="x", padx=8, pady=4)
        ttk.Label(row2, text="Number of sections").pack(side="left")
        spin = ttk.Spinbox(
            row2,
            from_=1,
            to=30,
            textvariable=self.sections_var,
            width=6,
            command=self._rebuild_length_rows,
        )
        spin.pack(side="left", padx=6)
        spin.bind("<Return>", lambda e: self._rebuild_length_rows())
        spin.bind("<FocusOut>", lambda e: self._rebuild_length_rows())

        ttk.Label(row2, text="Section order").pack(side="left", padx=(24, 0))
        order = ttk.Combobox(
            row2,
            textvariable=self.order_var,
            values=SUPPORTED_SECTION_ORDERS,
            state="readonly",
            width=16,
        )
        order.pack(side="left", padx=6)

        lengths_frame = ttk.LabelFrame(inputs, text="Section lengths (meters)")
        lengths_frame.pack(fill="x", padx=8, pady=4)
        self.lengths_frame = lengths_frame

        start_frame = ttk.Frame(inputs)
        start_frame.pack(fill="x", padx=8, pady=4)
        ttk.Checkbutton(
            start_frame,
            text="Specify start time (seconds)  [leave unchecked for interactive timeline]",
            variable=self.start_enabled_var,
        ).pack(side="left")
        ttk.Entry(start_frame, textvariable=self.start_var, width=10).pack(
            side="left", padx=6
        )

        self._rebuild_length_rows()

        actions = ttk.Frame(self)
        actions.pack(fill="x", **pad)
        self.start_btn = ttk.Button(
            actions, text="Start Analysis", command=self._start_analysis
        )
        self.start_btn.pack(side="left", padx=4)
        self.analyze_btn = ttk.Button(
            actions, text="Analyze Existing Tracking", command=self._analyze_existing
        )
        self.analyze_btn.pack(side="left", padx=4)
        self.open_btn = ttk.Button(
            actions, text="Open Output Folder", command=self._open_output_folder
        )
        self.open_btn.pack(side="left", padx=4)
        self.exit_btn = ttk.Button(actions, text="Exit", command=self._on_close)
        self.exit_btn.pack(side="right", padx=4)

        log_frame = ttk.LabelFrame(self, text="Log")
        log_frame.pack(fill="both", expand=True, **pad)
        self.log = LogPanel(log_frame)
        self.log.pack(fill="both", expand=True, padx=8, pady=4)

        self._append_log(f"Road Roller Compaction Analyzer {APP_VERSION}\n")
        self._append_log(
            "Start Analysis runs the production SAM2 tracking pipeline.\n"
        )
        self._append_log(
            "The OpenCV timeline and road-roller selection windows appear in a "
            "separate window during processing.\n"
        )

    def _rebuild_length_rows(self):
        for child in self.lengths_frame.winfo_children():
            child.destroy()
        self.length_vars = []
        self.length_entries = []

        try:
            count = max(1, int(self.sections_var.get()))
        except Exception:  # noqa: BLE001
            count = 1

        for i in range(1, count + 1):
            row = ttk.Frame(self.lengths_frame)
            row.pack(fill="x", padx=8, pady=2)
            ttk.Label(row, text=f"Section {i} Length (m)", width=20).pack(side="left")
            var = tk.StringVar()
            entry = ttk.Entry(row, textvariable=var, width=12)
            entry.pack(side="left")
            self.length_vars.append(var)
            self.length_entries.append(entry)

    # ----- environment detection ---------------------------------------------
    def _start_env_detection(self):
        def worker():
            self.env = detect_environment()
            self.output_queue.put(("__env__", None))

        threading.Thread(target=worker, daemon=True).start()
        self.after(100, self._poll)

    def _apply_env(self):
        env = self.env
        self.mode_var.set(f"Runtime: {env['mode']}")
        self.device_var.set(f"Device: {env['device']}")
        self.python_var.set(f"Python: {env['python']}  |  torch {env['torch'] or 'n/a'}"
                            f"  |  OpenCV {env['opencv'] or 'n/a'}")
        if env["errors"]:
            self.status_var.set(
                f"Environment degraded: {len(env['errors'])} problem(s) "
                f"(see Verify via setup)"
            )
        elif not env["imshow"] or not env["selectROI"]:
            self.status_var.set(
                "WARNING: OpenCV GUI (imshow/selectROI) unavailable."
            )
        elif env["sam2"] and env["easyocr"] and env["openpyxl"]:
            self.status_var.set("Environment OK")
        else:
            self.status_var.set("Environment incomplete; run SETUP again.")

    # ----- logging -----------------------------------------------------------
    def _append_log(self, line):
        self.log.append(line)

    def _poll(self):
        try:
            while True:
                item = self.output_queue.get_nowait()
                if item[0] == "__env__":
                    self._apply_env()
                elif item[0] == "__done__":
                    self._finish_run(item[1])
                else:
                    self._append_log(item[1])
        except queue.Empty:
            pass
        self.after(100, self._poll)

    # ----- subprocess helpers -------------------------------------------------
    def _start_subprocess(self, cmd, kind):
        if self.running:
            return
        self.running = True
        self._set_buttons_state(False)
        self._append_log("\n" + "-" * 70 + "\n")
        self._append_log(f"$ {' '.join(cmd)}\n\n")

        try:
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=CREATE_NO_WINDOW,
            )
        except Exception as exc:  # noqa: BLE001
            self._append_log(f"Failed to launch process: {exc}\n")
            self._finish_run(1)
            return

        def reader():
            try:
                for line in self.proc.stdout:
                    self.output_queue.put(("log", line))
            except Exception as exc:  # noqa: BLE001
                self.output_queue.put(("log", f"\n[log reader error] {exc}\n"))
            finally:
                self.output_queue.put(("__done__", None))

        threading.Thread(target=reader, daemon=True).start()

    def _finish_run(self, _unused):
        if self.proc is not None:
            try:
                self.proc.wait(timeout=1)
            except Exception:  # noqa: BLE001
                pass
            self.proc = None
        self.running = False
        self._set_buttons_state(True)
        self._append_log("\nProcess finished.\n")

    def _set_buttons_state(self, enabled):
        state = "normal" if enabled else "disabled"
        self.start_btn.configure(state=state)
        self.analyze_btn.configure(state=state)
        self.open_btn.configure(state=state)

    def _terminate(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.terminate()
            except Exception:  # noqa: BLE001
                pass

    # ----- actions ------------------------------------------------------------
    def _browse_video(self):
        path = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", " ".join(f"*{e}" for e in VIDEO_EXTENSIONS)),
                ("All files", "*.*"),
            ],
        )
        if path:
            self.video_var.set(path)

    def _validate_inputs(self):
        video = self.video_var.get().strip()
        if not video:
            messagebox.showerror("Missing input", "Please select a video file.")
            return None
        if not Path(video).exists():
            messagebox.showerror("Missing input", f"Video not found:\n{video}")
            return None

        try:
            sections = int(self.sections_var.get())
        except Exception:  # noqa: BLE001
            messagebox.showerror("Invalid input", "Number of sections must be an integer.")
            return None
        if sections < 1:
            messagebox.showerror("Invalid input", "Number of sections must be >= 1.")
            return None

        lengths = []
        for i, var in enumerate(self.length_vars[:sections], start=1):
            raw = var.get().strip()
            try:
                value = float(raw)
            except Exception:  # noqa: BLE001
                messagebox.showerror(
                    "Invalid input", f"Section {i} length must be a number."
                )
                return None
            if value <= 0:
                messagebox.showerror(
                    "Invalid input", f"Section {i} length must be > 0."
                )
                return None
            lengths.append(raw)

        if self.start_enabled_var.get() and self.start_var.get().strip():
            try:
                start = float(self.start_var.get().strip())
            except Exception:  # noqa: BLE001
                messagebox.showerror("Invalid input", "Start time must be a number.")
                return None
            if start < 0:
                messagebox.showerror("Invalid input", "Start time must be >= 0.")
                return None

        return {"video": video, "sections": sections, "lengths": lengths}

    def _start_analysis(self):
        result = self._validate_inputs()
        if result is None:
            return

        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "track_roller.py"),
            "--video",
            result["video"],
            "--sections",
            str(result["sections"]),
            "--lengths",
            *result["lengths"],
            "--section-order",
            self.order_var.get(),
            "--output-root",
            str(PROJECT_ROOT / "output"),
        ]
        if self.start_enabled_var.get() and self.start_var.get().strip():
            cmd += ["--start-sec", self.start_var.get().strip()]

        self._start_subprocess(cmd, "tracking")

    def _analyze_existing(self):
        run_dir = filedialog.askdirectory(
            title="Select an existing output/run directory (containing trajectory.csv)"
        )
        if not run_dir:
            return
        trajectory = Path(run_dir) / "trajectory.csv"
        if not trajectory.exists():
            messagebox.showerror(
                "Invalid directory",
                f"No trajectory.csv found in:\n{run_dir}",
            )
            return

        self.last_run_dir = run_dir
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "analyze_compaction.py"),
            "--run-dir",
            run_dir,
        ]
        self._start_subprocess(cmd, "analysis")

    def _open_output_folder(self):
        target = (
            Path(self.last_run_dir)
            if self.last_run_dir and Path(self.last_run_dir).exists()
            else PROJECT_ROOT / "output"
        )
        try:
            target.mkdir(parents=True, exist_ok=True)
            os.startfile(str(target))  # noqa: S606
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Could not open output folder:\n{exc}")

    # ----- close --------------------------------------------------------------
    def _on_close(self):
        if self.running:
            if not messagebox.askyesno(
                "Process running",
                "A process is still running. Stop it and exit?",
            ):
                return
            self._terminate()
        self.destroy()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()