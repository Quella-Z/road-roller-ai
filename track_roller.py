import os
import cv2
import csv
import json
import sys
import argparse
import subprocess
from pathlib import Path
from collections import deque

import numpy as np
import torch
from ultralytics.models.sam import SAM2VideoPredictor


TARGET_FPS = 1
MAX_WIDTH = 1280
SMOOTH_WINDOW = 7
MAX_INTERPOLATION_GAP_SECONDS = 3.0
DEFAULT_COMPLETION_RATIO = 0.80
DEFAULT_SPEED_LIMIT_KMH = 2.0
DEFAULT_TIMESTAMP_ROI = (0.64, 0.86, 1.00, 1.00)

PROJECT_ROOT = Path(__file__).resolve().parent


def safe_stem(path):
    stem = Path(path).stem.strip()
    allowed = []
    for ch in stem:
        if ch.isalnum() or ch in ("-", "_"):
            allowed.append(ch)
        else:
            allowed.append("_")
    result = "".join(allowed).strip("_")
    return result or "video"


def validate_args(args):
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"Video not found: {args.video}")

    if args.sections < 1:
        raise ValueError("--sections must be >= 1")

    if len(args.lengths) != args.sections:
        raise ValueError(
            f"--sections is {args.sections}, but {len(args.lengths)} lengths were provided. "
            f"Provide exactly one length per section."
        )

    if any(v <= 0 for v in args.lengths):
        raise ValueError("All section lengths must be > 0")

    if args.start_sec is not None and args.start_sec < 0:
        raise ValueError("--start-sec must be >= 0")


def get_video_info(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0:
        fps = 30.0

    duration = frame_count / fps if frame_count > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "duration_sec": duration,
    }


def read_frame_at(cap, time_sec):
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
    ok, frame = cap.read()
    return frame if ok else None


def choose_tracking_start(video_path):
    info = get_video_info(video_path)
    duration_sec = max(1, int(round(info["duration_sec"])))

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video for start-time selection.")

    window_name = "Choose Tracking Start - ENTER confirm | ESC cancel"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1100, 650)

    state = {"sec": 0}

    def on_trackbar(value):
        state["sec"] = int(value)

    cv2.createTrackbar("Time (sec)", window_name, 0, duration_sec, on_trackbar)

    print("")
    print("Choose the first moment where the road roller is clearly visible.")
    print("Drag the Time (sec) slider.")
    print("Press ENTER to confirm, or ESC to cancel.")
    print("")

    selected = None

    while True:
        sec = cv2.getTrackbarPos("Time (sec)", window_name)
        frame = read_frame_at(cap, sec)

        if frame is None:
            canvas = np.zeros((500, 900, 3), dtype=np.uint8)
            cv2.putText(
                canvas,
                "Unable to read this frame",
                (40, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 255),
                2,
            )
            display = canvas
        else:
            display = frame.copy()
            cv2.rectangle(display, (12, 12), (350, 66), (0, 0, 0), -1)
            cv2.putText(
                display,
                f"Selected start: {sec}s",
                (25, 49),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow(window_name, display)
        key = cv2.waitKey(80) & 0xFF

        if key in (13, 10):
            selected = float(sec)
            break
        if key == 27:
            break

    cap.release()
    cv2.destroyWindow(window_name)

    if selected is None:
        raise RuntimeError("Tracking start selection cancelled.")

    print(f"Tracking start selected: {selected:.1f} sec")
    return selected


def create_proxy_video(input_path, output_path, start_sec):
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open input video.")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS))
    src_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if src_fps <= 0:
        src_fps = 30.0

    frame_step = max(1, round(src_fps / TARGET_FPS))
    output_fps = src_fps / frame_step

    scale = min(1.0, MAX_WIDTH / max(src_width, 1))
    out_width = max(2, int(src_width * scale))
    out_height = max(2, int(src_height * scale))
    out_width -= out_width % 2
    out_height -= out_height % 2

    start_frame = max(0, int(round(start_sec * src_fps)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        output_fps,
        (out_width, out_height),
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create proxy video: {output_path}")

    source_frame_index = start_frame
    saved_frames = 0

    print("")
    print("Creating lightweight 1 FPS tracking proxy...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if (source_frame_index - start_frame) % frame_step == 0:
            if scale != 1.0:
                frame = cv2.resize(
                    frame,
                    (out_width, out_height),
                    interpolation=cv2.INTER_AREA,
                )

            writer.write(frame)
            saved_frames += 1

            if saved_frames % 300 == 0:
                print(f"Prepared {saved_frames} proxy frames...")

        source_frame_index += 1

    cap.release()
    writer.release()

    if saved_frames == 0:
        raise RuntimeError("Proxy video contains no frames.")

    print(f"Proxy video ready: {saved_frames} frames")
    print(f"Proxy path: {output_path}")

    return {
        "source_fps": src_fps,
        "start_frame": start_frame,
        "start_sec": float(start_sec),
        "proxy_fps": float(output_fps),
        "proxy_frames": saved_frames,
        "width": out_width,
        "height": out_height,
    }


def select_target(video_path):
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError("Failed to read the first proxy frame.")

    height, width = frame.shape[:2]
    display_scale = min(1.0, 1200 / width, 700 / height)

    if display_scale < 1:
        display = cv2.resize(
            frame,
            None,
            fx=display_scale,
            fy=display_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        display = frame.copy()

    print("")
    print("Select the ROAD ROLLER only.")
    print("Draw a box around the machine, then press ENTER or SPACE.")
    print("")

    roi = cv2.selectROI(
        "Select Road Roller - ENTER confirm",
        display,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyAllWindows()

    x, y, w, h = roi

    if w == 0 or h == 0:
        raise RuntimeError("No road roller was selected.")

    x1 = int(x / display_scale)
    y1 = int(y / display_scale)
    x2 = int((x + w) / display_scale)
    y2 = int((y + h) / display_scale)

    return [x1, y1, x2, y2]


def get_ground_point(result):
    if result.masks is None:
        return None

    masks = result.masks.data
    if masks is None or len(masks) == 0:
        return None

    mask = masks[0].detach().cpu().numpy()

    frame_height, frame_width = result.orig_img.shape[:2]

    if mask.shape != (frame_height, frame_width):
        mask = cv2.resize(
            mask,
            (frame_width, frame_height),
            interpolation=cv2.INTER_NEAREST,
        )

    ys, xs = np.where(mask > 0.5)

    if len(xs) < 20:
        return None

    bottom_threshold = np.percentile(ys, 85)
    bottom_region = ys >= bottom_threshold
    bottom_xs = xs[bottom_region]
    bottom_ys = ys[bottom_region]

    x = int(np.median(bottom_xs))
    y = int(np.percentile(bottom_ys, 90))

    return x, y


def smooth_point(history, point):
    history.append(point)
    xs = [p[0] for p in history]
    ys = [p[1] for p in history]
    return int(np.mean(xs)), int(np.mean(ys))


def interpolate_short_tracking_gaps(
    csv_rows,
    proxy_fps,
    max_gap_seconds=MAX_INTERPOLATION_GAP_SECONDS,
):
    """Fill only short, bounded tracking gaps by linear XY interpolation.

    A gap is filled only when:
    - it is between two valid tracked points; and
    - its duration is <= max_gap_seconds.

    Longer gaps, leading gaps, and trailing gaps remain unchanged.
    The existing CSV schema is preserved.
    """
    if not csv_rows:
        return 0

    max_gap_frames = max(1, int(round(float(proxy_fps) * max_gap_seconds)))
    filled = 0
    i = 0

    while i < len(csv_rows):
        if int(csv_rows[i][6]) == 1:
            i += 1
            continue

        gap_start = i
        while i < len(csv_rows) and int(csv_rows[i][6]) == 0:
            i += 1

        gap_end = i - 1
        gap_length = gap_end - gap_start + 1
        left_idx = gap_start - 1
        right_idx = i

        bounded = left_idx >= 0 and right_idx < len(csv_rows)

        if (
            bounded
            and gap_length <= max_gap_frames
            and int(csv_rows[left_idx][6]) == 1
            and int(csv_rows[right_idx][6]) == 1
        ):
            x0 = float(csv_rows[left_idx][4])
            y0 = float(csv_rows[left_idx][5])
            x1 = float(csv_rows[right_idx][4])
            y1 = float(csv_rows[right_idx][5])
            span = right_idx - left_idx

            for row_idx in range(gap_start, right_idx):
                alpha = (row_idx - left_idx) / span
                csv_rows[row_idx][4] = int(round(x0 + alpha * (x1 - x0)))
                csv_rows[row_idx][5] = int(round(y0 + alpha * (y1 - y0)))
                csv_rows[row_idx][6] = 1
                filled += 1

    return filled


def save_run_config(run_dir, args, start_sec, proxy_meta):
    config = {
        "video_path": str(Path(args.video).resolve()),
        "sections": int(args.sections),
        "lengths_m": [float(v) for v in args.lengths],
        "section_order": args.section_order,
        "completion_ratio": float(args.completion_ratio),
        "speed_limit_kmh": float(args.speed_limit),
        "timestamp_roi": list(DEFAULT_TIMESTAMP_ROI),
        "tracking_start_sec": float(start_sec),
        "target_fps": TARGET_FPS,
        "proxy_meta": proxy_meta,
    }

    config_path = Path(run_dir) / "run_config.json"
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return config_path


def run_tracking(args):
    validate_args(args)

    video_path = Path(args.video).resolve()
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / safe_stem(video_path)
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.start_sec is None:
        start_sec = choose_tracking_start(video_path)
    else:
        start_sec = float(args.start_sec)
        print(f"Using --start-sec: {start_sec:.1f} sec")

    proxy_path = run_dir / "tracking_proxy.mp4"

    proxy_meta = create_proxy_video(
        video_path,
        proxy_path,
        start_sec,
    )

    bbox = select_target(proxy_path)
    print(f"Selected road roller box: {bbox}")

    config_path = save_run_config(
        run_dir,
        args,
        start_sec,
        proxy_meta,
    )
    print(f"Run config: {config_path}")

    device = 0 if torch.cuda.is_available() else "cpu"

    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("Using CPU")

    overrides = {
        "conf": 0.01,
        "task": "segment",
        "mode": "predict",
        "imgsz": 1024,
        "model": "sam2.1_t.pt",
        "device": device,
        "save": False,
        "verbose": False,
    }

    print("Loading SAM 2.1 Tiny...")
    predictor = SAM2VideoPredictor(overrides=overrides)

    cap = cv2.VideoCapture(str(proxy_path))
    proxy_fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if proxy_fps <= 0:
        proxy_fps = TARGET_FPS

    output_video = run_dir / "sam2_tracked_video.mp4"
    writer = cv2.VideoWriter(
        str(output_video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        proxy_fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to create tracked video: {output_video}")

    results = predictor(
        source=str(proxy_path),
        bboxes=bbox,
        stream=True,
    )

    smooth_history = deque(maxlen=SMOOTH_WINDOW)
    trajectory = []
    csv_rows = []
    trail = np.zeros((height, width, 3), dtype=np.uint8)

    previous_point = None
    total_frames = 0
    tracked_frames = 0

    source_fps = float(proxy_meta["source_fps"])
    start_source_frame = int(proxy_meta["start_frame"])
    frame_step = max(1, round(source_fps / TARGET_FPS))

    print("")
    print("SAM2 tracking started...")

    try:
        for frame_index, result in enumerate(results):
            total_frames += 1
            frame = result.orig_img.copy()

            point = get_ground_point(result)

            if point is not None:
                tracked_frames += 1
                point = smooth_point(smooth_history, point)
                trajectory.append(point)

                if previous_point is not None:
                    cv2.line(
                        trail,
                        previous_point,
                        point,
                        (255, 0, 0),
                        2,
                    )

                previous_point = point

                cv2.circle(
                    frame,
                    point,
                    5,
                    (0, 0, 255),
                    -1,
                )

                x_value = point[0]
                y_value = point[1]
                tracked = 1
            else:
                x_value = ""
                y_value = ""
                tracked = 0

            relative_time_sec = frame_index / proxy_fps
            source_frame = start_source_frame + frame_index * frame_step
            source_time_sec = source_frame / source_fps

            csv_rows.append(
                [
                    frame_index,
                    round(relative_time_sec, 3),
                    source_frame,
                    round(source_time_sec, 3),
                    x_value,
                    y_value,
                    tracked,
                ]
            )

            combined = cv2.add(frame, trail)

            cv2.putText(
                combined,
                f"Tracked: {tracked_frames}/{total_frames}",
                (20, 38),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            writer.write(combined)

            if total_frames % 100 == 0:
                success_rate = 100.0 * tracked_frames / total_frames
                print(
                    f"Frame {total_frames} | "
                    f"Tracking success: {success_rate:.1f}%"
                )
    finally:
        writer.release()

    interpolated_frames = interpolate_short_tracking_gaps(
        csv_rows,
        proxy_fps,
    )
    repaired_tracked_frames = sum(int(row[6]) == 1 for row in csv_rows)

    trajectory_path = run_dir / "trajectory.csv"

    with open(
        trajectory_path,
        "w",
        newline="",
        encoding="utf-8-sig",
    ) as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(
            [
                "frame",
                "relative_time_sec",
                "source_frame",
                "source_time_sec",
                "x",
                "y",
                "tracked",
            ]
        )
        csv_writer.writerows(csv_rows)

    raw_success_rate = (
        100.0 * tracked_frames / total_frames
        if total_frames > 0
        else 0.0
    )
    repaired_success_rate = (
        100.0 * repaired_tracked_frames / total_frames
        if total_frames > 0
        else 0.0
    )

    print("")
    print("Tracking complete.")
    print(f"Raw tracking success: {raw_success_rate:.1f}%")
    print(f"Interpolated short-gap frames: {interpolated_frames}")
    print(f"Tracking success after interpolation: {repaired_success_rate:.1f}%")
    print(f"Run folder: {run_dir}")
    print(f"Trajectory CSV: {trajectory_path}")
    print(f"Tracked video: {output_video}")

    return run_dir


def run_analysis(run_dir):
    script_path = Path(__file__).with_name("analyze_compaction.py")

    if not script_path.exists():
        print("")
        print("Tracking finished, but analyze_compaction.py was not found.")
        print(f"Expected: {script_path}")
        print(
            f'Run analysis later with: python "{script_path}" '
            f'--run-dir "{run_dir}"'
        )
        return

    print("")
    print("Starting section / speed / OCR analysis...")

    subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--run-dir",
            str(run_dir),
        ],
        check=True,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Track a road roller with SAM2, then automatically "
            "analyze sections, completion, timestamp OCR, speed, "
            "effective passes, Excel tables and QA overlay."
        )
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Full path to the source video.",
    )
    parser.add_argument(
        "--sections",
        required=True,
        type=int,
        help="Expected number of compaction sections.",
    )
    parser.add_argument(
        "--lengths",
        required=True,
        nargs="+",
        type=float,
        help="Section lengths in metres, one value per section.",
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help=(
            "Optional tracking start time in seconds. "
            "If omitted, an interactive timeline is shown."
        ),
    )
    parser.add_argument(
        "--section-order",
        choices=["left-to-right", "right-to-left"],
        default="left-to-right",
        help="How automatically clustered sections are numbered.",
    )
    parser.add_argument(
        "--completion-ratio",
        type=float,
        default=DEFAULT_COMPLETION_RATIO,
        help="Minimum fraction of section traversal required to count as complete.",
    )
    parser.add_argument(
        "--speed-limit",
        type=float,
        default=DEFAULT_SPEED_LIMIT_KMH,
        help="Maximum valid compaction speed in km/h.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "output"),
        help="Root folder for per-video outputs.",
    )

    args = parser.parse_args()

    run_dir = run_tracking(args)
    run_analysis(run_dir)


if __name__ == "__main__":
    main()
