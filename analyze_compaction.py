import argparse
import csv
import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np


DEFAULT_COMPLETION_RATIO = 0.80
DEFAULT_SPEED_LIMIT_KMH = 2.0
DEFAULT_MIN_POINTS_PER_PASS = 8
DEFAULT_MIN_TRAVEL_RATIO = 0.18
DEFAULT_SECTION_REVIEW_CONFIDENCE = 0.35
DEFAULT_TIMESTAMP_ROI = (0.64, 0.86, 1.00, 1.00)  # normalized x1, y1, x2, y2

# Section-local pass refinement. The original global pass segmentation is retained
# as a coarse first stage for section assignment, then each consecutive section
# block is re-segmented with a local scale so distant/small trajectories are not
# suppressed by the larger perspective scale of nearer sections.
DEFAULT_LOCAL_MIN_POINTS_PER_PASS = 5
DEFAULT_LOCAL_MIN_TRAVEL_RATIO = 0.08
DEFAULT_LOCAL_PROJECTION_SMOOTH = 5
DEFAULT_LOCAL_DELTA_SMOOTH = 5
DEFAULT_LOCAL_EPS_RATIO = 0.003
DEFAULT_LOCAL_EPS_PX = 0.5
DEFAULT_SECTION_REFERENCE_PERCENTILE = 90.0

# Stationary-time handling for speed calculation.
DEFAULT_STATIONARY_MIN_SECONDS = 8.0
DEFAULT_STATIONARY_TRACKED_RATIO = 0.015
DEFAULT_STATIONARY_GAP_RATIO = 0.03
DEFAULT_STATIONARY_MIN_PX = 2.0


# ----------------------------
# Generic helpers
# ----------------------------

def moving_average(values, window):
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or window <= 1:
        return values.copy()
    window = min(int(window), len(values))
    if window <= 1:
        return values.copy()
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    smoothed = np.convolve(padded, kernel, mode="valid")
    return smoothed[: len(values)]


def smooth_points(points, window=7):
    if len(points) == 0:
        return points.copy()
    xs = moving_average(points[:, 0], window)
    ys = moving_average(points[:, 1], window)
    return np.column_stack([xs, ys])


def principal_axis(points):
    center = points.mean(axis=0)
    centered = points - center
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    start_mean = points[: min(10, len(points))].mean(axis=0)
    end_mean = points[-min(10, len(points)) :].mean(axis=0)
    if np.dot(end_mean - start_mean, axis) < 0:
        axis = -axis
    perp = np.array([-axis[1], axis[0]], dtype=float)
    # Orient perpendicular axis so positive roughly means image-right.
    if perp[0] < 0:
        perp = -perp
    return center, axis, perp


def project(points, center, axis):
    return np.dot(points - center, axis)


def fill_zero_signs(signs):
    signs = signs.copy()
    last = 0
    for i in range(len(signs)):
        if signs[i] != 0:
            last = signs[i]
        elif last != 0:
            signs[i] = last
    last = 0
    for i in range(len(signs) - 1, -1, -1):
        if signs[i] != 0:
            last = signs[i]
        elif last != 0:
            signs[i] = last
    return signs


def build_runs(signs):
    if len(signs) == 0:
        return []
    runs = []
    start = 0
    current = int(signs[0])
    for i in range(1, len(signs)):
        if int(signs[i]) != current:
            runs.append([start, i - 1, current])
            start = i
            current = int(signs[i])
    runs.append([start, len(signs) - 1, current])
    return runs


def merge_adjacent_same_sign(runs):
    if not runs:
        return []
    merged = [runs[0][:]]
    for start, end, sign in runs[1:]:
        if sign == merged[-1][2]:
            merged[-1][1] = end
        else:
            merged.append([start, end, sign])
    return merged


# ----------------------------
# Input loading / pass segmentation
# ----------------------------

def load_trajectory(csv_path):
    rows = []
    with open(csv_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if str(row.get("tracked", "")).strip() != "1":
                continue
            if row.get("x", "") == "" or row.get("y", "") == "":
                continue
            rows.append(
                {
                    "frame": int(float(row.get("frame", 0))),
                    "relative_time_sec": float(row.get("relative_time_sec", row.get("time_sec", 0.0))),
                    "source_frame": int(float(row.get("source_frame", row.get("frame", 0)))),
                    "source_time_sec": float(row.get("source_time_sec", row.get("time_sec", 0.0))),
                    "x": float(row["x"]),
                    "y": float(row["y"]),
                }
            )
    if not rows:
        raise RuntimeError("No tracked points found in trajectory.csv")
    points = np.array([[r["x"], r["y"]] for r in rows], dtype=float)
    return rows, points


def segment_passes(points, projections, min_points_per_pass, min_travel_ratio):
    total_range = float(np.max(projections) - np.min(projections))
    if total_range < 1:
        raise RuntimeError("Trajectory range is too small to segment passes")

    delta = np.diff(projections, prepend=projections[0])
    delta = moving_average(delta, 5)
    eps = max(1.0, total_range * 0.005)
    signs = np.zeros(len(delta), dtype=int)
    signs[delta > eps] = 1
    signs[delta < -eps] = -1
    signs = fill_zero_signs(signs)

    runs = [r for r in build_runs(signs) if r[2] != 0]
    runs = merge_adjacent_same_sign(runs)
    min_travel = total_range * min_travel_ratio

    # Iteratively absorb short/noisy direction reversals.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        new_runs = []
        i = 0
        while i < len(runs):
            start, end, sign = runs[i]
            num_points = end - start + 1
            travel = abs(projections[end] - projections[start])
            too_small = num_points < min_points_per_pass or travel < min_travel
            if too_small:
                changed = True
                if i == 0 and i + 1 < len(runs):
                    runs[i + 1][0] = start
                elif new_runs:
                    new_runs[-1][1] = end
                i += 1
                continue
            new_runs.append([start, end, sign])
            i += 1
        runs = merge_adjacent_same_sign(new_runs)

    passes = []
    for pass_id, (start, end, sign) in enumerate(runs, start=1):
        if end <= start:
            continue
        passes.append(
            {
                "pass_id": pass_id,
                "start_idx": int(start),
                "end_idx": int(end),
                "direction": "forward" if sign > 0 else "backward",
                "sign": int(sign),
                "projected_distance_px": float(abs(projections[end] - projections[start])),
            }
        )
    return passes, total_range


# ----------------------------
# Section clustering
# ----------------------------

def pass_lateral_feature(pass_points, center, perp_axis):
    lateral = np.dot(pass_points - center, perp_axis)
    if len(lateral) == 1:
        return np.repeat(lateral[0], 5)
    q = [10, 30, 50, 70, 90]
    return np.percentile(lateral, q).astype(float)


def kmeans_numpy(features, k, max_iter=100):
    features = np.asarray(features, dtype=float)
    n = len(features)
    if k < 1 or k > n:
        raise ValueError(f"Invalid section count {k} for {n} passes")
    if k == 1:
        return np.zeros(n, dtype=int), np.mean(features, axis=0, keepdims=True)

    order = np.argsort(np.median(features, axis=1))
    seeds = np.linspace(0, n - 1, k).round().astype(int)
    centers = features[order[seeds]].copy()
    labels = np.zeros(n, dtype=int)

    for _ in range(max_iter):
        dists = np.linalg.norm(features[:, None, :] - centers[None, :, :], axis=2)
        new_labels = np.argmin(dists, axis=1)
        new_centers = centers.copy()
        for cluster_id in range(k):
            members = features[new_labels == cluster_id]
            if len(members) > 0:
                new_centers[cluster_id] = members.mean(axis=0)
        if np.array_equal(new_labels, labels) and np.allclose(new_centers, centers):
            labels = new_labels
            centers = new_centers
            break
        labels = new_labels
        centers = new_centers
    return labels, centers


def assign_sections(passes, points, center, perp_axis, section_count, section_order):
    features = []
    for p in passes:
        pts = points[p["start_idx"] : p["end_idx"] + 1]
        features.append(pass_lateral_feature(pts, center, perp_axis))

    labels, centers = kmeans_numpy(np.asarray(features), section_count)
    center_lateral = np.median(centers, axis=1)
    cluster_order = np.argsort(center_lateral)
    if section_order == "right-to-left":
        cluster_order = cluster_order[::-1]

    cluster_to_section = {int(cluster_id): idx + 1 for idx, cluster_id in enumerate(cluster_order)}

    for idx, p in enumerate(passes):
        cluster_id = int(labels[idx])
        p["section"] = cluster_to_section[cluster_id]

        if section_count == 1:
            confidence = 1.0
        else:
            d = np.linalg.norm(np.asarray(features[idx])[None, :] - centers, axis=1)
            ordered = np.sort(d)
            assigned = ordered[0]
            second = ordered[1]
            confidence = 1.0 - assigned / max(second, 1e-6)
            confidence = float(np.clip(confidence, 0.0, 1.0))
        p["section_confidence"] = confidence

    return passes, centers, cluster_to_section


def build_section_models(provisional_passes, points):
    """Build one local travel axis for each automatically assigned section."""
    models = {}
    section_ids = sorted({int(p["section"]) for p in provisional_passes})

    for section_id in section_ids:
        indices = []
        for p in provisional_passes:
            if int(p["section"]) != section_id:
                continue
            indices.extend(range(int(p["start_idx"]), int(p["end_idx"]) + 1))

        if not indices:
            continue

        indices = sorted(set(indices))
        section_points = points[indices]
        center, axis, _ = principal_axis(section_points)
        models[section_id] = {
            "center": center,
            "axis": axis,
        }

    return models


def group_consecutive_section_blocks(provisional_passes):
    """Group chronological coarse passes into consecutive blocks of the same section."""
    if not provisional_passes:
        return []

    ordered = sorted(provisional_passes, key=lambda p: int(p["start_idx"]))
    blocks = [[ordered[0]]]

    for p in ordered[1:]:
        if int(p["section"]) == int(blocks[-1][-1]["section"]):
            blocks[-1].append(p)
        else:
            blocks.append([p])

    return blocks


def segment_local_projection(
    local_projection,
    min_points_per_pass=DEFAULT_LOCAL_MIN_POINTS_PER_PASS,
    min_travel_ratio=DEFAULT_LOCAL_MIN_TRAVEL_RATIO,
):
    """Segment one section block using that block's own projection scale.

    This is intentionally more sensitive than the coarse global segmentation.
    It is used only after section assignment, which prevents a distant section's
    smaller pixel motion from being swallowed by the perspective scale of a
    nearer section.
    """
    local_projection = np.asarray(local_projection, dtype=float)
    if len(local_projection) < 2:
        return []

    total_range = float(np.max(local_projection) - np.min(local_projection))
    if total_range < 1.0:
        return [[0, len(local_projection) - 1, 1]]

    delta = np.diff(local_projection, prepend=local_projection[0])
    delta = moving_average(delta, DEFAULT_LOCAL_DELTA_SMOOTH)

    eps = max(DEFAULT_LOCAL_EPS_PX, total_range * DEFAULT_LOCAL_EPS_RATIO)
    signs = np.zeros(len(delta), dtype=int)
    signs[delta > eps] = 1
    signs[delta < -eps] = -1
    signs = fill_zero_signs(signs)

    runs = [r for r in build_runs(signs) if r[2] != 0]
    runs = merge_adjacent_same_sign(runs)

    min_travel = total_range * float(min_travel_ratio)

    # Absorb only very small local reversals. Because the threshold is section
    # local, a real turn in a distant section is no longer compared with the
    # much larger pixel span of a near-camera section.
    changed = True
    while changed and len(runs) > 1:
        changed = False
        new_runs = []
        i = 0

        while i < len(runs):
            start, end, sign = runs[i]
            num_points = end - start + 1
            travel = abs(local_projection[end] - local_projection[start])
            too_small = (
                num_points < int(min_points_per_pass)
                or travel < min_travel
            )

            if too_small:
                changed = True
                if i == 0 and i + 1 < len(runs):
                    runs[i + 1][0] = start
                elif new_runs:
                    new_runs[-1][1] = end
                i += 1
                continue

            new_runs.append([start, end, sign])
            i += 1

        runs = merge_adjacent_same_sign(new_runs)

    return runs


def refine_passes_by_section_blocks(
    provisional_passes,
    points,
    section_models,
):
    """Re-segment consecutive section blocks using section-local geometry.

    Single isolated transition passes are left intact to avoid creating extra
    fragments at the very end of a video. Main working blocks containing two
    or more coarse passes are refined.
    """
    refined = []

    for block in group_consecutive_section_blocks(provisional_passes):
        section_id = int(block[0]["section"])
        start_idx = int(block[0]["start_idx"])
        end_idx = int(block[-1]["end_idx"])
        confidence = float(np.median([p["section_confidence"] for p in block]))

        model = section_models.get(section_id)
        if model is None:
            for p in block:
                refined.append(dict(p))
            continue

        center = model["center"]
        axis = model["axis"]
        block_points = points[start_idx : end_idx + 1]
        local_projection = moving_average(
            project(block_points, center, axis),
            DEFAULT_LOCAL_PROJECTION_SMOOTH,
        )

        # For a single isolated coarse pass, keep the original boundaries.
        # Its travel distance is still measured on the local section axis.
        if len(block) == 1:
            sign = 1 if local_projection[-1] >= local_projection[0] else -1
            local_runs = [[0, len(local_projection) - 1, sign]]
        else:
            local_runs = segment_local_projection(local_projection)

        for local_start, local_end, sign in local_runs:
            if local_end <= local_start:
                continue

            abs_start = start_idx + int(local_start)
            abs_end = start_idx + int(local_end)
            travel = float(
                abs(local_projection[local_end] - local_projection[local_start])
            )

            refined.append(
                {
                    "pass_id": 0,  # renumbered chronologically below
                    "start_idx": abs_start,
                    "end_idx": abs_end,
                    "direction": "forward" if sign > 0 else "backward",
                    "sign": int(sign),
                    "projected_distance_px": travel,
                    "section": section_id,
                    "section_confidence": confidence,
                }
            )

    refined.sort(key=lambda p: int(p["start_idx"]))
    for pass_id, p in enumerate(refined, start=1):
        p["pass_id"] = pass_id

    return refined


def compute_section_completion(
    passes,
    section_lengths,
    completion_threshold,
):
    """Compute completion with a section-local reference span.

    The reference for each section is the 90th percentile of pass travel on that
    section's own axis. This is robust to short transition fragments and avoids
    using one global pixel scale for all sections.
    """
    section_reference_spans = {}

    for section_id, length_m in enumerate(section_lengths, start=1):
        section_passes = [p for p in passes if int(p["section"]) == section_id]
        if not section_passes:
            continue

        distances = np.asarray(
            [
                float(p.get("projected_distance_px", 0.0))
                for p in section_passes
                if float(p.get("projected_distance_px", 0.0)) > 0
            ],
            dtype=float,
        )

        if len(distances) == 0:
            reference_span = 1.0
        elif len(distances) == 1:
            reference_span = float(distances[0])
        else:
            reference_span = float(
                np.percentile(distances, DEFAULT_SECTION_REFERENCE_PERCENTILE)
            )

        reference_span = max(reference_span, 1e-6)
        section_reference_spans[section_id] = reference_span

        for p in section_passes:
            travel = float(p.get("projected_distance_px", 0.0))
            ratio = float(np.clip(travel / reference_span, 0.0, 1.0))
            p["completion_ratio"] = ratio
            p["estimated_covered_length_m"] = ratio * float(length_m)
            p["completed"] = bool(ratio >= completion_threshold)
            p["length_m"] = float(length_m)
            p["section_reference_span_px"] = reference_span

    return passes, section_reference_spans


def _merge_time_intervals(intervals):
    if not intervals:
        return []

    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return []

    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1] + 1e-6:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return [(a, b) for a, b in merged]


def estimate_stationary_time(
    rows,
    points,
    start_idx,
    end_idx,
    section_reference_span_px,
):
    """Estimate long stops inside one pass.

    Two cases are handled:
    1) the roller remains tracked but its 2-D position stays inside a very small
       area for at least DEFAULT_STATIONARY_MIN_SECONDS;
    2) tracking disappears for a long gap and the points immediately before and
       after the gap are still at nearly the same position.

    A long tracking gap whose endpoints are far apart is NOT assumed to be a
    stop. It is returned as unresolved_gap_sec so speed can be withheld for
    manual review rather than fabricated.
    """
    start_idx = int(start_idx)
    end_idx = int(end_idx)

    if end_idx <= start_idx:
        return {
            "stationary_time_sec": 0.0,
            "unresolved_gap_sec": 0.0,
            "stationary_intervals": [],
        }

    times = np.asarray(
        [float(rows[i]["source_time_sec"]) for i in range(start_idx, end_idx + 1)],
        dtype=float,
    )
    xy = np.asarray(points[start_idx : end_idx + 1], dtype=float)

    if len(times) < 2:
        return {
            "stationary_time_sec": 0.0,
            "unresolved_gap_sec": 0.0,
            "stationary_intervals": [],
        }

    dt = np.diff(times)
    positive_dt = dt[dt > 0]
    nominal_dt = float(np.median(positive_dt)) if len(positive_dt) else 1.0

    tracked_stop_threshold_px = max(
        DEFAULT_STATIONARY_MIN_PX,
        float(section_reference_span_px) * DEFAULT_STATIONARY_TRACKED_RATIO,
    )
    long_gap_same_place_threshold_px = max(
        DEFAULT_STATIONARY_MIN_PX,
        float(section_reference_span_px) * DEFAULT_STATIONARY_GAP_RATIO,
    )

    stationary_intervals = []
    unresolved_gap_sec = 0.0

    # Long tracking gaps. If the roller reappears essentially in the same place,
    # treat only the extra missing time beyond one normal frame interval as stop.
    for i, gap_sec in enumerate(dt):
        if gap_sec < DEFAULT_STATIONARY_MIN_SECONDS:
            continue

        displacement = float(np.linalg.norm(xy[i + 1] - xy[i]))
        extra_gap = max(0.0, float(gap_sec) - nominal_dt)

        if displacement <= long_gap_same_place_threshold_px:
            interval_start = float(times[i] + nominal_dt / 2.0)
            interval_end = float(times[i + 1] - nominal_dt / 2.0)
            if interval_end > interval_start:
                stationary_intervals.append((interval_start, interval_end))
        else:
            unresolved_gap_sec += extra_gap

    # Tracked stationary periods. Use a conservative 2-D spatial envelope so a
    # slow turn/repositioning near an endpoint is not mistaken for "not moving".
    candidate_intervals = []
    n = len(times)

    for i in range(n - 1):
        min_x = max_x = float(xy[i, 0])
        min_y = max_y = float(xy[i, 1])

        for j in range(i + 1, n):
            # Do not bridge a long tracking gap here; it was handled above.
            if times[j] - times[j - 1] >= DEFAULT_STATIONARY_MIN_SECONDS:
                break

            min_x = min(min_x, float(xy[j, 0]))
            max_x = max(max_x, float(xy[j, 0]))
            min_y = min(min_y, float(xy[j, 1]))
            max_y = max(max_y, float(xy[j, 1]))

            envelope = math.hypot(max_x - min_x, max_y - min_y)
            if envelope > tracked_stop_threshold_px:
                break

            duration = float(times[j] - times[i])
            if duration >= DEFAULT_STATIONARY_MIN_SECONDS:
                candidate_intervals.append((float(times[i]), float(times[j])))

    # Candidate windows are deliberately not chained across slow creeping
    # movement. Merge only direct overlaps in time; the conservative spatial
    # threshold above keeps these windows limited to near-static periods.
    stationary_intervals.extend(candidate_intervals)
    stationary_intervals = _merge_time_intervals(stationary_intervals)

    stationary_time_sec = float(
        sum(end - start for start, end in stationary_intervals)
    )

    return {
        "stationary_time_sec": stationary_time_sec,
        "unresolved_gap_sec": float(unresolved_gap_sec),
        "stationary_intervals": stationary_intervals,
    }

# ----------------------------
# Timestamp OCR
# ----------------------------

def parse_time_text(text):
    if not text:
        return None
    normalized = text.replace("：", ":").replace(".", ":")
    matches = re.findall(r"(?<!\d)(\d{1,2})\s*:\s*(\d{2})\s*:\s*(\d{2})(?!\d)", normalized)
    valid = []
    for hh, mm, ss in matches:
        h, m, s = int(hh), int(mm), int(ss)
        if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
            valid.append((h, m, s))
    if valid:
        h, m, s = valid[-1]
        return f"{h:02d}:{m:02d}:{s:02d}"

    digits = re.sub(r"\D", "", normalized)
    if len(digits) >= 6:
        candidate = digits[-6:]
        h, m, s = int(candidate[:2]), int(candidate[2:4]), int(candidate[4:6])
        if 0 <= h <= 23 and 0 <= m <= 59 and 0 <= s <= 59:
            return f"{h:02d}:{m:02d}:{s:02d}"
    return None


def shift_hms(time_str, seconds_delta):
    dt = datetime.strptime(time_str, "%H:%M:%S") + timedelta(seconds=float(seconds_delta))
    return dt.strftime("%H:%M:%S")


def hms_to_seconds(time_str):
    h, m, s = [int(v) for v in time_str.split(":")]
    return h * 3600 + m * 60 + s


def duration_between_hms(start_time, end_time):
    a = hms_to_seconds(start_time)
    b = hms_to_seconds(end_time)
    if b < a:
        b += 24 * 3600
    return b - a


class TimestampReader:
    def __init__(self, video_path, roi_norm, debug_dir):
        self.video_path = str(video_path)
        self.roi_norm = tuple(float(v) for v in roi_norm)
        self.debug_dir = Path(debug_dir)
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        try:
            import easyocr
        except ImportError as exc:
            raise RuntimeError(
                "easyocr is required for timestamp reading. Run: python -m pip install easyocr"
            ) from exc
        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open original video for OCR: {self.video_path}")

    def close(self):
        self.cap.release()

    def _read_frame(self, time_sec):
        self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, float(time_sec)) * 1000.0)
        ok, frame = self.cap.read()
        return frame if ok else None

    def _crop(self, frame):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = self.roi_norm
        xa, ya = int(w * x1), int(h * y1)
        xb, yb = int(w * x2), int(h * y2)
        return frame[ya:yb, xa:xb].copy()

    @staticmethod
    def _variants(crop):
        if crop is None or crop.size == 0:
            return []
        variants = [crop]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
        variants.append(gray)
        eq = cv2.equalizeHist(gray)
        variants.append(eq)
        _, th = cv2.threshold(eq, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(th)
        variants.append(cv2.bitwise_not(th))
        return variants

    def read_time(self, source_time_sec, debug_name=None):
        offsets = [0, -1, 1, -2, 2]
        attempts = []
        for offset in offsets:
            frame = self._read_frame(source_time_sec + offset)
            if frame is None:
                continue
            crop = self._crop(frame)
            for variant_id, variant in enumerate(self._variants(crop)):
                results = self.reader.readtext(
                    variant,
                    detail=0,
                    paragraph=False,
                    allowlist="0123456789:",
                )
                text = " ".join(results)
                parsed = parse_time_text(text)
                attempts.append({"offset": offset, "variant": variant_id, "text": text, "parsed": parsed})
                if parsed:
                    corrected = shift_hms(parsed, -offset)
                    if debug_name:
                        cv2.imwrite(str(self.debug_dir / f"{debug_name}_ocr.png"), crop)
                    return corrected, text, offset, attempts
        if debug_name:
            frame = self._read_frame(source_time_sec)
            if frame is not None:
                cv2.imwrite(str(self.debug_dir / f"{debug_name}_ocr_failed.png"), self._crop(frame))
        return None, "", None, attempts


# ----------------------------
# Output tables / workbook
# ----------------------------

def write_csvs(run_dir, pass_rows, section_rows):
    pass_csv = run_dir / "pass_details.csv"
    section_csv = run_dir / "section_summary.csv"

    pass_headers = list(pass_rows[0].keys()) if pass_rows else []
    with open(pass_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=pass_headers)
        writer.writeheader()
        writer.writerows(pass_rows)

    section_headers = list(section_rows[0].keys()) if section_rows else []
    with open(section_csv, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=section_headers)
        writer.writeheader()
        writer.writerows(section_rows)

    return pass_csv, section_csv


def create_excel(run_dir, pass_rows, section_rows, speed_limit):
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError as exc:
        raise RuntimeError(
            "openpyxl is required for Excel output. Run: python -m pip install openpyxl"
        ) from exc

    wb = Workbook()
    ws = wb.active
    ws.title = "Pass Details"

    pass_headers = [
        "Pass ID",
        "Section",
        "Section Confidence",
        "Length (m)",
        "Estimated Covered Length (m)",
        "Completion Ratio",
        "Start Time",
        "End Time",
        "Elapsed Duration (s)",
        "Stationary Time (s)",
        "Moving Duration (s)",
        "Unobserved Gap (s)",
        "Speed (km/h)",
        "Completed",
        "Speed Valid",
        "Effective",
        "Direction",
        "Time Source",
        "Manual Review",
    ]
    ws.append(pass_headers)
    for row in pass_rows:
        ws.append([row.get(h, "") for h in pass_headers])

    header_fill = PatternFill("solid", fgColor="1F4E78")
    header_font = Font(color="FFFFFF", bold=True)
    thin = Side(style="thin", color="D9E1F2")
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=thin)

    red_fill = PatternFill("solid", fgColor="F4CCCC")
    gray_fill = PatternFill("solid", fgColor="E7E6E6")
    green_fill = PatternFill("solid", fgColor="D9EAD3")
    orange_fill = PatternFill("solid", fgColor="FCE5CD")

    for row_idx in range(2, ws.max_row + 1):
        completed = str(ws.cell(row_idx, 14).value)
        speed_valid = str(ws.cell(row_idx, 15).value)
        effective = str(ws.cell(row_idx, 16).value)
        manual_review = str(ws.cell(row_idx, 19).value)
        fill = None
        if completed == "N":
            fill = gray_fill
        elif speed_valid == "N":
            fill = red_fill
        elif manual_review == "Y":
            fill = orange_fill
        elif effective == "Y":
            fill = green_fill
        if fill:
            for col in range(1, ws.max_column + 1):
                ws.cell(row_idx, col).fill = fill

    ws.freeze_panes = "A2"
    widths = [10, 9, 19, 12, 28, 18, 12, 12, 19, 19, 18, 18, 15, 12, 13, 11, 12, 18, 16]
    for i, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(i)].width = width
    for row in ws.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    summary = wb.create_sheet("Section Summary")
    summary_headers = [
        "User Col 1",
        "User Col 2",
        "User Col 3",
        "Section",
        "Length (m)",
        "Start Time",
        "End Time",
        "Effective Passes",
        "Average Speed (km/h)",
        "User Col 10",
        "User Col 11",
    ]
    summary.append(summary_headers)
    for row in section_rows:
        summary.append([row.get(h, "") for h in summary_headers])

    for cell in summary[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = Border(bottom=thin)
    summary.freeze_panes = "A2"
    summary_widths = [14, 14, 14, 10, 12, 12, 12, 18, 22, 14, 14]
    for i, width in enumerate(summary_widths, start=1):
        summary.column_dimensions[get_column_letter(i)].width = width
    for row in summary.iter_rows():
        for cell in row:
            cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)

    # Small note outside the requested 11-column table.
    summary["M1"] = "Rules"
    summary["M1"].fill = header_fill
    summary["M1"].font = header_font
    summary["M2"] = "Completion threshold"
    summary["N2"] = ">=80%"
    summary["M3"] = "Speed limit"
    summary["N3"] = f"<= {speed_limit:.2f} km/h"
    summary["M4"] = "Average speed"
    summary["N4"] = "Effective passes only"
    summary["M5"] = "Speed duration"
    summary["N5"] = "Elapsed - stationary"
    summary["M6"] = "Stationary rule"
    summary["N6"] = f">= {DEFAULT_STATIONARY_MIN_SECONDS:g} s near-static"
    summary["M7"] = "Long unknown gap"
    summary["N7"] = "Review; no speed"
    summary.column_dimensions["M"].width = 22
    summary.column_dimensions["N"].width = 24

    xlsx_path = run_dir / "compaction_report.xlsx"
    wb.save(xlsx_path)
    return xlsx_path


# ----------------------------
# Section QA overlay
# ----------------------------

def section_color(section_id):
    palette = [
        (230, 126, 34),   # orange
        (52, 152, 219),   # blue
        (46, 204, 113),   # green
        (155, 89, 182),   # purple
        (241, 196, 15),   # yellow
        (26, 188, 156),   # teal
    ]
    return palette[(section_id - 1) % len(palette)]


def draw_polyline(img, pts, color, thickness=2):
    pts = np.asarray(pts, dtype=int)
    if len(pts) < 2:
        return
    cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, color, thickness, cv2.LINE_AA)


def create_section_overlay(run_dir, proxy_path, points, passes, section_lengths, stats):
    cap = cv2.VideoCapture(str(proxy_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None

    background = frame.copy()
    dark = np.zeros_like(background)
    background = cv2.addWeighted(background, 0.78, dark, 0.22, 0)
    hull_layer = background.copy()

    # Coarse section areas (convex hulls). Overlap is intentionally allowed.
    for section_id in range(1, len(section_lengths) + 1):
        section_pts = []
        for p in passes:
            if p["section"] == section_id:
                section_pts.extend(points[p["start_idx"] : p["end_idx"] + 1].tolist())
        if len(section_pts) >= 3:
            arr = np.asarray(section_pts, dtype=np.int32)
            hull = cv2.convexHull(arr)
            color = section_color(section_id)
            cv2.fillConvexPoly(hull_layer, hull, color)
            cv2.polylines(background, [hull], True, color, 2, cv2.LINE_AA)

    background = cv2.addWeighted(hull_layer, 0.12, background, 0.88, 0)

    for p in passes:
        pts = points[p["start_idx"] : p["end_idx"] + 1]
        if not p.get("completed", False):
            color = (150, 150, 150)
            thickness = 2
        elif p.get("speed_valid") is False:
            color = (0, 0, 255)
            thickness = 4
        elif p.get("speed_valid") is None:
            color = (0, 165, 255)
            thickness = 3
        else:
            color = section_color(p["section"])
            thickness = 3
        draw_polyline(background, pts, color, thickness)

    # Section labels at cluster centroids.
    for section_id, length_m in enumerate(section_lengths, start=1):
        section_pts = []
        for p in passes:
            if p["section"] == section_id:
                section_pts.extend(points[p["start_idx"] : p["end_idx"] + 1].tolist())
        if not section_pts:
            continue
        arr = np.asarray(section_pts, dtype=float)
        cx, cy = np.median(arr, axis=0).astype(int)
        text = f"Section {section_id}  |  {length_m:g} m"
        color = section_color(section_id)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        x = max(10, min(background.shape[1] - tw - 20, cx - tw // 2))
        y = max(35, min(background.shape[0] - 20, cy))
        cv2.rectangle(background, (x - 8, y - th - 10), (x + tw + 8, y + 7), (255, 255, 255), -1)
        cv2.rectangle(background, (x - 8, y - th - 10), (x + tw + 8, y + 7), color, 2)
        cv2.putText(background, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (25, 25, 25), 2, cv2.LINE_AA)

    panel_lines = [
        f"Sections: {len(section_lengths)}",
        f"Detected passes: {stats['detected']}",
        f"Effective passes: {stats['effective']}",
        f"Overspeed: {stats['overspeed']}",
        f"Incomplete: {stats['incomplete']}",
        f"OCR review: {stats['ocr_review']}",
    ]
    panel_w, panel_h = 280, 30 + 26 * len(panel_lines)
    overlay = background.copy()
    cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (0, 0, 0), -1)
    background = cv2.addWeighted(overlay, 0.62, background, 0.38, 0)
    for i, line in enumerate(panel_lines):
        cv2.putText(background, line, (30, 45 + i * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)

    legend_y = background.shape[0] - 55
    legend = [
        (section_color(1), "Section color = valid pass"),
        ((0, 0, 255), "Red = >2 km/h"),
        ((150, 150, 150), "Gray = incomplete"),
        ((0, 165, 255), "Orange = OCR/manual review"),
    ]
    x = 18
    for color, label in legend:
        cv2.line(background, (x, legend_y), (x + 28, legend_y), color, 4)
        cv2.putText(background, label, (x + 36, legend_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        x += 215

    out_path = run_dir / "section_assignment_overlay.png"
    cv2.imwrite(str(out_path), background)
    return out_path


# ----------------------------
# Main analysis
# ----------------------------

def run_analysis(run_dir, min_points_per_pass=DEFAULT_MIN_POINTS_PER_PASS, min_travel_ratio=DEFAULT_MIN_TRAVEL_RATIO):
    run_dir = Path(run_dir)
    config_path = run_dir / "run_config.json"
    trajectory_path = run_dir / "trajectory.csv"
    proxy_path = run_dir / "tracking_proxy.mp4"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not trajectory_path.exists():
        raise FileNotFoundError(f"Missing trajectory: {trajectory_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    video_path = Path(config["video_path"])
    section_count = int(config["sections"])
    section_lengths = [float(v) for v in config["lengths_m"]]
    section_order = config.get("section_order", "left-to-right")
    completion_threshold = float(config.get("completion_ratio", DEFAULT_COMPLETION_RATIO))
    speed_limit = float(config.get("speed_limit_kmh", DEFAULT_SPEED_LIMIT_KMH))
    timestamp_roi = tuple(config.get("timestamp_roi", DEFAULT_TIMESTAMP_ROI))

    rows, raw_points = load_trajectory(trajectory_path)
    points = smooth_points(raw_points, 7)
    center, axis, perp_axis = principal_axis(points)
    projections = moving_average(project(points, center, axis), 9)

    # Stage 1: keep the original global segmentation only as a coarse pass set
    # for section assignment.
    provisional_passes, total_range = segment_passes(
        points,
        projections,
        min_points_per_pass=min_points_per_pass,
        min_travel_ratio=min_travel_ratio,
    )
    if len(provisional_passes) < section_count:
        raise RuntimeError(
            f"Detected only {len(provisional_passes)} coarse passes, fewer than requested {section_count} sections."
        )

    provisional_passes, _, _ = assign_sections(
        provisional_passes,
        points,
        center,
        perp_axis,
        section_count=section_count,
        section_order=section_order,
    )

    # Stage 2: re-segment the main working block of each section using that
    # section's own pixel scale. This is the key protection against distant,
    # slow-moving passes being merged together.
    section_models = build_section_models(provisional_passes, points)
    passes = refine_passes_by_section_blocks(
        provisional_passes,
        points,
        section_models,
    )

    passes, section_reference_spans = compute_section_completion(
        passes,
        section_lengths,
        completion_threshold,
    )

    # Exclude incomplete paths from all official calculations and outputs.
    # Completion is evaluated first so the section reference span is still
    # learned from the full detected trajectory. Only paths meeting the
    # configured completion threshold (default 80%) proceed to OCR, speed,
    # section timing, pass tables, summaries and QA overlay.
    excluded_incomplete_passes = sum(
        1 for p in passes if not p.get("completed", False)
    )
    passes = [p for p in passes if p.get("completed", False)]

    # Renumber retained passes chronologically after incomplete paths are removed.
    passes.sort(key=lambda p: int(p["start_idx"]))
    for pass_id, p in enumerate(passes, start=1):
        p["pass_id"] = pass_id

    if not passes:
        raise RuntimeError(
            "No complete passes remain after applying the completion threshold."
        )

    ocr = TimestampReader(video_path, timestamp_roi, run_dir / "ocr_debug")
    pass_rows = []

    try:
        for p in passes:
            start_row = rows[p["start_idx"]]
            end_row = rows[p["end_idx"]]
            source_start = float(start_row["source_time_sec"])
            source_end = float(end_row["source_time_sec"])
            expected_duration = max(0.0, source_end - source_start)

            start_time, _, _, _ = ocr.read_time(source_start, f"P{p['pass_id']}_start")
            end_time, _, _, _ = ocr.read_time(source_end, f"P{p['pass_id']}_end")

            duration_sec = None
            time_source = "OCR"
            time_ok = start_time is not None and end_time is not None
            if time_ok:
                duration_sec = float(duration_between_hms(start_time, end_time))
                tolerance = max(3.0, expected_duration * 0.20)
                if duration_sec <= 0 or abs(duration_sec - expected_duration) > tolerance:
                    time_ok = False
                    time_source = "OCR_REVIEW"
            else:
                time_source = "OCR_FAIL"

            section_id = int(p["section"])
            reference_span_px = float(
                section_reference_spans.get(
                    section_id,
                    p.get("section_reference_span_px", 1.0),
                )
            )

            stationary_info = estimate_stationary_time(
                rows,
                points,
                p["start_idx"],
                p["end_idx"],
                reference_span_px,
            )
            stationary_time_sec = float(stationary_info["stationary_time_sec"])
            unresolved_gap_sec = float(stationary_info["unresolved_gap_sec"])

            moving_duration_sec = None
            if time_ok and duration_sec is not None:
                # Never allow estimated stop time to create zero/negative motion time.
                stationary_time_sec = min(
                    stationary_time_sec,
                    max(0.0, duration_sec - 0.001),
                )
                moving_duration_sec = max(
                    0.0,
                    float(duration_sec) - stationary_time_sec,
                )

            speed_kmh = None
            speed_valid = None
            effective = False

            # If a long tracking gap reappears far from where it disappeared,
            # the movement during the gap is unknown. Do not fabricate speed.
            motion_ok = unresolved_gap_sec <= 0.0

            if (
                p["completed"]
                and time_ok
                and motion_ok
                and moving_duration_sec is not None
                and moving_duration_sec > 0
            ):
                speed_kmh = float(
                    p["length_m"] / moving_duration_sec * 3.6
                )
                speed_valid = bool(speed_kmh <= speed_limit)
                effective = bool(speed_valid)

            manual_review = (
                p["section_confidence"] < DEFAULT_SECTION_REVIEW_CONFIDENCE
                or not time_ok
                or not motion_ok
            )

            p["start_time"] = start_time
            p["end_time"] = end_time
            p["duration_sec"] = duration_sec if time_ok else None
            p["stationary_time_sec"] = stationary_time_sec
            p["moving_duration_sec"] = moving_duration_sec
            p["unresolved_gap_sec"] = unresolved_gap_sec
            p["speed_kmh"] = speed_kmh
            p["speed_valid"] = speed_valid
            p["effective"] = effective
            p["manual_review"] = manual_review
            p["time_source"] = time_source
            p["source_start_sec"] = source_start
            p["source_end_sec"] = source_end

            pass_rows.append(
                {
                    "Pass ID": f"P{p['pass_id']}",
                    "Section": p["section"],
                    "Section Confidence": round(p["section_confidence"], 3),
                    "Length (m)": round(p["length_m"], 3),
                    "Estimated Covered Length (m)": round(p["estimated_covered_length_m"], 3),
                    "Completion Ratio": round(p["completion_ratio"], 3),
                    "Start Time": start_time or "",
                    "End Time": end_time or "",
                    "Elapsed Duration (s)": round(duration_sec, 1) if time_ok and duration_sec is not None else "",
                    "Stationary Time (s)": round(stationary_time_sec, 1),
                    "Moving Duration (s)": round(moving_duration_sec, 1) if moving_duration_sec is not None else "",
                    "Unobserved Gap (s)": round(unresolved_gap_sec, 1) if unresolved_gap_sec > 0 else "",
                    "Speed (km/h)": round(speed_kmh, 4) if speed_kmh is not None else "",
                    "Completed": "Y" if p["completed"] else "N",
                    "Speed Valid": "Y" if speed_valid is True else ("N" if speed_valid is False else ""),
                    "Effective": "Y" if effective else "N",
                    "Direction": p["direction"],
                    "Time Source": time_source,
                    "Manual Review": "Y" if manual_review else "N",
                }
            )
    finally:
        ocr.close()

    section_rows = []
    for section_id, length_m in enumerate(section_lengths, start=1):
        ps = [p for p in passes if p["section"] == section_id]
        ps_sorted = sorted(ps, key=lambda x: x["source_start_sec"])
        section_start = ""
        section_end = ""
        if ps_sorted:
            first_with_time = next((p for p in ps_sorted if p.get("start_time")), None)
            last_with_time = next((p for p in reversed(ps_sorted) if p.get("end_time")), None)
            if first_with_time:
                section_start = first_with_time["start_time"]
            if last_with_time:
                section_end = last_with_time["end_time"]

        effective_passes = [p for p in ps if p.get("effective")]
        avg_speed = (
            float(np.mean([p["speed_kmh"] for p in effective_passes]))
            if effective_passes
            else None
        )
        section_rows.append(
            {
                "User Col 1": "",
                "User Col 2": "",
                "User Col 3": "",
                "Section": section_id,
                "Length (m)": round(length_m, 3),
                "Start Time": section_start,
                "End Time": section_end,
                "Effective Passes": len(effective_passes),
                "Average Speed (km/h)": round(avg_speed, 4) if avg_speed is not None else "",
                "User Col 10": "",
                "User Col 11": "",
            }
        )

    pass_csv, section_csv = write_csvs(run_dir, pass_rows, section_rows)
    xlsx_path = create_excel(run_dir, pass_rows, section_rows, speed_limit)

    stats = {
        "detected": len(passes),
        "effective": sum(1 for p in passes if p.get("effective")),
        "overspeed": sum(1 for p in passes if p.get("speed_valid") is False),
        "incomplete": sum(1 for p in passes if not p.get("completed", False)),
        "ocr_review": sum(1 for p in passes if p.get("time_source") != "OCR"),
    }
    overlay_path = create_section_overlay(
        run_dir,
        proxy_path,
        points,
        passes,
        section_lengths,
        stats,
    )

    raw_pass_summary = run_dir / "pass_summary.csv"
    with open(raw_pass_summary, "w", newline="", encoding="utf-8-sig") as f:
        headers = [
            "pass_id",
            "section",
            "section_confidence",
            "direction",
            "start_source_sec",
            "end_source_sec",
            "elapsed_duration_sec",
            "stationary_time_sec",
            "moving_duration_sec",
            "unobserved_gap_sec",
            "completion_ratio",
            "completed",
            "speed_kmh",
            "speed_valid",
            "effective",
        ]
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for p in passes:
            writer.writerow(
                {
                    "pass_id": p["pass_id"],
                    "section": p["section"],
                    "section_confidence": round(p["section_confidence"], 4),
                    "direction": p["direction"],
                    "start_source_sec": round(p["source_start_sec"], 3),
                    "end_source_sec": round(p["source_end_sec"], 3),
                    "elapsed_duration_sec": "" if p.get("duration_sec") is None else round(p["duration_sec"], 3),
                    "stationary_time_sec": round(p.get("stationary_time_sec", 0.0), 3),
                    "moving_duration_sec": "" if p.get("moving_duration_sec") is None else round(p["moving_duration_sec"], 3),
                    "unobserved_gap_sec": round(p.get("unresolved_gap_sec", 0.0), 3),
                    "completion_ratio": round(p["completion_ratio"], 4),
                    "completed": int(p["completed"]),
                    "speed_kmh": "" if p["speed_kmh"] is None else round(p["speed_kmh"], 4),
                    "speed_valid": "" if p["speed_valid"] is None else int(p["speed_valid"]),
                    "effective": int(p["effective"]),
                }
            )

    print("")
    print("Analysis complete.")
    print(f"Complete passes retained: {stats['detected']}")
    print(f"Incomplete passes excluded: {excluded_incomplete_passes}")
    print(f"Effective passes: {stats['effective']}")
    print(f"Overspeed: {stats['overspeed']}")
    print(f"Incomplete: {stats['incomplete']}")
    print(f"OCR/manual review: {stats['ocr_review']}")
    print(f"Pass details CSV: {pass_csv}")
    print(f"Section summary CSV: {section_csv}")
    print(f"Excel report: {xlsx_path}")
    if overlay_path:
        print(f"Section overlay: {overlay_path}")
    return {
        "pass_csv": str(pass_csv),
        "section_csv": str(section_csv),
        "xlsx": str(xlsx_path),
        "overlay": str(overlay_path) if overlay_path else None,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze tracked roller passes and generate section/speed reports.")
    parser.add_argument("--run-dir", required=True, help="Per-video output folder created by track_roller.py")
    parser.add_argument("--min-points-per-pass", type=int, default=DEFAULT_MIN_POINTS_PER_PASS)
    parser.add_argument("--min-travel-ratio", type=float, default=DEFAULT_MIN_TRAVEL_RATIO)
    args = parser.parse_args()
    run_analysis(
        args.run_dir,
        min_points_per_pass=args.min_points_per_pass,
        min_travel_ratio=args.min_travel_ratio,
    )


if __name__ == "__main__":
    main()
