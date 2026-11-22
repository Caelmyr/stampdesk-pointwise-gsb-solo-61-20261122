"""Measure visible punch spacing along one marked edge of registered stock."""
import numpy as np
from PIL import ImageDraw
from scipy import ndimage, signal

from .art import base_rgba
from .model import InputError


def edge_track(frame, linear, available, start, end):
    extent = np.asarray(frame.extent, dtype=float)
    endpoints = np.asarray([start, end], dtype=float) * extent - .5
    tangent = endpoints[1] - endpoints[0]
    span = float(np.linalg.norm(tangent))
    if span < 36:
        raise InputError("Mark at least 36 image pixels along the perforation row.")
    stations = np.linspace(0, 1, int(span) + 1)
    line = endpoints[0] + stations[:, None] * tangent
    normal = np.asarray([-tangent[1], tangent[0]]) / span
    bands = line[:, None, :] + np.arange(-1, 2)[None, :, None] * normal
    coordinates = (bands[..., 1], bands[..., 0])
    luma = linear @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
    usable = ndimage.map_coordinates(available.astype(np.float32), coordinates, order=1, mode="constant") > .995
    readings = ndimage.map_coordinates(luma, coordinates, order=1, mode="constant")
    total = usable.sum(axis=1)
    values = (readings * usable).sum(axis=1) / np.maximum(total, 1)
    return line, values, total >= 2, span


def punch_centers(values, supported, minimum_px, prominence, polarity):
    boundaries = np.flatnonzero(np.diff(np.pad(supported.astype(np.int8), 1)))
    centers, groups = [], []
    for first, stop in zip(boundaries[::2], boundaries[1::2], strict=True):
        if stop - first < 7:
            continue
        readings = ndimage.gaussian_filter1d(values[first:stop], .70)
        local = ndimage.gaussian_filter1d(readings, max(minimum_px * .72, 2.2))
        contrast = readings - local
        if polarity == "dark":
            contrast = -contrast
        peaks, _ = signal.find_peaks(contrast, distance=minimum_px, prominence=prominence)
        groups.extend([len(centers)] * len(peaks))
        centers.extend((peaks + first).tolist())
    return centers, groups


def measure_perforation(frame, area, linear, options):
    available = frame.valid & ~frame.clipped & area
    points, values, support, span = edge_track(frame, linear, available, options["start"], options["end"])
    holes, groups = punch_centers(values, support, options["minimum_px"], options["prominence"], options["polarity"])
    extent = np.asarray(frame.extent, dtype=float)
    measured = np.linalg.norm((points[-1] - points[0]) * np.asarray(frame.scale or (1, 1)))
    stride = float(measured / (len(points) - 1))
    intervals = []
    rows = []
    for index, peak in enumerate(holes):
        spacing = None
        if index and groups[index] == groups[index - 1]:
            spacing = (peak - holes[index - 1]) * stride
            intervals.append(spacing)
        rows.append({
            "punch": index + 1,
            "distance": round(peak * stride, 4),
            "relative_position": np.round((points[peak] + .5) / extent, 6).tolist(),
            "spacing_from_previous": round(spacing, 4) if spacing is not None else None,
            "luminance": round(float(values[peak]), 5),
        })
    median = float(np.median(intervals)) if len(intervals) >= 3 else None
    spread = float(np.quantile(intervals, .75) - np.quantile(intervals, .25)) if median is not None else None
    picture = base_rgba(frame)
    painter = ImageDraw.Draw(picture)
    painter.line([tuple(points[0]), tuple(points[-1])], fill=(245, 202, 113), width=2)
    for peak in holes:
        x, y = points[peak]
        painter.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(113, 227, 215))
    return {
        "punch_count": len(holes),
        "measured_interval_count": len(intervals),
        "spacing_unit": "mm" if frame.scale else "px",
        "median_spacing": round(median, 4) if median is not None else None,
        "spacing_iqr": round(spread, 4) if spread is not None else None,
        "perforations_per_20mm": round(20 / median, 3) if median and frame.scale else None,
        "registered_span_px": round(span, 4),
        "supported_percent": round(100 * float(np.mean(support)), 3),
        "rows": rows,
        "note": "Visible peaks on the marked row; intervals cannot cross an unmeasured gap. An uncalibrated image reports pixels only.",
    }, picture
