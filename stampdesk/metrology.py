import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .analysis import capture_provenance, specimen_provenance
from .art import ACCENT, INK, SHADOW, base_rgba, curve_points, stored_preview
from .geometry import active_region, polygon_mask, region_domain, signed_area, specimen_frame, validate_polygon
from .model import InputError, collection, dictionary, enumeration, identity, numeric_pair, string
from .rulers import ruler_split


MIN_SEGMENT = 2.0
MIN_MARK_PIXELS = 20
REGION_OUTLINE = (199, 212, 151)
EXCLUDED_OUTLINE = (237, 122, 110)
REFERENCE_OUTLINE = (245, 176, 92)
REFERENCE_FILL = (245, 176, 92, 30)
MASK_THROUGH = (102, 180, 147, 94)


def path_points(payload):
    raw = collection(payload, "Trace points", minimum=2, maximum=45)
    return [numeric_pair(item, "Trace point", lower=0, upper=1) for item in raw]


def calibrated_dimensions(frame):
    width, height = frame.extent
    if frame.scale is not None:
        return np.array([width * frame.scale[0], height * frame.scale[1]]), "mm"
    return np.array([width, height]), "px"


def path_segments(frame, points, closed=False):
    factors, units = calibrated_dimensions(frame)
    point_pixels = np.asarray(points, dtype=np.float64) * frame.extent
    vertices = np.asarray(points, dtype=np.float64) * factors
    if closed:
        vertices = np.vstack((vertices, vertices[0]))
        point_pixels = np.vstack((point_pixels, point_pixels[0]))
    deltas = np.diff(vertices, axis=0)
    deltas_px = np.diff(point_pixels, axis=0)
    lengths_px = np.linalg.norm(deltas_px, axis=1)
    if np.any(lengths_px < MIN_SEGMENT):
        raise InputError("Each trace segment needs at least two registered image pixels.")
    length = np.linalg.norm(deltas, axis=1)
    segments = []
    for index, (delta, value, pixels) in enumerate(zip(deltas, length, lengths_px)):
        vector = delta.tolist()
        segments.append({
            "index": index + 1,
            "length": round(float(value), 5),
            "length_px": round(float(pixels), 5),
            "units": units,
            "dx": round(vector[0], 5),
            "dy": round(vector[1], 5),
            "clockwise_heading": round(math.degrees(math.atan2(vector[1], vector[0])) % 360, 3),
        })
    return segments, units


def path_cumulative(segments):
    distance = 0
    totals = [0]
    for segment in segments:
        distance += segment["length"]
        totals.append(round(distance, 5))
    return totals


def path_turns(segments, closed=False):
    angles = [segment["clockwise_heading"] for segment in segments]
    linked = angles[1:] + [angles[0]] if closed else angles[1:]
    return [round((after - before + 180) % 360 - 180, 3) for before, after in zip(angles, linked)]


def sample_support_along(available, frame, points, closed=False):
    positions = points + [points[0]] if closed else points
    xy = np.asarray(positions, dtype=np.float32) * frame.extent - 0.5
    mask = available.astype(np.float32)
    pieces = []
    for entry, (start, stop) in enumerate(zip(xy, xy[1:]), 1):
        length = float(np.linalg.norm(stop - start))
        count = min(1800, int(np.ceil(length)) + 1)
        positions = np.linspace(start, stop, count)
        row = np.clip(positions[:, 1], 0, mask.shape[0] - 1)
        col = np.clip(positions[:, 0], 0, mask.shape[1] - 1)
        sampled = ndimage.map_coordinates(mask, (row, col), order=1, mode="nearest")
        actual = sampled >= 0.995
        transitions = np.diff(np.pad(actual.astype(np.int8), 1))
        starts = np.flatnonzero(transitions == 1)
        stops = np.flatnonzero(transitions == -1)
        pieces.append({
            "index": entry,
            "samples": count,
            "supported_samples": int(np.count_nonzero(actual)),
            "support_percent": round(100 * float(np.mean(actual)), 3),
            "supported_intervals": [
                [round(int(lo) / count, 4), round(int(hi) / count, 4)] for lo, hi in zip(starts, stops)
            ],
        })
    return pieces


def region_outline_bounds(vertices, totals):
    values = np.asarray(vertices, dtype=np.float64)
    xy = values * totals
    minimum = np.min(xy, axis=0)
    maximum = np.max(xy, axis=0)
    extent = maximum - minimum
    return {
        "bounds": np.round(np.r_[minimum, maximum], 5).tolist(),
        "axis_width": round(float(extent[0]), 5),
        "axis_height": round(float(extent[1]), 5),
    }


def shape_center(vertices, total_area):
    points = np.asarray(vertices, dtype=np.float64)
    next_points = np.roll(points, -1, axis=0)
    area_terms = points[:, 0] * next_points[:, 1] - points[:, 1] * next_points[:, 0]
    weighted = np.sum((points + next_points) * area_terms[:, None], axis=0)
    return np.round(weighted / (6 * total_area), 6).tolist()


def proper_axis(support, scale):
    rows, cols = np.nonzero(support)
    if len(rows) < 7:
        return None
    points = np.stack((cols + 0.5, rows + 0.5), axis=1)
    if scale:
        points *= np.asarray(scale)
    centroid = np.mean(points, axis=0)
    cov = np.cov((points - centroid).T)
    eigen, directions = np.linalg.eigh(cov)
    major = directions[:, -1]
    minor_width = max(float(eigen[0]), 1e-12)
    major_width = float(eigen[1])
    elongation = math.sqrt(major_width / minor_width)
    return {
        "support_centroid_xy": np.round(centroid, 5).tolist(),
        "support_major_angle": round(math.degrees(math.atan2(major[1], major[0])) % 180, 3),
        "support_elongation": round(elongation, 4),
    }


def occupied_holes(mask):
    filled = ndimage.binary_fill_holes(mask)
    background = filled & ~mask
    identities, count = ndimage.label(background)
    if not count:
        return []
    areas = np.bincount(identities.ravel())[1:]
    areas = np.sort(areas[areas >= 4])[::-1]
    return [int(value) for value in areas[:24]]


def mask_geometry(frame, selected, requested, supported):
    dimensions, units = calibrated_dimensions(frame)
    area = signed_area(selected["vertices"])
    outer = abs(area) * float(np.prod(dimensions))
    occupied = int(np.count_nonzero(supported))
    if occupied < MIN_MARK_PIXELS:
        raise InputError("The marked area has fewer than 20 supported image pixels.")
    raster = int(np.count_nonzero(requested))
    factor = float(np.prod(frame.scale)) if frame.scale else 1
    axes = proper_axis(supported, frame.scale)
    return {
        "vector_outline_area": round(outer, 5),
        "raster_area": round(raster * factor, 5),
        "raster_supported_area": round(occupied * factor, 5),
        "requested_pixels": raster,
        "supported_pixels": occupied,
        "no_support_percent": round(100 * (1 - occupied / raster), 4),
        "outside_or_excluded_holes_px": occupied_holes(supported),
        "shape_centroid": shape_center(selected["vertices"], area),
        "support_spread": axes,
        "units": f"{units}²",
        **region_outline_bounds(selected["vertices"], dimensions),
    }


def split_trace_segments(frame, points, split):
    """Split a traced path where it crosses the ruler boundary.

    Pieces on the side whose ruler could not meet the tolerance keep their
    measured length as a reference value but do not enter the cuttable total.
    """
    factors, units = calibrated_dimensions(frame)
    vertices = np.asarray(points, dtype=np.float64)
    pixels = vertices * frame.extent
    physical = vertices * factors
    boundary = float(split["boundary_x"])
    failing = split["failing_side"]
    pieces = []
    cut_total = 0.0
    all_total = 0.0
    for index, (start, stop, start_px, stop_px) in enumerate(zip(physical, physical[1:], pixels, pixels[1:]), 1):
        delta = stop - start
        length = float(np.linalg.norm(delta))
        heading = round(math.degrees(math.atan2(delta[1], delta[0])) % 360, 3)
        first, last = vertices[index - 1], vertices[index]
        cuts = [0.0, 1.0]
        if first[0] != last[0] and min(first[0], last[0]) < boundary < max(first[0], last[0]):
            cuts.append((boundary - first[0]) / (last[0] - first[0]))
        cuts = sorted(set(round(value, 12) for value in cuts))
        for part, (start_fraction, stop_fraction) in enumerate(zip(cuts, cuts[1:]), 1):
            if stop_fraction <= start_fraction:
                continue
            middle_x = first[0] + (last[0] - first[0]) * (start_fraction + stop_fraction) / 2
            side = "right" if middle_x > boundary else "left"
            counts = side != failing
            piece_length = length * (stop_fraction - start_fraction)
            all_total += piece_length
            if counts:
                cut_total += piece_length
            pieces.append({
                "index": len(pieces) + 1,
                "source_index": index,
                "part": part,
                "parts": len(cuts) - 1,
                "length": round(piece_length, 5),
                "units": units,
                "dx": round(float(delta[0]) * (stop_fraction - start_fraction), 5),
                "dy": round(float(delta[1]) * (stop_fraction - start_fraction), 5),
                "clockwise_heading": heading,
                "side": side,
                "counts_toward_cut": counts,
                "reference_only": not counts,
                "start": np.round(first + (last - first) * start_fraction, 6).tolist(),
                "end": np.round(first + (last - first) * stop_fraction, 6).tolist(),
                "start_px": np.round(start_px + (stop_px - start_px) * start_fraction, 3).tolist(),
                "end_px": np.round(start_px + (stop_px - start_px) * stop_fraction, 3).tolist(),
                "cumulative_all": round(all_total, 5),
                "cumulative_cut": round(cut_total, 5),
            })
    return {
        **split,
        "segments": pieces,
        "cut_length": round(cut_total, 5),
        "reference_length": round(all_total - cut_total, 5),
        "measured_length": round(all_total, 5),
        "units": units,
    }


def draw_ruler_marks(image, split):
    """Tint the failing ruler half and draw both checked spans on the overlay."""
    from PIL import Image

    width, height = image.size
    boundary = min(max(split["boundary_x"], 0.0), 1.0) * width
    failing = split["failing_side"]
    left_edge, right_edge = (0, boundary) if failing == "left" else (boundary, width)
    tint = Image.new("RGBA", image.size, (0, 0, 0, 0))
    ImageDraw.Draw(tint).rectangle(
        (int(round(left_edge)), 0, int(round(right_edge)) - 1, height - 1),
        fill=REFERENCE_FILL,
    )
    image = Image.alpha_composite(image.convert("RGBA"), tint)
    draw = ImageDraw.Draw(image)
    stroke = max(1, round(min(width, height) / 310))
    draw.line([(boundary, 0), (boundary, height)], fill=REFERENCE_OUTLINE, width=stroke)
    for side, check in split["checks"].items():
        points = [tuple(np.asarray(point, dtype=np.float64) * [width, height]) for point in check["segment"]]
        region = check.get("region_vertices")
        if region:
            outline = curve_points([*region, region[0]], (width, height))
            draw.line(outline, fill=REFERENCE_OUTLINE if side == failing else REGION_OUTLINE, width=stroke)
        draw.line(points, fill=REFERENCE_OUTLINE if side == failing else ACCENT, width=stroke + 2)
        for x, y in points:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK, outline=SHADOW, width=1)
        x, y = points[0]
        draw.text(
            (x + 5, y + 5),
            f"{side} {check['measured_mm']:.2f} mm ({check['error_mm']:+.2f})",
            fill=INK,
            stroke_width=2,
            stroke_fill=SHADOW,
        )
    draw.text((boundary + 4, 6), f"{failing} side · reference only", fill=INK, stroke_width=2, stroke_fill=SHADOW)
    return image


def geometry_painter(frame, points, segments, closed=False, support=None, centroid=None, split=None):
    image = base_rgba(frame)
    width, height = image.size
    pixels = curve_points(points, image.size)
    if support is not None:
        raw = np.asarray(image, dtype=np.float32).copy()
        raw[support, :3] = raw[support, :3] * 0.72 + np.array(MASK_THROUGH[:3], dtype=np.float32) * 0.28
        from PIL import Image
        image = Image.fromarray(raw.astype(np.uint8))
    draw = ImageDraw.Draw(image)
    edges = [*pixels, pixels[0]] if closed else pixels
    stroke = max(1, round(min(width, height) / 310))
    draw.line(edges, fill=SHADOW, width=stroke + 3, joint="curve")
    draw.line(edges, fill=REGION_OUTLINE if closed else ACCENT, width=stroke + 1, joint="curve")
    if split is not None:
        for piece in split["segments"]:
            if piece["reference_only"]:
                draw.line([tuple(piece["start_px"]), tuple(piece["end_px"])], fill=REFERENCE_OUTLINE, width=stroke + 1)
        cut_by_source = {}
        for piece in split["segments"]:
            entry = cut_by_source.setdefault(piece["source_index"], [0.0, False])
            entry[0] += piece["length"] if piece["counts_toward_cut"] else 0.0
            entry[1] = entry[1] or piece["reference_only"]
    for index, (point_a, point_b) in enumerate(zip(edges, edges[1:])):
        x = (point_a[0] + point_b[0]) / 2
        y = (point_a[1] + point_b[1]) / 2
        line = segments[index]
        label = f"{line['length']:.2f} {line['units']}"
        if split is not None and index + 1 in cut_by_source:
            cut, partial = cut_by_source[index + 1]
            label += f" · reference only" if partial and cut == 0 else f" · cut {cut:.2f}"
        draw.text((x + 4, y - 9), label, fill=INK, stroke_width=2, stroke_fill=SHADOW)
    for index, (x, y) in enumerate(pixels, 1):
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=INK, outline=SHADOW, width=1)
        draw.text((x + 6, y + 3), str(index), fill=INK, stroke_width=2, stroke_fill=SHADOW)
    if centroid is not None:
        x, y = curve_points([centroid], image.size)[0]
        draw.line((x - 6, y, x + 6, y), fill=EXCLUDED_OUTLINE, width=2)
        draw.line((x, y - 6, x, y + 6), fill=EXCLUDED_OUTLINE, width=2)
    return image


def trace_geometry(frame, available, points, split=None):
    segments, units = path_segments(frame, points)
    cumulative = path_cumulative(segments)
    pieces = sample_support_along(available, frame, points)
    samples = sum(item["samples"] for item in pieces)
    present = sum(item["supported_samples"] for item in pieces)
    detail = {
        "points": points,
        "segments": segments,
        "trace_length": cumulative[-1],
        "cumulative": cumulative,
        "turn_angles": path_turns(segments),
        "support_along_line": pieces,
        "overall_support": round(100 * present / samples, 3),
        "length_unit": units,
    }
    divided = split_trace_segments(frame, points, split) if split else None
    image = geometry_painter(frame, points, segments, split=divided)
    if divided is not None:
        image = draw_ruler_marks(image, divided)
        detail["ruler_split"] = divided
        detail["cut_length"] = divided["cut_length"]
        detail["reference_length"] = divided["reference_length"]
        detail["cut_length_unit"] = divided["units"]
    return detail, image


def polygon_geometry(tile, frame, selection, available):
    vertices = validate_polygon(selection["vertices"])
    requested = polygon_mask(frame.extent, vertices)
    supported = requested & available
    dimensions, units = calibrated_dimensions(frame)
    geometry = mask_geometry(frame, selection, requested, supported)
    segments, _ = path_segments(frame, vertices, closed=True)
    perimeter = path_cumulative(segments)[-1]
    geometry.update(
        points=vertices,
        outline_length=perimeter,
        outline_unit=units,
        outline_segments=segments,
        mask_purpose=selection["purpose"],
        contour_support=sample_support_along(available, frame, vertices, closed=True),
    )
    painter = geometry_painter(frame, vertices, segments, True, supported, geometry["shape_centroid"])
    return geometry, painter


def create_metrology(store, specimen_id, payload):
    data = dictionary(payload, "Image measurement")
    title = string(data.get("title"), "Measurement title", maximum=90)
    mode = enumeration(data.get("mode"), ("trace", "area"), "Geometry measurement")
    tile = store.tile(specimen_id)
    frame = specimen_frame(store.blobs, tile)
    available = region_domain(frame.extent, tile) & frame.valid
    if mode == "trace":
        points = path_points(data.get("points"))
        shape, overlay = trace_geometry(frame, available, points, ruler_split(tile))
        region = None
    else:
        region = active_region(tile, identity(data.get("region_id"), "Region ID"))
        shape, overlay = polygon_geometry(tile, frame, region, available)
    return store.save_metrology({
        "title": title,
        "mode": mode,
        "tile_id": tile["id"],
        "specimen": specimen_provenance(tile),
        "source": capture_provenance(frame),
        "region": region,
        "exclusions": [candidate for candidate in tile["regions"] if candidate["purpose"] == "ignore"],
        "shape": shape,
        "image": stored_preview(overlay, store.blobs),
    })
