"""Reconcile the left and right horizontal ruler checks into one cut-card scale.

A photographed specimen may show a horizontal ruler at each side.  Warping or
paper curl makes their apparent ten millimetre spans disagree.  A cut card can
only carry one horizontal and one vertical scale, so this module records both
checks, chooses the scale that best satisfies them, and marks any side that
cannot meet the stated tolerance.  Geometry measurements use the resulting
scale and can then keep the non-qualifying side as a reference-only length.
"""

import math

import numpy as np

from .model import (
    InputError,
    MAX_REGISTERED_DIM,
    collection,
    dictionary,
    identity,
    match_slot,
    numeric_pair,
    scalar,
    string,
)


DEFAULT_SPAN_MM = 10.0
DEFAULT_TOLERANCE_MM = 0.2
SIDES = ("left", "right")
MIN_CHECK_SPAN_PX = 8.0
ROUND = 7


def _finite(value):
    return value is not None and math.isfinite(float(value)) and float(value) > 0


def _pair(value, label):
    return numeric_pair(value, label, lower=0, upper=1)


def check_segment(data):
    """Return the two normalized endpoints of a marked ruler span."""
    if "segment" in data:
        raw = collection(data.get("segment"), "Ruler span", minimum=2, maximum=2)
    else:
        raw = [data.get("start"), data.get("end")]
    return [_pair(item, "Ruler span point") for item in raw]


def check_polygon(data, tile):
    """Resolve the paper area that belongs to one ruler check.

    A submitted vertex snapshot is preferred so the check keeps working if the
    live sampling region is later edited or removed.
    """
    raw = data.get("vertices", data.get("region_vertices"))
    if raw is not None:
        vertices = [numeric_pair(item, "Ruler check vertex", lower=0, upper=1)
                    for item in collection(raw, "Ruler check vertices", minimum=3, maximum=40)]
        return {
            "id": data.get("region_id"),
            "name": string(data.get("region_name", data.get("name")), "Ruler check area", maximum=80, required=False),
            "purpose": "inspection",
            "vertices": vertices,
        }
    region_id = data.get("region_id", data.get("region"))
    if region_id is not None:
        region = match_slot(tile["regions"], identity(region_id, "Ruler check region"), "sampling region")
        return {
            "id": region["id"],
            "name": region["name"],
            "purpose": region["purpose"],
            "vertices": [list(vertex) for vertex in region["vertices"]],
        }
    return None


def side_check(key, data, tile):
    data = dictionary(data, f"{key.title()} ruler check")
    segment = check_segment(data)
    span_mm = scalar(data.get("span_mm", data.get("span", DEFAULT_SPAN_MM)), "Ruler span", lower=0.1, upper=10000)
    return {
        "side": key,
        "segment": segment,
        "span_mm": span_mm,
        "region": check_polygon(data, tile),
    }


def ruler_payload(payload, tile, previous=None):
    """Validate a ruler calibration request and return the stored check set.

    A side may be omitted on an update. The previously stored check is kept so
    saving one re-marked ruler does not discard the other side.
    """
    data = dictionary(payload, "Ruler calibration")
    previous = previous or {}
    previous_sides = previous.get("sides", {})
    raw_sides = data.get("sides")
    if raw_sides is None:
        raw_sides = {key: data.get(key) for key in SIDES if data.get(key) is not None}
    raw_sides = dictionary(raw_sides, "Ruler checks")
    unknown = set(raw_sides) - set(SIDES)
    if unknown:
        raise InputError("Ruler checks can only be recorded for the left and right sides.")
    reset = data.get("reset_sides") is True
    sides = {}
    for key in SIDES:
        if raw_sides.get(key) is not None:
            sides[key] = side_check(key, raw_sides[key], tile)
        elif not reset and key in previous_sides:
            sides[key] = previous_sides[key]
    if not sides:
        raise InputError("Mark a ten-millimetre span for at least one horizontal ruler.")
    tolerance = scalar(
        data.get("tolerance_mm", previous.get("tolerance_mm", DEFAULT_TOLERANCE_MM)),
        "Ruler tolerance",
        lower=0.001,
        upper=10,
    )
    vertical = previous.get("vertical")
    if data.get("vertical") is not None:
        vertical_data = dictionary(data["vertical"], "Vertical ruler check")
        vertical = {
            "segment": check_segment(vertical_data),
            "span_mm": scalar(
                vertical_data.get("span_mm", vertical_data.get("span", DEFAULT_SPAN_MM)),
                "Vertical ruler span",
                lower=0.1,
                upper=10000,
            ),
        }
    elif data.get("reset_vertical") is True:
        vertical = None
    return {
        "sides": sides,
        "tolerance_mm": tolerance,
        "span_mm": DEFAULT_SPAN_MM,
        "vertical": vertical,
    }


def _vertical_scale(tile, extent, config, base_sx):
    """Vertical millimetres per registered pixel; the upright ruler is trusted."""
    width, height = extent
    vertical = config.get("vertical")
    if vertical:
        start, end = np.asarray(vertical["segment"], dtype=np.float64) * [width, height]
        dy = abs(float(end[1] - start[1]))
        dx = abs(float(end[0] - start[0]))
        if dx and base_sx:
            horizontal_mm = dx * base_sx
            remaining = vertical["span_mm"] ** 2 - horizontal_mm ** 2
            if remaining <= 0:
                raise InputError("The vertical ruler span is too tilted to recover a vertical scale.")
            return math.sqrt(remaining) / dy, "ruler_check"
        if dy < MIN_CHECK_SPAN_PX:
            raise InputError("Mark a longer vertical ruler span.")
        return vertical["span_mm"] / dy, "ruler_check"
    height_mm = tile.get("height_mm")
    if height_mm is None:
        raise InputError("Enter the specimen height or mark a vertical ruler span before saving ruler checks.")
    return float(height_mm) / height, "specimen_dimensions"


def _measure_side(check, extent, vertical):
    width, height = extent
    points = np.asarray(check["segment"], dtype=np.float64) * [width, height]
    delta = points[1] - points[0]
    dx = abs(float(delta[0]))
    dy = abs(float(delta[1]))
    span_px = math.hypot(dx, dy)
    if span_px < MIN_CHECK_SPAN_PX:
        raise InputError("Mark a ruler span of at least eight registered pixels.")
    if dx < 1.0:
        raise InputError("Mark the horizontal ruler span across the image rather than along one column.")
    sy = vertical[0]
    vertical_mm = dy * sy
    remaining = check["span_mm"] ** 2 - vertical_mm ** 2
    if remaining <= 0:
        raise InputError("The marked ruler span is too tilted to recover a horizontal scale.")
    ideal_sx = math.sqrt(remaining) / dx
    return {
        **check,
        "span_px": [round(dx, ROUND), round(dy, ROUND)],
        "span_length_px": round(span_px, ROUND),
        "ideal_horizontal_mm_per_px": ideal_sx,
    }


def measured_mm(check, sx, sy):
    dx, dy = check["span_px"]
    return math.hypot(dx * sx, dy * sy)


def check_error(check, sx, sy):
    return measured_mm(check, sx, sy) - check["span_mm"]


def _side_error(check, sx, sy):
    return check_error(check, sx, sy)


def _balanced_scale(first, second, sy):
    low = min(first["ideal_horizontal_mm_per_px"], second["ideal_horizontal_mm_per_px"])
    high = max(first["ideal_horizontal_mm_per_px"], second["ideal_horizontal_mm_per_px"])
    if math.isclose(low, high, rel_tol=1e-12, abs_tol=1e-15):
        return low
    # One side reads short and the other long; split the excess between them.
    for _ in range(80):
        middle = (low + high) / 2
        if _side_error(first, middle, sy) + _side_error(second, middle, sy) <= 0:
            low = middle
        else:
            high = middle
    return (low + high) / 2


def _region_outer_x(region, side):
    if not region:
        return None
    values = [float(vertex[0]) for vertex in region["vertices"]]
    return max(values) if side == "left" else min(values)


def side_boundary(sides):
    """Normalized vertical line separating the left and right ruler areas."""
    left = sides.get("left", {}).get("region")
    right = sides.get("right", {}).get("region")
    left_inner = _region_outer_x(left, "left")
    right_inner = _region_outer_x(right, "right")
    if left_inner is not None and right_inner is not None and left_inner <= right_inner:
        boundary = (left_inner + right_inner) / 2
    elif left_inner is not None and right_inner is not None:
        left_center = float(np.mean([vertex[0] for vertex in left["vertices"]]))
        right_center = float(np.mean([vertex[0] for vertex in right["vertices"]]))
        boundary = (left_center + right_center) / 2
    elif left_inner is not None:
        boundary = min(0.95, left_inner)
    elif right_inner is not None:
        boundary = max(0.05, right_inner)
    else:
        boundary = 0.5
    return round(min(0.999, max(0.001, boundary)), ROUND)


def _status(key, mode, reference, failing):
    if key == failing:
        return "fail"
    if mode == "shared":
        return "pass"
    if key == reference:
        return "reference"
    return "pass"


def fit_rulers(tile, config, extent=None):
    """Choose one horizontal scale from the side checks and report both errors."""
    plane = tile.get("plane")
    if plane is None:
        raise InputError("Set the specimen plane before checking rulers.")
    width, height = extent or tuple(plane["size"])
    if not (0 < width <= MAX_REGISTERED_DIM and 0 < height <= MAX_REGISTERED_DIM):
        raise InputError("The specimen plane is not a usable registered size.")
    base = None
    if tile.get("width_mm") is not None and tile.get("height_mm") is not None:
        base = (float(tile["width_mm"]) / width, float(tile["height_mm"]) / height)
    vertical = _vertical_scale(tile, (width, height), config, base[0] if base else None)
    sy = vertical[0]
    checks = {}
    for key, check in config["sides"].items():
        checks[key] = _measure_side(check, (width, height), vertical)

    valid = sorted(checks)
    mode = None
    reference = None
    failing = None
    if len(valid) == 1:
        reference = valid[0]
        chosen = checks[reference]["ideal_horizontal_mm_per_px"]
        mode = "single"
    elif len(valid) == 2:
        first, second = (checks[key] for key in valid)
        balanced = _balanced_scale(first, second, sy)
        balanced_error = max(abs(check_error(first, balanced, sy)), abs(check_error(second, balanced, sy)))
        if balanced_error <= config["tolerance_mm"] + 1e-12:
            chosen = balanced
            mode = "shared"
        else:
            options = []
            nominal = base[0] if base else balanced
            for key in valid:
                candidate = checks[key]["ideal_horizontal_mm_per_px"]
                worst = max(
                    abs(check_error(other, candidate, sy))
                    for other_key, other in checks.items()
                    if other_key != key
                )
                options.append((worst, abs(candidate - nominal), SIDES.index(key), key, candidate))
            _, _, _, reference, chosen = min(options)
            failing = next(key for key in valid if key != reference)
            mode = "reference"
    else:
        raise InputError("Record at most one horizontal ruler check per side.")

    if mode == "shared":
        failing = None
    if mode == "reference" and reference == failing:
        raise InputError("The failing ruler side cannot define the cut-card scale.")

    rendered = {}
    for key in SIDES:
        check = checks.get(key)
        if check is None:
            continue
        error = check_error(check, chosen, sy)
        rendered[key] = {
            "region_id": (check.get("region") or {}).get("id"),
            "region_name": (check.get("region") or {}).get("name") or "",
            "region_vertices": (check.get("region") or {}).get("vertices"),
            "segment": check["segment"],
            "span_mm": check["span_mm"],
            "span_px": check["span_px"],
            "span_length_px": check["span_length_px"],
            "ideal_horizontal_mm_per_px": round(check["ideal_horizontal_mm_per_px"], ROUND),
            "measured_mm": round(measured_mm(check, chosen, sy), ROUND),
            "error_mm": round(error, ROUND),
            "within_tolerance": bool(abs(error) <= config["tolerance_mm"] + 1e-12),
            "status": _status(key, mode, reference, failing),
        }
    tolerance = config["tolerance_mm"]
    if mode == "shared":
        note = (
            f"Both horizontal ruler checks agree within {tolerance:g} mm. "
            "Their shared scale is written to the cut card."
        )
    elif mode == "reference":
        note = (
            f"The two horizontal checks cannot both meet {tolerance:g} mm. "
            f"The {reference} ruler defines the cut-card scale; the {failing} side is reference-only "
            "for lengths that cross the ruler boundary."
        )
    else:
        note = (
            f"Only the {reference} horizontal ruler check is available. It defines the cut-card scale; "
            "the other half has no independent horizontal check."
        )
    boundary = side_boundary({key: check for key, check in config["sides"].items()})
    return {
        "horizontal_mm_per_px": round(chosen, ROUND),
        "vertical_mm_per_px": round(sy, ROUND),
        "vertical_source": vertical[1],
        "nominal_horizontal_mm_per_px": round(base[0], ROUND) if base else None,
        "tolerance_mm": tolerance,
        "mode": mode,
        "reference_side": reference,
        "failing_side": failing,
        "boundary_x": boundary,
        "checks": rendered,
        "note": note,
    }


def current_rulers(tile):
    """Return the saved checks plus a freshly reconciled scale for this plane."""
    config = tile.get("rulers")
    if not config:
        return None
    result = {
        "sides": config.get("sides", {}),
        "tolerance_mm": config.get("tolerance_mm", DEFAULT_TOLERANCE_MM),
        "span_mm": config.get("span_mm", DEFAULT_SPAN_MM),
        "vertical": config.get("vertical"),
        "saved_at": config.get("saved_at"),
    }
    try:
        result["selected"] = fit_rulers(tile, config)
        result["error"] = None
    except (InputError, ValueError, KeyError, TypeError) as error:
        result["selected"] = None
        result["error"] = str(error)
    return result


def frame_scale(tile, width, height):
    """Registered-plane pixel scale, using the reconciled ruler pair when present."""
    base = None
    if tile.get("width_mm") is not None and tile.get("height_mm") is not None:
        base = (float(tile["width_mm"]) / width, float(tile["height_mm"]) / height)
    config = tile.get("rulers")
    if not config:
        return base
    try:
        selected = fit_rulers(tile, config, (width, height))
    except (InputError, ValueError, KeyError, TypeError):
        return base
    sx = selected.get("horizontal_mm_per_px")
    sy = selected.get("vertical_mm_per_px")
    if _finite(sx) and _finite(sy):
        return (float(sx), float(sy))
    return base


def ruler_split(tile):
    """The boundary and failing side used to split Geometry traces, or None.

    Only an out-of-tolerance check restricts cutting. With a single check the
    unchecked half is not split: no second ruler was marked to dispute it.
    """
    config = tile.get("rulers")
    if not config:
        return None
    try:
        selected = fit_rulers(tile, config)
    except (InputError, ValueError, KeyError, TypeError):
        return None
    if selected["failing_side"] is None or selected["mode"] == "single":
        return None
    return {
        "failing_side": selected["failing_side"],
        "reference_side": selected["reference_side"],
        "boundary_x": selected["boundary_x"],
        "tolerance_mm": selected["tolerance_mm"],
        "horizontal_mm_per_px": selected["horizontal_mm_per_px"],
        "vertical_mm_per_px": selected["vertical_mm_per_px"],
        "checks": selected["checks"],
        "note": selected["note"],
    }
