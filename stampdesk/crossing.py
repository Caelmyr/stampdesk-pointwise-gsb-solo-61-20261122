"""Pair the four visible arms of one traced crack crossing into two damage numbers.

The fine-line network trace (``craze``) records every observed dark thread but the
archive book numbers one injury at a time. At a crossing the four visible arms have
to be split into two pairs; two kinds of evidence can disagree:

* continuity of direction ("straight" arms that read as one running fissure);
* similarity of tone ("turning" arms whose depth and darkness match better).

Every pairing here is a perfect matching: each traced arm is assigned to exactly one
damage number. Nothing is drawn across unsupported image space, so an
unphotographed gap never becomes part of a crack.
"""

import math

import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .art import INK, SHADOW, load_artifact_picture, stored_preview
from .craze import (
    LINE_CONNECTIVITY,
    contrast_suppress,
    degree_map,
    gradient_field,
    threshold_centerline,
)
from .geometry import polygon_mask, specimen_frame
from .model import InputError, collection, dictionary, numeric_pair, string
from .photos import luminance


MIN_BOX_SIDE = 0.008
MIN_ARM_PIXELS = 3
# A thick arm thins to a centerline whose crossing can occupy a few adjacent
# junction pixels; merge junction clusters within this pixel radius as one node.
JUNCTION_MERGE_RADIUS = 7
DAMAGE_LABEL_MAX = 24
CROSSING_REVISION_LIMIT = 1
DAMAGE_COLORS = ((107, 196, 165), (232, 158, 109))
JUNCTION_TINT = (244, 224, 128)
IDLE_TINT = (165, 179, 192)
BASIS_CHOICES = ("direction", "tone")


def crossing_box(payload):
    data = dictionary(payload, "Crossing frame")
    corners = collection(data.get("box"), "Crossing frame corners", minimum=2, maximum=2)
    points = [numeric_pair(item, "Frame corner", lower=0, upper=1) for item in corners]
    (ux0, vy0), (ux1, vy1) = points
    left, right = sorted((ux0, ux1))
    top, bottom = sorted((vy0, vy1))
    if right - left < MIN_BOX_SIDE or bottom - top < MIN_BOX_SIDE:
        raise InputError("Frame a larger area around the crossing; the box is too small to work from.")
    return [round(left, 6), round(top, 6), round(right, 6), round(bottom, 6)]


def label_fields(payload, existing=None):
    data = dictionary(payload, "Damage numbers")
    labels = []
    for index in (0, 1):
        fallback = existing[index]["label"] if existing else None
        label = string(
            data.get(f"label_{index + 1}", fallback),
            f"Damage number {index + 1}",
            maximum=DAMAGE_LABEL_MAX,
        )
        labels.append(label)
    if labels[0] == labels[1]:
        raise InputError("The two damage numbers must differ; one injury cannot carry two numbers.")
    return labels


def trace_network(store, record, box):
    """Re-run the saved craze tracing inside the framed area and return its arrays."""
    tile = store.tile(record["tile_id"])
    photo_id = record["provenance"]["photo"]["id"]
    frame = specimen_frame(store.blobs, tile, photo_id)
    options = dict(record["parameters"]["options"])
    rectangle = np.asarray([
        [box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]],
    ])
    domain = polygon_mask(frame.extent, rectangle)
    candidate = domain & frame.valid & ~frame.clipped
    if np.count_nonzero(candidate) < 120:
        raise InputError("The framed area has too little continuous, unclipped image support.")
    source = luminance(frame.linear)
    radius = options["radius"]
    filtered, valid = gradient_field(source, candidate, radius)
    fine, support = contrast_suppress(source, candidate, radius)
    coarse, margin = contrast_suppress(source, candidate, 2 * radius)
    support &= valid & margin
    if np.count_nonzero(support) < 100:
        raise InputError("Frame a larger area with contiguous, unclipped image support.")
    thin, binary, thresholds = threshold_centerline(filtered, fine, coarse, support, options)
    return frame, thin, binary, thresholds, options


def locate_junction(thin, box, extent):
    width, height = extent
    left = max(0, int(math.floor(box[0] * width)))
    top = max(0, int(math.floor(box[1] * height)))
    right = min(width, int(math.ceil(box[2] * width)))
    bottom = min(height, int(math.ceil(box[3] * height)))
    framed = np.zeros_like(thin)
    framed[top:bottom, left:right] = thin[top:bottom, left:right]
    if not framed.any():
        raise InputError("No traced crack line falls inside this frame. Re-run the fine-line trace on this area.")
    degree = degree_map(framed)
    junction_pixels = (degree >= 3) & framed
    _, raw_count = ndimage.label(junction_pixels, structure=LINE_CONNECTIVITY)
    if raw_count == 0:
        raise InputError("The frame contains no crossing (no line junction). Center it on the two-line intersection.")
    # Thinning a thick crossing can leave two nearby junction pixels. Group the
    # clusters whose centers sit within the merge radius, then pull every
    # centerline pixel between those clusters into the single junction node.
    labels, count = ndimage.label(
        ndimage.binary_dilation(
            junction_pixels,
            structure=ndimage.iterate_structure(
                ndimage.generate_binary_structure(2, 2), JUNCTION_MERGE_RADIUS,
            ),
        ) & framed,
        structure=LINE_CONNECTIVITY,
    )
    if count > 1:
        raise InputError("The frame contains more than one distinct crossing. Tighten it around the single crossing.")
    node_mask = labels == 1
    # Keep the junction node on the centerline only (dilation may touch off-line pixels).
    node_mask &= framed
    rows, cols = np.nonzero(node_mask)
    junction = {(int(r), int(c)) for r, c in zip(rows, cols)}
    center = (
        round((float(cols.mean()) + 0.5) / width, 6),
        round((float(rows.mean()) + 0.5) / height, 6),
    )
    return junction, center


def skeleton_neighbors(thin):
    height, width = thin.shape
    rows, cols = np.nonzero(thin)
    adjacency = {}
    for r, c in zip(rows.tolist(), cols.tolist()):
        linked = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and thin[nr, nc]:
                    linked.append((nr, nc))
        adjacency[(r, c)] = linked
    return adjacency


def walk_arm(adjacency, seed, junction):
    """Follow one centerline thread away from the junction until its endpoint.

    The whole merged junction cluster is one terminal node, so a thread that
    meets any of its pixels has arrived back at the crossing.
    """
    previous = None
    current = seed
    path = []
    visited = set(junction)
    while current not in junction:
        if current in visited:
            raise InputError("A closed loop meets the crossing; tighten the frame around one crossing only.")
        visited.add(current)
        path.append(current)
        onward = [node for node in adjacency[current] if node != previous and node not in junction]
        if not onward:
            return path
        if len(onward) > 1:
            # A second branch point on the same thread means this is not a
            # clean four-arm crossing and any pairing would be ambiguous.
            raise InputError("A second junction sits inside the frame; tighten it around one crossing only.")
        previous, current = current, onward[0]
    return path


def physical_factor(scale):
    return np.asarray(scale, dtype=np.float64) if scale else np.asarray((1.0, 1.0))


def arm_geometry(junction_center, path, extent, scale):
    width, height = extent
    factors = physical_factor(scale)
    junction_px = np.asarray(junction_center, dtype=np.float64) * extent
    base_row, base_col = path[0]
    tip_row, tip_col = path[-1]
    base_xy = np.asarray((base_col + 0.5, base_row + 0.5), dtype=np.float64) * factors
    tip_xy = np.asarray((tip_col + 0.5, tip_row + 0.5), dtype=np.float64) * factors
    vector = tip_xy - base_xy
    length_px = math.hypot(base_col + 0.5 - junction_px[0], base_row + 0.5 - junction_px[1])
    for first, second in zip(path, path[1:]):
        length_px += math.hypot(second[1] - first[1], second[0] - first[0])
    length = length_px * float(np.mean(factors))
    heading = math.degrees(math.atan2(vector[1], vector[0])) % 360
    outward = tip_xy - junction_px * factors
    return outward, length, length_px, heading


def local_normal(path, index):
    """Unit normal of the centerline tangent at one path point."""
    start = path[max(0, index - 2)]
    stop = path[min(len(path) - 1, index + 2)]
    dr = stop[0] - start[0]
    dc = stop[1] - start[1]
    norm = math.hypot(dr, dc) or 1.0
    # Perpendicular to the (row, col) tangent.
    return -dc / norm, dr / norm


def cross_section_width(binary, point, normal, max_reach=12):
    """Count contiguous binary-region pixels on the normal through one point."""
    r, c = point
    nr, nc = normal
    half = 0
    for step in range(1, max_reach + 1):
        rr, cc = int(round(r + nr * step)), int(round(c + nc * step))
        if not (0 <= rr < binary.shape[0] and 0 <= cc < binary.shape[1]) or not binary[rr, cc]:
            break
        half = step
    other = 0
    for step in range(1, max_reach + 1):
        rr, cc = int(round(r - nr * step)), int(round(c - nc * step))
        if not (0 <= rr < binary.shape[0] and 0 <= cc < binary.shape[1]) or not binary[rr, cc]:
            break
        other = step
    return 1 + half + other


def arm_tone(binary, path):
    # Tone/depth of a fissure is represented by its observed trace thickness:
    # a wider dark thread reads as a deeper, stronger injury. Each width is a
    # cross-section of the detected region normal to the local centerline.
    widths = [
        cross_section_width(binary, point, local_normal(path, index))
        for index, point in enumerate(path)
    ]
    return round(float(np.median(widths)), 3)


def build_arm(binary, junction_center, path, extent, scale):
    width, height = extent
    outward, length, length_px, heading = arm_geometry(junction_center, path, extent, scale)
    tip_row, tip_col = path[-1]
    return {
        "tip": [round((tip_col + 0.5) / width, 6), round((tip_row + 0.5) / height, 6)],
        "heading_deg": round(heading, 3),
        "length": round(length, 5),
        "length_px": round(length_px, 5),
        "median_width_px": arm_tone(binary, path),
        "trace_pixels": len(path),
        "path": [[int(r), int(c)] for r, c in path],
        "outward_xy": [round(float(outward[0]), 5), round(float(outward[1]), 5)],
    }


def extract_arms(thin, binary, junction, center, extent, scale):
    adjacency = skeleton_neighbors(thin)
    seeds = sorted({node for cell in junction for node in adjacency[cell] if node not in junction})
    arms = []
    seen = set()
    for seed in seeds:
        path = walk_arm(adjacency, seed, junction)
        if len(path) < MIN_ARM_PIXELS:
            continue
        signature = tuple(path)
        if signature in seen:
            continue
        seen.add(signature)
        arms.append(build_arm(binary, center, path, extent, scale))
    if len(arms) != 4:
        raise InputError(
            f"The framed junction exposes {len(arms)} usable thread end(s), not four. "
            "Adjust the fine-line trace or frame so every arm is visible to its end."
        )
    order = sorted(range(4), key=lambda index: math.atan2(arms[index]["outward_xy"][1],
                                                           arms[index]["outward_xy"][0]))
    return [arms[index] for index in order]


def angle_delta(first, second):
    """Smallest angle between two outward arm directions, 0..180 degrees."""
    delta = abs(first - second) % 360
    return min(delta, 360 - delta)


def pair_score(arms, left, right):
    # Opposite arms of one straight fissure point in nearly opposite outward
    # directions, i.e. their headings differ by ~180 degrees.
    straightness = angle_delta(arms[left]["heading_deg"], arms[right]["heading_deg"])
    continuity = round(180 - straightness, 4)
    width_gap = abs(arms[left]["median_width_px"] - arms[right]["median_width_px"])
    tone_gap = round(width_gap, 5)
    return {
        "arms": [left, right],
        "straightness_deg": round(straightness, 3),
        "continuity": continuity,
        "tone_gap_px": tone_gap,
        "width_gap_px": round(width_gap, 4),
    }


def perfect_matchings():
    # The three distinct perfect matchings of four labelled arms.
    return [
        ((0, 1), (2, 3)),
        ((0, 2), (1, 3)),
        ((0, 3), (1, 2)),
    ]


def score_matchings(arms):
    options = []
    for matching in perfect_matchings():
        pairs = [pair_score(arms, *pair) for pair in matching]
        options.append({
            "matching": [[pair["arms"][0], pair["arms"][1]] for pair in pairs],
            "pairs": pairs,
            "continuity_total": round(sum(pair["continuity"] for pair in pairs), 4),
            "tone_gap_total": round(sum(pair["tone_gap_px"] for pair in pairs), 5),
        })
    direction_best = max(options, key=lambda item: item["continuity_total"])
    tone_best = min(options, key=lambda item: item["tone_gap_total"])
    direction_best = {**direction_best, "basis": "direction"}
    tone_best = {**tone_best, "basis": "tone"}
    return options, direction_best, tone_best


def find_matching(options, basis):
    if basis not in BASIS_CHOICES:
        raise InputError("Choose the direction or the tone pairing.")
    if basis == "direction":
        chosen = max(options, key=lambda item: item["continuity_total"])
    else:
        chosen = min(options, key=lambda item: item["tone_gap_total"])
    return chosen["matching"]


def damage_summary(arms, matching, labels, unit, extent):
    width, height = extent
    damages = []
    for damage_index, pair in enumerate(matching):
        arm_ids = list(pair)
        chosen = [arms[arm_id] for arm_id in arm_ids]
        rows = [point for arm in chosen for point in arm["path"]]
        rr = np.asarray([point[0] for point in rows])
        cc = np.asarray([point[1] for point in rows])
        bounds = [
            round((cc.min() + 0.5) / width, 6),
            round((rr.min() + 0.5) / height, 6),
            round((cc.max() + 0.5) / width, 6),
            round((rr.max() + 0.5) / height, 6),
        ]
        damages.append({
            "damage_index": damage_index,
            "label": labels[damage_index],
            "arm_ids": arm_ids,
            "length": round(sum(arm["length"] for arm in chosen), 5),
            "length_unit": unit,
            "length_px": round(sum(arm["length_px"] for arm in chosen), 5),
            "trace_pixels": int(sum(arm["trace_pixels"] for arm in chosen)),
            "bounds": bounds,
            "median_width_px": round(float(np.median([arm["median_width_px"] for arm in chosen])), 3),
        })
    return damages


def public_arms(arms):
    return [{key: value for key, value in arm.items() if key != "path"} for arm in arms]


def analyze_crossing(store, record, box):
    frame, thin, binary, thresholds, options = trace_network(store, record, box)
    junction, center = locate_junction(thin, box, frame.extent)
    arms = extract_arms(thin, binary, junction, center, frame.extent, frame.scale)
    scored, direction_best, tone_best = score_matchings(arms)
    document = {
        "box": box,
        "junction_center": center,
        "unit": "mm" if frame.scale else "px",
        "arms": public_arms(arms),
        "arms_geometry": arms,
        "matching_options": [{key: value for key, value in option.items()} for option in scored],
        "direction_matching": direction_best["matching"],
        "tone_matching": tone_best["matching"],
        "bases_conflict": direction_best["matching"] != tone_best["matching"],
        "trace_options": options,
        "thresholds": thresholds,
        "extent": list(frame.extent),
    }
    return frame, document


def build_assignment(document, payload):
    basis = payload.get("basis", "direction")
    matching = find_matching(document["matching_options"], basis)
    labels = label_fields(payload)
    damages = damage_summary(
        document["arms_geometry"], matching, labels, document["unit"], document["extent"],
    )
    return {
        "basis": basis,
        "matching": matching,
        "damages": damages,
        "revision_count": 0,
        "history": [],
    }


def revise_assignment(document, assignment, payload):
    if assignment["revision_count"] >= CROSSING_REVISION_LIMIT:
        raise InputError("This crossing has already used its one pairing revision.")
    target = dictionary(payload, "Pairing revision").get("basis", "tone")
    if target not in BASIS_CHOICES:
        raise InputError("Choose the direction or the tone pairing.")
    if target == assignment["basis"]:
        raise InputError("The crossing already uses that pairing basis.")
    matching = find_matching(document["matching_options"], target)
    reason = string(payload.get("reason"), "Revision reason", maximum=400, required=False)
    labels = label_fields(payload, existing=assignment["damages"])
    damages = damage_summary(
        document["arms_geometry"], matching, labels, document["unit"], document["extent"],
    )
    history = [
        *assignment["history"],
        {
            "basis": assignment["basis"],
            "matching": assignment["matching"],
            "damages": assignment["damages"],
            "reason": reason,
            "recorded_revision": assignment["revision_count"] + 1,
        },
    ]
    return {
        "basis": target,
        "matching": matching,
        "damages": damages,
        "revision_count": assignment["revision_count"] + 1,
        "history": history,
    }


def _arm_polyline(arm, junction_center, width, height):
    points = [(junction_center[0] * width, junction_center[1] * height)]
    points.extend(
        ((col + 0.5) / width * width, (row + 0.5) / height * height)
        for row, col in arm["path"]
    )
    return points


def draw_crossing_overlay(blobs, record):
    image = load_artifact_picture(blobs, record["pictures"]["source"])
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, round(min(width, height) / 280))
    crossing = record.get("crossing")
    if crossing is None:
        return image
    arms = crossing["arms_geometry"]
    junction_x, junction_y = crossing["junction_center"]
    jx, jy = junction_x * width, junction_y * height
    box = crossing["box"]
    draw.rectangle(
        (box[0] * width, box[1] * height, box[2] * width, box[3] * height),
        outline=IDLE_TINT, width=1,
    )
    radius = max(4, line_width * 2)
    assignment = crossing.get("assignment")
    if assignment is None:
        for arm in arms:
            points = _arm_polyline(arm, crossing["junction_center"], width, height)
            draw.line(points, fill=SHADOW, width=line_width + 2, joint="curve")
            draw.line(points, fill=IDLE_TINT, width=line_width, joint="curve")
        draw.ellipse((jx - radius, jy - radius, jx + radius, jy + radius),
                     fill=JUNCTION_TINT, outline=SHADOW, width=1)
        return image
    for damage in assignment["damages"]:
        color = DAMAGE_COLORS[damage["damage_index"]]
        for arm_id in damage["arm_ids"]:
            points = _arm_polyline(arms[arm_id], crossing["junction_center"], width, height)
            draw.line(points, fill=SHADOW, width=line_width + 3, joint="curve")
            draw.line(points, fill=color, width=line_width, joint="curve")
        anchor = arms[damage["arm_ids"][0]]["tip"]
        draw.text((anchor[0] * width + 5, anchor[1] * height - 8),
                  damage["label"], fill=INK, stroke_width=2, stroke_fill=SHADOW)
    draw.ellipse((jx - radius, jy - radius, jx + radius, jy + radius),
                 fill=JUNCTION_TINT, outline=SHADOW, width=1)
    return image


def crossing_overlay_digest(store, record):
    return stored_preview(draw_crossing_overlay(store.blobs, record), store.blobs)


def preview_crossing(store, record, payload):
    """Return the analyzed crossing proposal without persisting anything."""
    if record["kind"] != "craze":
        raise InputError("Crossing pairing works from a fine-line network measurement.")
    box = crossing_box(payload)
    _, document = analyze_crossing(store, record, box)
    return {
        "box": document["box"],
        "junction_center": document["junction_center"],
        "unit": document["unit"],
        "arms": document["arms"],
        "matching_options": document["matching_options"],
        "direction_matching": document["direction_matching"],
        "tone_matching": document["tone_matching"],
        "bases_conflict": document["bases_conflict"],
        "thresholds": document["thresholds"],
    }


def create_crossing(store, record, expected_version, payload):
    box = crossing_box(payload)
    _, document = analyze_crossing(store, record, box)
    assignment = build_assignment(document, payload)
    crossing = {key: document[key] for key in (
        "box", "junction_center", "unit", "arms", "arms_geometry", "matching_options",
        "direction_matching", "tone_matching", "bases_conflict", "trace_options",
        "thresholds", "extent",
    )}
    crossing["assignment"] = assignment
    picture = crossing_overlay_digest(store, {**record, "crossing": crossing})
    labels = " / ".join(damage["label"] for damage in assignment["damages"])
    saved = store.save_crossing(
        record["id"], expected_version, crossing, picture,
        f"Paired crack crossing into damage numbers {labels}",
    )
    return saved


def update_crossing(store, record, expected_version, payload):
    crossing = record.get("crossing")
    if crossing is None:
        raise InputError("Frame and save the crossing before revising its pairing.")
    assignment = revise_assignment(crossing, crossing["assignment"], payload)
    updated = {**crossing, "assignment": assignment}
    picture = crossing_overlay_digest(store, {**record, "crossing": updated})
    labels = " / ".join(damage["label"] for damage in assignment["damages"])
    saved = store.revise_crossing(
        record["id"], expected_version, updated, picture,
        f"Revised crack crossing pairing ({assignment['basis']} basis) · {labels}",
    )
    return saved
