"""One 3-channel gain set shared by every plate in a trail batch.

The inspection desk can register only a single set of channel gains for a
batch, and every later reading of that batch has to use it. Because the
plates may have been photographed under different illumination, the one set
will not always suit them all: registration chooses the set that still
serves the most plates, and every reading reports the plates it cannot serve
together with the print area that is actually left for truthful color work.
"""

import numpy as np

from .color import (
    DEFAULT_CARD_TOLERANCE,
    DEFAULT_NEUTRAL_TARGET,
    corrected_card,
    corrected_overflow,
    float_list,
    measure_card,
)
from .geometry import analysis_mask, polygon_mask, region_domain, specimen_frame
from .model import InputError, collection, dictionary, identity, scalar, string

MIN_PRINT_SAMPLES = 64
MIN_REFERENCE_REMAINING_FRACTION = 0.10
MAX_BATCH_MEMBERS = 60
GAIN_LOWER = 0.22
GAIN_UPPER = 4.0
CARD_PURPOSES = ("neutral",)
PRINT_PURPOSES = ("inspection", "glazed")


def first_region(tile, purposes):
    for purpose in purposes:
        found = next((region for region in tile["regions"] if region["purpose"] == purpose), None)
        if found is not None:
            return found
    return None


def card_region(tile):
    return first_region(tile, CARD_PURPOSES)


def print_regions(tile):
    return [region for region in tile["regions"] if region["purpose"] in PRINT_PURPOSES]


def print_mask(frame, tile):
    """The printed face, excluding the neutral card, mount samples and exclusions."""
    regions = print_regions(tile)
    if not regions:
        return None
    mask = np.zeros(frame.valid.shape, dtype=bool)
    for region in regions:
        mask |= polygon_mask(frame.extent, region["vertices"])
    exclusions = np.zeros_like(mask)
    for region in tile["regions"]:
        if region["purpose"] == "ignore":
            exclusions |= polygon_mask(frame.extent, region["vertices"])
    return mask & ~exclusions


def plate_frame(store, tile):
    if not tile.get("plane"):
        raise InputError("A plate needs a marked specimen plane before it can join a batch gain set.")
    return specimen_frame(store.blobs, tile)


def card_median(frame, tile):
    card = card_region(tile)
    if card is None:
        return None
    try:
        return measure_card(frame, card, region_domain(frame.extent, tile))["card"]
    except InputError:
        return None


def plate_spans(frame, tile):
    """Return the card patch measurement and the printed-face mask."""
    card = card_region(tile)
    card_info = None
    card_error = None
    if card is None:
        card_error = "No neutral color card region is marked on this plate."
    else:
        try:
            card_info = measure_card(frame, card, region_domain(frame.extent, tile))
        except InputError as error:
            card_error = str(error)
    printed = print_mask(frame, tile)
    support = None
    if printed is not None:
        support = analysis_mask(frame, printed, minimum=1, allow_clipped=False)
    return card, card_info, card_error, support


def print_statistics(frame, support, gains):
    """Count supported print pixels and those pushed out of range by gains."""
    corrected = frame.linear * gains
    overflow = corrected_overflow(corrected)
    supported = int(np.count_nonzero(support))
    overflowed = int(np.count_nonzero(support & overflow))
    remaining = supported - overflowed
    scale = frame.scale
    return {
        "print_supported_pixels": supported,
        "print_overflow_pixels": overflowed,
        "print_remaining_pixels": remaining,
        "print_remaining_percent": round(100 * remaining / supported, 2) if supported else 0.0,
        "print_remaining_area_mm2": round(remaining * scale[0] * scale[1], 3) if scale else None,
    }


def candidate_gains(entries, target):
    """One gain triple per usable card plus a shared consensus triple.

    `entries` are tuples of (label, card_median, frame, print_support).
    """
    candidates = []
    arrays = [card for _, card, _, _ in entries if card is not None]
    if not arrays:
        return candidates
    consensus = np.median(np.stack(arrays, axis=0), axis=0)
    candidates.append(("consensus", consensus))
    for label, card, _, _ in entries:
        if card is not None:
            candidates.append((f"card:{label}", card))
    usable = []
    for source, median in candidates:
        gains = np.asarray(target, dtype=np.float64) / np.asarray(median, dtype=np.float64)
        if np.max(gains) > GAIN_UPPER or np.min(gains) < GAIN_LOWER:
            continue
        usable.append((source, gains))
    return usable


def card_evaluation(card_median, gains, target, tolerance):
    if card_median is None:
        return None
    return corrected_card(card_median, gains, target, tolerance)


def choose_gains(entries, target, tolerance):
    """Pick the one gain set that serves the most plates truthfully.

    Ties are broken by how much print area survives across the batch, so a
    set that is technically inside tolerance but destroys most of another
    plate's ink is not preferred.
    """
    candidates = candidate_gains(entries, target)
    if not candidates:
        raise InputError("The batch cards would need channel gains outside the supported range.")
    ranked = []
    for source, gains in candidates:
        passing = 0
        worst = 0.0
        total = 0.0
        remaining_fraction = 0.0
        for _, card, frame, print_support in entries:
            check = card_evaluation(card, gains, target, tolerance)
            if check is not None:
                passing += int(check["passed"])
                worst = max(worst, check["max_absolute_deviation"])
                total += check["max_absolute_deviation"]
            if print_support is not None and np.count_nonzero(print_support):
                stats = print_statistics(frame, print_support, gains)
                remaining_fraction += stats["print_remaining_percent"]
        ranked.append((-passing, -remaining_fraction, worst, total, source, gains))
    ranked.sort(key=lambda row: (row[0], row[1], row[2], row[3], row[4]))
    best = ranked[0]
    return best[5], best[4]


def evaluate_member(tile, frame, gains, target, tolerance):
    card, card_info, card_error, print_support = plate_spans(frame, tile)
    printed = print_regions(tile)
    member = {
        "tile_id": tile["id"],
        "name": tile["name"],
        "batch": tile["batch"],
        "photo_id": frame.photo["id"],
        "card_region_id": card["id"] if card else None,
        "card_region_name": card["name"] if card else None,
        "print_region_id": printed[0]["id"] if printed else None,
        "print_region_name": printed[0]["name"] if printed else None,
        "print_region_count": len(printed),
        "passed": False,
        "reason": card_error,
    }
    if card_info is not None:
        check = card_evaluation(card_info["card"], gains, target, tolerance)
        member.update({
            "card_samples": card_info["samples"],
            "card_median_linear_rgb": check["card_median_linear_rgb"],
            "corrected_card_linear_rgb": check["corrected_card_linear_rgb"],
            "channel_deviations": check["channel_deviations"],
            "max_absolute_deviation": check["max_absolute_deviation"],
            "passed": check["passed"],
            "reason": None if check["passed"] else "A linear card channel falls outside the target tolerance.",
        })
    if print_support is None:
        member.update(print_statistics(frame, np.zeros(frame.valid.shape, dtype=bool), gains))
        if not member["reason"]:
            member["reason"] = "This plate has no marked printed face to check for overflow."
    else:
        member.update(print_statistics(frame, print_support, gains))
    return member


def register_batch_calibration(store, payload):
    data = dictionary(payload, "Batch gain set")
    batch = string(data.get("batch"), "Batch or tray", maximum=80)
    target = scalar(data.get("target", DEFAULT_NEUTRAL_TARGET), "Neutral target", lower=0.08, upper=0.8)
    tolerance = scalar(
        data.get("tolerance", DEFAULT_CARD_TOLERANCE), "Card tolerance", lower=0.001, upper=0.05
    )
    requested = data.get("tile_ids")
    candidates = [tile for tile in store.list_tiles() if tile.get("batch") == batch]
    if requested is not None:
        chosen_ids = [identity(item, "Specimen ID") for item in collection(requested, "Specimens", maximum=MAX_BATCH_MEMBERS)]
        if not chosen_ids:
            raise InputError("Choose at least one plate for this batch gain set.")
        if len(set(chosen_ids)) != len(chosen_ids):
            raise InputError("The batch selection repeats a plate.")
        known = {tile["id"]: tile for tile in candidates}
        missing = [item for item in chosen_ids if item not in known]
        if missing:
            raise InputError("Every plate in a batch gain set must carry the same batch or tray label.")
        candidates = [known[item] for item in chosen_ids]
    if not candidates:
        raise InputError("No specimen carries this batch or tray label yet.")
    if len(candidates) > MAX_BATCH_MEMBERS:
        raise InputError(f"A batch gain set covers at most {MAX_BATCH_MEMBERS} plates.")

    entries = []
    usable = 0
    for tile in candidates:
        try:
            frame = plate_frame(store, tile)
        except InputError:
            entries.append((tile["name"], None, None, None))
            continue
        card = card_median(frame, tile)
        printed = print_mask(frame, tile)
        support = None
        if printed is not None:
            try:
                support = analysis_mask(frame, printed, minimum=1, allow_clipped=False)
            except InputError:
                support = None
        entries.append((tile["name"], card, frame, support))
        usable += int(card is not None)

    if not usable:
        raise InputError("No plate in this batch has a usable neutral-card patch to derive gains from.")
    gains, source = choose_gains(entries, target, tolerance)

    # Re-read each plate against the chosen gains so the report reflects the
    # one registered set rather than the candidate that produced it.
    members = []
    for tile in candidates:
        try:
            frame = plate_frame(store, tile)
        except InputError as error:
            members.append({
                "tile_id": tile["id"],
                "name": tile["name"],
                "batch": tile["batch"],
                "passed": False,
                "reason": str(error),
                "print_supported_pixels": 0,
                "print_overflow_pixels": 0,
                "print_remaining_pixels": 0,
                "print_remaining_percent": 0.0,
                "print_remaining_area_mm2": None,
            })
            continue
        members.append(evaluate_member(tile, frame, gains, target, tolerance))
    passing = [item for item in members if item["passed"]]
    failing = [item for item in members if not item["passed"]]
    record = {
        "batch": batch,
        "target": target,
        "tolerance": tolerance,
        "gains": float_list(gains, 7),
        "candidate_source": source,
        "members": members,
        "plate_count": len(members),
        "passing_count": len(passing),
        "failing_count": len(failing),
        "all_passed": not failing,
        "failed_plates": [
            {
                "tile_id": item["tile_id"],
                "name": item["name"],
                "max_absolute_deviation": item.get("max_absolute_deviation"),
                "print_remaining_percent": item.get("print_remaining_percent"),
                "reason": item["reason"],
            }
            for item in failing
        ],
    }
    return store.save_batch_calibration(record)


def plate_verdict(tile, frame, area, record):
    """Evaluate one plate against the registered batch gains.

    Returns the corrected linear image, the overflow mask, the card check (or
    None), the print-face statistics, and the reasons the plate cannot be
    color-matched truthfully. `area` is the region under review; the whole
    printed face is tracked as well, so a gain set that leaves too little of
    the plate usable is rejected even when the chosen patch survives.
    """
    gains = np.asarray(record["gains"], dtype=np.float32)
    linear = frame.linear * gains
    overflow = corrected_overflow(linear)
    chosen = analysis_mask(frame, area, minimum=1, allow_clipped=False)
    face = print_mask(frame, tile)
    face_support = chosen if face is None else analysis_mask(frame, face, minimum=1, allow_clipped=False)
    supported_count = int(np.count_nonzero(face_support))
    remaining_count = int(np.count_nonzero(face_support & ~overflow))
    chosen_count = int(np.count_nonzero(chosen))
    chosen_remaining = int(np.count_nonzero(chosen & ~overflow))
    scale = frame.scale
    card = card_region(tile)
    card_check = None
    card_error = None
    reasons = []
    if card is None:
        card_error = "No neutral color card region is marked on this plate."
    else:
        try:
            info = measure_card(frame, card, region_domain(frame.extent, tile))
            card_check = card_evaluation(info["card"], gains, record["target"], record["tolerance"])
        except InputError as error:
            card_error = str(error)
    if card_error:
        reasons.append(card_error)
    elif card_check is not None and not card_check["passed"]:
        reasons.append(
            "A linear card channel falls outside the batch target of "
            f"{record['target']:.3f} ± {record['tolerance']:.3f}."
        )
    if chosen_remaining < MIN_PRINT_SAMPLES or remaining_count < MIN_PRINT_SAMPLES:
        reasons.append("Too little print area is left after out-of-range pixels are excluded.")
    elif supported_count and remaining_count / supported_count < MIN_REFERENCE_REMAINING_FRACTION:
        reasons.append(
            "Less than "
            f"{MIN_REFERENCE_REMAINING_FRACTION:.0%} of the measured print area survives the batch gains."
        )
    stats = {
        "print_face_pixels": supported_count,
        "print_face_overflow_pixels": supported_count - remaining_count,
        "print_face_remaining_pixels": remaining_count,
        "print_face_remaining_percent": round(100 * remaining_count / supported_count, 2) if supported_count else 0.0,
        "print_face_remaining_area_mm2": round(remaining_count * scale[0] * scale[1], 3) if scale else None,
        "supported_pixels": chosen_count,
        "overflow_pixels": chosen_count - chosen_remaining,
        "remaining_pixels": chosen_remaining,
        "remaining_percent": round(100 * chosen_remaining / chosen_count, 2) if chosen_count else 0.0,
        "remaining_area_mm2": round(chosen_remaining * scale[0] * scale[1], 3) if scale else None,
    }
    return linear, overflow, card, card_check, card_error, stats, reasons


def measurement_batch(store, tile, frame, area, record):
    """Apply the batch gains to one reading and describe what is left."""
    linear, overflow, card, card_check, card_error, stats, reasons = plate_verdict(
        tile, frame, area, record
    )
    member = next((item for item in record["members"] if item["tile_id"] == tile["id"]), None)
    report = {
        "scope": "batch",
        "batch": record["batch"],
        "target": record["target"],
        "tolerance": record["tolerance"],
        "gains": list(record["gains"]),
        "candidate_source": record.get("candidate_source"),
        "registered": member is not None,
        "plate_passed": not reasons,
        "member_passed": member["passed"] if member is not None else not reasons,
        "plate_reason": " ".join(reasons) if reasons else None,
        "card_region_id": card["id"] if card else None,
        "card_error": card_error,
        **stats,
        "plate_count": record["plate_count"],
        "passing_count": record["passing_count"],
        "failing_count": record["failing_count"],
        "all_passed": record["all_passed"],
        "failed_plates": record.get("failed_plates", []),
    }
    if card_check is not None:
        report.update({
            "card_median_linear_rgb": card_check["card_median_linear_rgb"],
            "corrected_card_linear_rgb": card_check["corrected_card_linear_rgb"],
            "channel_deviations": card_check["channel_deviations"],
            "max_absolute_deviation": card_check["max_absolute_deviation"],
        })
    return linear, overflow, report


def approval_gate(tile, frame, area, record):
    """Reasons this plate cannot be a truthful batch reference, if any."""
    linear, overflow, _, _, _, _, reasons = plate_verdict(tile, frame, area, record)
    return linear, overflow, reasons


def summarize_batch(record):
    return {
        key: record[key]
        for key in (
            "batch", "target", "tolerance", "gains", "candidate_source", "plate_count",
            "passing_count", "failing_count", "all_passed", "failed_plates", "updated_at",
        )
    }
