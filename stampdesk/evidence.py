import base64
import csv
from html import escape
from io import BytesIO, StringIO
import json
from pathlib import PurePath
import re
from zipfile import ZIP_DEFLATED, ZipFile
import zlib

from .art import comparison_sheet, load_artifact_picture, stored_preview
from .model import InputError, dictionary, enumeration, flag, string, tags, timestamp, unique_ids
from .store import document, make_id


MAX_ARCHIVE_BYTES = 24 * 1024 * 1024
EXPORT_NAME_PATTERN = re.compile(r"[^A-Za-z0-9._-]+")
ZIP_COMPRESS_LEVEL = 4
ZIP_ENTRY_OVERHEAD = 76  # local header (30) + central record (46), excluding the member name
ZIP_TRAILER_BYTES = 22  # end of central directory record


def filename_stem(value, fallback):
    clean = EXPORT_NAME_PATTERN.sub("-", value).strip("-_.")
    return clean[:65] or fallback


def csv_lines(headers, rows):
    stream = StringIO(newline="")
    out = csv.writer(stream, lineterminator="\r\n")
    out.writerow(headers)
    for row in rows:
        out.writerow([csv_cell(value) for value in row])
    return stream.getvalue().encode("utf-8-sig")


def csv_cell(value):
    if isinstance(value, str) and value and value[0] in "=+-@\t\r":
        return "'" + value
    return document(value) if isinstance(value, (dict, list)) else value


def result_metrics_table(record):
    kind = record["kind"]
    metrics = record["metrics"]
    if kind == "profile":
        columns = (
            "position", "position_unit", "relative_position", "samples", "lightness", "a", "b", "intensity",
        )
        rows = ([item[key] for key in columns] for item in metrics["rows"])
    elif kind == "defects":
        columns = (
            "key", "type", "polarity", "area_pixels", "area_mm2", "normalized_center", "bounds", "peak_contrast",
            "decision", "notes",
        )
        rows = []
        decisions = record.get("decisions", {})
        for marker in metrics["observations"]:
            decision = decisions.get(marker["key"], {})
            row = [marker[key] for key in columns[:-2]]
            rows.append([*row, decision.get("decision", ""), decision.get("notes", "")])
    elif kind == "coverage":
        columns = ("relative_top", "relative_bottom", "measured", "bare_percent", "mixed_percent", "glazed_percent",
                   "unmeasured_percent")
        rows = (
            [band["relative_top"], band["relative_bottom"], band["measured"]]
            + [band["percent"][key] for key in ("bare", "mixed", "glazed", "unmeasured")]
            for band in metrics["depth_bands"]
        )
    elif kind == "color":
        columns = ("cell", "bounds", "pixels", "lab", "delta_from_center", "within_p90", "swatch")
        rows = ([item[key] for key in columns] for item in metrics["cells"])
    elif kind == "texture":
        columns = ("direction", "spacing_px", "spacing_mm", "sample_pairs", "correlation")
        rows = (
            [curve["direction"], point["spacing_px"], point.get("spacing_mm"), point["sample_pairs"], point["correlation"]]
            for curve in metrics["lag_curves"]
            for point in curve["points"]
        )
    elif kind == "rim":
        columns = ("side", "depth_from", "depth_to", "pixels", "median_lab", "median_from_center_delta_e")
        rows = (
            [side["side"], *entry["fraction_from_edge"], entry["pixels"], entry["median_lab"], entry["median_from_center_delta_e"]]
            for side in metrics["sides"]
            for entry in side["swaths"]
        )
    elif kind == "craze":
        columns = ("cell", "pixels", "area_mm2", "bounds")
        rows = ([i, item["pixels"], item["mm2"], item["bounds"]] for i, item in enumerate(metrics["largest_cells"], 1))
    elif kind == "direction":
        columns = ("cell", "median_signed", "strong_percent", "positive_percent")
        rows = ([item[key] for key in columns] for item in metrics["cells"])
    elif kind == "glare":
        columns = ("pixels", "area_mm2", "bounds")
        rows = ([item[key] for key in columns] for item in metrics["hotspots"])
    elif kind == "marking":
        fields = ("condition", "first_station", "last_station", "stations", "midpoint_normalized", "median_cross_width")
        columns = ("unit", *fields)
        rows = ([metrics["unit"], *(item[key] for key in fields)] for item in metrics["runs"])
    elif kind == "joint":
        fields = ("position", "left", "right", "right_minus_left", "status", "normalized_position")
        columns = ("unit", *fields)
        rows = ([metrics["unit"], *(item[key] for key in fields)] for item in metrics["stations"])
    elif kind == "perforation":
        columns = ("punch", "distance", "spacing_from_previous", "relative_position", "luminance")
        rows = ([item[key] for key in columns] for item in metrics["rows"])
    elif kind == "blade":
        fields = ("position", "center", "state", "width", "endpoints")
        columns = ("unit", *fields)
        rows = ([metrics["length_unit"], *(item[key] for key in fields)] for item in metrics["stations"])
    elif kind == "aggregate":
        fields = ("rank", "registered_pixels", "normalized_center", "contrast", "bounding_width", "bounding_height")
        columns = ("unit", *fields)
        rows = ([metrics["unit"], *(item[key] for key in fields)] for item in metrics["candidates"])
    else:
        columns = ("metric", "value")
        rows = [[key, val] for key, val in metrics.items() if not isinstance(val, (dict, list))]
    return csv_lines(columns, rows)


def study_table(report):
    columns = ("tile_name", "tile_id", "group", "measurement_id", "version", "value", "support")
    return csv_lines(columns, ([item[key] for key in columns] for item in report["rows"]))


def source_catalog(store, tile, records):
    """Unique original photographs used by the pinned measurements, keyed by digest."""
    catalog = {}
    for result in records:
        provenance = result["provenance"]
        for capture in (provenance["photo"], provenance["paired_photo"]):
            if not capture:
                continue
            digest = capture["original_digest"]
            if digest in catalog:
                continue
            current = next((item for item in tile["photos"] if item["id"] == capture["id"]), None)
            path = store.blobs.path(digest)
            catalog[digest] = {
                "digest": digest,
                "name": capture["display_name"],
                "role": capture["role"],
                "mime": current["original_mime"] if current else None,
                "bytes": path.stat().st_size if path.is_file() else None,
            }
    return catalog


def candidate_order(records, include_sources, chosen):
    """Source digests to try packing, in the order they should be attached."""
    appearance = []
    by_id = {}
    for result in records:
        provenance = result["provenance"]
        for capture in (provenance["photo"], provenance["paired_photo"]):
            if not capture:
                continue
            digest = capture["original_digest"]
            by_id.setdefault(capture["id"], digest)
            if digest not in appearance:
                appearance.append(digest)
    if chosen is None:
        return appearance if include_sources else []
    order = []
    for photo_id in chosen:
        digest = by_id.get(photo_id)
        if digest is None:
            raise InputError("A chosen source photo is not used by the pinned measurements.")
        if digest not in order:
            order.append(digest)
    return order


def pinned_fields(store, request):
    data = dictionary(request, "Evidence packet")
    targets = unique_ids(data.get("result_ids"), "Measurement", minimum=1, maximum=12)
    title = string(data.get("title"), "Evidence title", maximum=115)
    notes = string(data.get("notes"), "Evidence notes", maximum=2400, required=False)
    view = enumeration(data.get("view", "overlay"), ("overlay", "source"), "Export image")
    include_sources = flag(data.get("include_sources", False), "Include original uploads")
    chosen = data.get("source_photos")
    if chosen is not None:
        chosen = unique_ids(chosen, "Source photo", minimum=0, maximum=10)
    return data, targets, title, notes, view, include_sources, chosen


def image_card(store, result, view):
    provenance = result["provenance"]
    specimen = provenance["specimen"]
    chosen_picture = load_artifact_picture(store.blobs, result["pictures"][view])
    coverage = float(result["support"]["available_percent"])
    return {
        "image": chosen_picture,
        "heading": f"{result['label']} · {specimen['name']}",
        "description": f"Specimen r{result['source_version']} · {coverage:.1f}% support",
    }


def evidence_sheets(store, title, records, view):
    digests = []
    for offset in range(0, len(records), 6):
        selected = records[offset:offset + 6]
        cards = [image_card(store, result, view) for result in selected]
        label = title if len(records) <= 6 else f"{title} · Sheet {offset // 6 + 1}"
        sheet = comparison_sheet(cards, label)
        digests.append(stored_preview(sheet, store.blobs))
    return digests


def original_arcname(source):
    suffix = PurePath(source["name"]).suffix.lower().strip(".")
    suffix = suffix if suffix in ("png", "jpg", "jpeg", "tif", "tiff", "webp") else "img"
    basename = filename_stem(PurePath(source["name"]).stem, "original")
    return f"originals/{basename}-{source['digest'][:10]}.{suffix}"


def measurement_sources(record, catalog, chosen):
    sources = []
    provenance = record["provenance"]
    for capture in (provenance["photo"], provenance["paired_photo"]):
        if not capture:
            continue
        digest = capture["original_digest"]
        info = catalog[digest]
        sources.append({
            "photo_id": capture["id"],
            "name": capture["display_name"],
            "role": capture["role"],
            "digest": digest,
            "bytes": info["bytes"],
            "mime": info["mime"],
            "attachment": "included" if digest in chosen else "omitted",
        })
    return sources


def apply_sources(entry, catalog, order):
    """The evidence record as it looks with the given original digests attached."""
    chosen = set(order)
    return {
        **entry,
        "originals": [
            {key: catalog[digest][key] for key in ("digest", "name", "role", "mime")}
            for digest in order
        ],
        "original_bytes": sum(catalog[digest]["bytes"] for digest in order),
        "measurements": [
            {**record, "sources": measurement_sources(record, catalog, chosen)}
            for record in entry["measurements"]
        ],
    }


def deflated_size(data):
    compressor = zlib.compressobj(ZIP_COMPRESS_LEVEL, zlib.DEFLATED, -15)
    return len(compressor.compress(data) + compressor.flush())


def member_footprint(name, compressed_size):
    return ZIP_ENTRY_OVERHEAD + 2 * len(name.encode("utf-8")) + compressed_size


def is_summary_member(name):
    return name in ("inspection.html", "manifest.json") or (
        name.startswith("measurements/") and name.endswith(".json")
    )


def plan_originals(store, entry, catalog, order, budget):
    """Choose which candidate originals fit the packet budget, in priority order.

    Member sizes are exact deflated sizes, so the plan matches the final archive
    byte for byte; create_evidence still re-measures the real ZIP before saving.
    """
    static = ZIP_TRAILER_BYTES
    for name, data in packet_entries(store, apply_sources(entry, catalog, ())):
        if not is_summary_member(name):
            static += member_footprint(name, deflated_size(data))

    def summary_size(trial_entry):
        return sum(
            member_footprint(name, deflated_size(data))
            for name, data in packet_entries(store, trial_entry)
            if is_summary_member(name)
        )

    footprints = {}
    for digest in order:
        compressed = deflated_size(store.blobs.get(digest))
        footprints[digest] = member_footprint(original_arcname(catalog[digest]), compressed)
    included = []
    for digest in order:
        trial = [*included, digest]
        total = static + summary_size(apply_sources(entry, catalog, trial))
        total += sum(footprints[item] for item in trial)
        if total <= budget:
            included = trial
    return included


def create_evidence(store, identifier, payload):
    data, targets, title, notes, view, include_sources, chosen = pinned_fields(store, payload)
    tile, records = store.sources_for_evidence(identifier, targets)
    if any(result["tile_id"] != tile["id"] for result in records):
        raise InputError("An evidence packet can contain measurements from its own specimen only.")
    catalog = source_catalog(store, tile, records)
    order = candidate_order(records, include_sources, chosen)
    entry = {
        "id": make_id(),
        "created_at": timestamp(),
        "title": title,
        "notes": notes,
        "tags": tags(data.get("tags", [])),
        "tile_id": tile["id"],
        "tile_name": tile["name"],
        "tile_version_at_pin": tile["version"],
        "synthetic": tile["synthetic"],
        "view": view,
        "measurements": records,
        "sheets": evidence_sheets(store, title, records, view),
    }
    included = plan_originals(store, entry, catalog, order, MAX_ARCHIVE_BYTES)
    while True:
        final = apply_sources(entry, catalog, included)
        packed = evidence_zip(store, final)
        if len(packed) <= MAX_ARCHIVE_BYTES:
            break
        if not included:
            raise InputError("The pinned measurements alone exceed the 24 MiB evidence packet limit. Pin fewer measurements.")
        included = included[:-1]
    return store.save_evidence(final), len(packed)


def inline_picture(blobstore, digest):
    return "data:image/png;base64," + base64.b64encode(blobstore.get(digest)).decode("ascii")


def sources_note(record):
    sources = record.get("sources")
    if not sources:
        return ""
    parts = []
    for item in sources:
        label = f"{escape(item['name'])} ({escape(item['role'])}, sha256 {escape(item['digest'][:12])}…"
        if item["attachment"] == "included":
            parts.append(f"{label}, original bytes attached)")
        else:
            parts.append(f"{label}, summary only, original not attached)")
    return f'<p class="sources">Source photographs: {"; ".join(parts)}</p>'


def measurement_article(record, sheet):
    metrics = record["metrics"]
    rows = []
    for key, value in metrics.items():
        if not isinstance(value, (str, bool, float, int)) or len(str(value)) > 180:
            continue
        label = key.replace("_", " ")
        rows.append(f"<tr><th>{escape(label)}</th><td>{escape(str(value))}</td></tr>")
    capture = record["provenance"]["photo"]
    title = escape(record["label"])
    location = escape(capture["display_name"])
    rev = record["source_version"]
    summaries = "".join(rows)
    figures = ""
    if sheet:
        figures = f'<img alt="{title} from {location}" src="{sheet}">'
    return (
        f'<article><h2>{title}</h2><p>{location} · specimen revision {rev}</p>'
        f'{figures}{sources_note(record)}<table>{summaries}</table></article>'
    )


def exported_image_name(record, view):
    return f"measurements/{result_entry_names(record)}-{view}.png"


def specimen_html(store, entry, interactive=True):
    title = escape(entry["title"])
    name = escape(entry["tile_name"])
    date = escape(entry["created_at"])
    notes = escape(entry["notes"]).replace("\n", "<br>") or "No inspection notes."
    badge = "<p>Specimen uses synthetic training material.</p>" if entry["synthetic"] else ""
    measurements = []
    for record in entry["measurements"]:
        digest = record["pictures"][entry["view"]]
        url = inline_picture(store.blobs, digest) if interactive else exported_image_name(record, entry["view"])
        measurements.append(measurement_article(record, url))
    body = "".join(measurements)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
body{{font:16px/1.5 system-ui,sans-serif;max-width:860px;margin:50px auto;padding:0 20px;background:#f9f5eb;color:#252e2d}}
h1,h2{{line-height:1.12}}
h1{{font-size:2.5rem}}
article{{border-top:1px solid #c9c5b8;padding:22px 0;break-inside:avoid}}
img{{max-width:100%;max-height:520px;background:#242b2b}}
table{{border-collapse:collapse;margin:1rem 0;width:100%}}
td,th{{border-bottom:1px solid #ccc;text-align:left;padding:0.45rem 0;vertical-align:top}}
</style></head><body><h1>{title}</h1><p>{name} · {date}</p>{badge}<p>{notes}</p>{body}
<footer>StampDesk · Capture-relative visual evidence</footer></body></html>""".encode("utf8")


def result_entry_names(record):
    stem = filename_stem(record["label"], "measurement")
    return f"{stem}-{record['id'][:8]}"


def artifact_json(payload):
    return (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n").encode("utf8")


def packet_entries(store, entry):
    """Evidence archive members except the original photographs, in archive order."""
    yield "inspection.html", specimen_html(store, entry, interactive=False)
    yield "manifest.json", artifact_json(entry)
    for index, digest in enumerate(entry["sheets"], 1):
        yield f"sheets/sheet-{index:02d}.png", store.blobs.get(digest)
    for record in entry["measurements"]:
        name = result_entry_names(record)
        yield f"measurements/{name}.json", artifact_json(record)
        yield f"measurements/{name}.csv", result_metrics_table(record)
        for view, digest in record["pictures"].items():
            if digest:
                yield exported_image_name(record, view), store.blobs.get(digest)


def evidence_zip(store, entry):
    stream = BytesIO()
    with ZipFile(stream, "w", compression=ZIP_DEFLATED, compresslevel=ZIP_COMPRESS_LEVEL) as zipfile:
        for name, data in packet_entries(store, entry):
            zipfile.writestr(name, data)
        for source in entry["originals"]:
            zipfile.write(store.blobs.require(source["digest"]), original_arcname(source))
    return stream.getvalue()


def summarize_result(record):
    metrics = {
        "color": ("median_lab", "swatch", "delta_e_from_tile_median"),
        "perforation": ("punch_count", "spacing_unit", "median_spacing", "perforations_per_20mm"),
        "coverage": ("percent", "measured"),
        "defects": ("total_observations", "reported_observations"),
        "glare": ("strong_glare_percent", "hotspot_count"),
        "texture": ("texture_orientation", "detail_noise"),
        "craze": ("network_length", "length_unit", "closed_cell_count"),
        "rim": ("all_edges", "rim_band_pixels"),
        "profile": ("line_length", "distance_unit", "supported_stations"),
        "reference": ("reference_title", "delta_e76"),
        "direction": ("left_strong_percent", "right_strong_percent"),
        "marking": ("observed_percent", "midrib_percent_of_observed", "stations"),
        "joint": ("measured_stations", "unmeasured_stations", "median_absolute_step"),
        "blade": ("axis_length", "maximum_width", "observed_stations"),
        "aggregate": ("candidate_clusters", "observed_pixels", "bright_fraction_of_support"),
    }[record["kind"]]
    summary = {key: record[key] for key in (
        "id", "kind", "label", "tile_id", "source_version", "created_at", "version", "basis",
    )}
    summary["tile_name"] = record["provenance"]["specimen"]["name"]
    summary["brief"] = {key: record["metrics"][key] for key in metrics}
    summary["overlay_url"] = f"/api/results/{record['id']}/image/overlay"
    return summary


def summarize_evidence(record):
    keys = ("title", "notes", "id", "tags", "created_at", "tile_id", "tile_name", "tile_version_at_pin", "view")
    summary = {**{key: record[key] for key in keys}, "count": len(record["measurements"]), "sheet_count": len(record["sheets"])}
    summary["originals"] = len(record.get("originals", []))
    sources = [item for result in record["measurements"] for item in result.get("sources", [])]
    if sources:
        summary["sources_attached"] = sum(1 for item in sources if item["attachment"] == "included")
        summary["sources_total"] = len(sources)
    return summary
