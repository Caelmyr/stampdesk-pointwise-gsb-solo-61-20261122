# StampDesk

StampDesk is a local image workbench for inspecting mounted stamps and joined strips. It retains each upload, registers repeat photographs to the same plane and links measured evidence to the images used.

## Start

Run `docker compose up --build --wait` and open the loopback address at port `19971`. `STAMPDESK_PORT` selects another port. Stopping its Compose service preserves the named project volume. The native entry point is `./start.sh` with Python 3.11 and the pinned dependencies from `requirements.txt`.

The build can use an already available dependency image with the identical Python package versions, set via `STAMPDESK_DEPS_IMAGE`. No training samples are required for import. The training button generates four clearly labeled synthetic stamp trays, with diffuse, polarized and raking photographs of each.

## Use the workbench

- In Captures, upload an image, mark the four plane corners and enter physical dimensions if available. Register each repeat photograph to the same plane. Compare views side by side or as a checkerboard and inspect clipped or missing regions before measuring.
- Sampling regions defines inspection polygons, printed stock and mount references, a neutral card and optional exclusions. Native source files remain downloadable.
- Measure includes color variation, ink coverage, spots and elongated marks, polarization, opposing illumination, texture, thin-line networks, margin color, line profiles, printed-strip widths, printed-rule visibility, mount seams and pale ink flecks. Approved reference matching preserves the original source and sampling basis for later review.
- A fine-line network result can frame one four-arm crack crossing and split its traced threads into two archive damage numbers. Each visible arm is assigned to exactly one number; unsupported, unphotographed gaps are never drawn as crack. Direction continuity and matching depth/tone are scored separately, so when they disagree you can adopt one basis and keep the set-aside reading visible for review. After checking earlier photographs the pairing may be switched once; both damage numbers, their lengths and normalized image fragments update together, the previous pairing is retained in a revision history, and reopening the result still shows the adopted basis.
- In Perforation gauge, place two points along one physical row of visible holes. Select their light or dark polarity and run the reading. The overlay marks the observed centers and the table shows the individual intervals. The reported gauge uses a median of measured adjacent intervals. Gaps in capture support do not become measured intervals. Pixels are used when the physical scale is unavailable.
- In Geometry, mark a path or area on the same plane. Evidence stores a snapshot and can include the actual original photographs. Received evidence verifies the archive manifest and opens a local copy.

Supported input includes PNG, JPEG, WebP and single-frame TIFF up to 24 MiB and 12 megapixels. The registered plane is limited to 1800 pixels on its longest side. A specimen holds up to ten photographs and 24 regions. Raking and polarized views make observed image changes easier to inspect. The workbench does not infer absolute relief or decide stamp authenticity. Perforation gauge measures one marked band at a time and does not fill hidden punches or infer whether two rows belong to the same printing plate.

StampDesk reuses the local LeafLedger image registration, evidence and measurement libraries. The printed training trays and the edge perforation gauge are StampDesk code.
