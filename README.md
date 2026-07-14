# evlab

**Inspect, clean, convert, and benchmark event-camera data from the command line.**

Event-based vision has a preprocessing problem: every sensor and dataset ships a
different container (AEDAT, ROS bags, CSV, NPZ, proprietary `.dat`), every paper
wants a different representation (voxel grids, time surfaces, frames), and every
pipeline reimplements the same denoising filters. `evlab` is the small, boring
tool that handles that mess so you can get to the actual work.

```bash
pip install evlab[viz]

evlab info recording.npz                 # what is this file?
evlab convert events.aedat4 events.npz   # normalize the container
evlab convert drive.dat events.npz       # Prophesee .dat works too
evlab convert flight.bag events.npz      # ...and ROS bags (dvs_msgs)
evlab denoise events.npz clean.npz --filter baf --window 5000
evlab voxel clean.npz voxels.npy --bins 10
evlab visualize clean.npz preview.gif --mode gif
evlab benchmark events.npz clean.npz     # what did the filter do?
evlab corrupt clean.npz bad.npz --emit-recipe r.json   # benchmark generator
```

How good is a denoising filter, really? Generate a labeled stream and score it:

```bash
evlab synth labeled.npz --signal-rate 20000 --noise-rate 8000
evlab denoise-bench labeled.npz --filter baf --window 3000
#   precision     : 99.2%
#   recall        : 37.8%
#   f1            : 0.548
#   noise removed : 99.2%
```

Need a robustness benchmark instead of a clean one? `evlab corrupt` injects
six physically modeled sensor failure modes (leak-event hot pixels, mains
flicker, background-activity bursts, dead regions, readout congestion,
polarity faults) as timed episodes with per-event ground-truth labels, at two
documented severities. Injected timestamps are snapped to the recording's
detected clock quantum, so detectors find the corruption, not the injection.
`--emit-recipe`/`--recipe` make a corrupted benchmark exactly reproducible
from the source recording:

```bash
evlab corrupt drive.npz drive_c.npz --types flicker,burst --severity low \
    --seed 3 --emit-recipe drive_c.recipe.json
evlab corrupt drive.npz drive_c_again.npz --recipe drive_c.recipe.json  # identical
```

## Status

Early alpha. Works and is tested: the canonical representation; loading
NPZ/CSV/TXT, AEDAT4 (`[aedat]` extra), Prophesee legacy `.dat` (with
timestamp-wrap handling), and ROS1/ROS2 bags with `EventArray` topics
(`[ros]` extra); BAF/refractory denoising; voxel grids, time surfaces,
accumulate frames; synthetic labeled streams and precision/recall filter
scoring; six-type corruption injection with reproducible recipes; and the
nine CLI commands above. Planned next: Prophesee EVT3
`.raw`, dataset-aware loaders (via [Tonic]), streaming via [Faery], and
more filters (STCF, IE/YNoise) under `denoise-bench`.

[Tonic]: https://github.com/neuromorphs/tonic
[Faery]: https://github.com/aestream/faery

## Python API

```python
import evlab

data = evlab.load("recording.npz")          # EventData: (x, y, t, p) + sensor size
print(data.event_rate)

from evlab.filters import background_activity_filter
clean = background_activity_filter(data, time_window_us=5000)

from evlab.representations import voxel_grid
grid = voxel_grid(clean, bins=10)            # (10, H, W) float32, ready for torch
```

## Design notes

- One canonical in-memory format: a t-sorted structured numpy array
  (`x: u2, y: u2, t: i8 (µs), p: i1`) plus sensor geometry. Every loader
  normalizes into it; every filter/representation consumes it.
- Zero heavy dependencies in the core (`numpy` + `click`). Visualization,
  AEDAT support, and dataset loaders are opt-in extras.
- Filters return copies; nothing mutates your data behind your back.

## Development

```bash
pip install -e .[dev]
pytest
ruff check src tests
```

## License

MIT
