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
evlab denoise events.npz clean.npz --filter baf --window 5000
evlab voxel clean.npz voxels.npy --bins 10
evlab visualize clean.npz preview.gif --mode gif
evlab benchmark events.npz clean.npz     # what did the filter do?
```

## Status

Early alpha. The canonical representation, NPZ/CSV/TXT round-trip, AEDAT4
reading, BAF/refractory denoising, voxel grids, time surfaces, and the six CLI
commands above work and are tested. Planned next: Prophesee `.dat`/`.raw`,
ROS bag extraction, dataset-aware loaders (via [Tonic]), streaming via
[Faery], and denoising quality benchmarks against labeled datasets.

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
