---
name: verify
description: Build, launch, and drive evlab (CLI + web UI) to verify changes end-to-end.
---

# Verifying evlab

## Build and install

```bash
.venv/bin/python -m build --wheel -o /tmp/dist        # needs `pip install build`
python -m venv /tmp/vv && /tmp/vv/bin/pip install "/tmp/dist/evlab-*.whl[ui]"
```

Check the wheel ships non-Python assets: `python -m zipfile -l dist/*.whl | grep static`.

## Drive the CLI

Generate fixtures with synth — no real recordings needed:

```bash
evlab synth clip.npz --pattern moving-bar --duration 2.0 --seed 7
evlab corrupt clip.npz bad.npz --emit-recipe r.json   # expect episodes >= 1 even on short clips
evlab denoise clip.npz out.npz --filter baf
```

## Drive the web UI

```bash
evlab serve --port 8731 &   # GET / must return the page (<title>evlab</title>)
curl -F file=@clip.npz -F op=denoise http://127.0.0.1:8731/api/process
curl -F file=@clip.npz -F op=corrupt -F seed=1 http://127.0.0.1:8731/api/process
# download: GET /api/result/{token}, then formats.load() the npz and check meta['corruption']
pkill -f "evlab serve --port 8731"
```

Good probes: `op=explode`, `types=nonsense`, junk bytes as .npz, unsupported
extension (all → 400), unknown token (404), non-int form field (422), and
`evlab serve` in a venv without the ui extra (clean ClickException).

## Gotchas

- No docker on this desktop — the Dockerfile can't be built locally.
- server.py must NOT use `from __future__ import annotations` (breaks FastAPI
  dependency resolution via ForwardRef).
- matplotlib previews auto-skip when matplotlib is missing (previews null).
