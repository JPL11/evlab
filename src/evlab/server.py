"""Local web UI for quick denoise/corrupt runs on a clip.

A single-page app served by FastAPI: drop an event file in any supported
format, pick an operation and its parameters, and get back before/after
renders, summary statistics, and a downloadable .npz result. Start it with
``evlab serve`` (requires ``pip install evlab[ui]``) or the project's
Dockerfile.
"""

import base64
import importlib.util
import os
import tempfile
import uuid
from importlib import resources

from . import formats, metrics
from .corrupt import TYPES, apply_schedule, make_schedule
from .filters import FILTERS
from .formats import EventData

# Cap uploads: event files bigger than this are better handled by the CLI.
MAX_UPLOAD_BYTES = 512 * 1024 * 1024


def _preview_png(data: EventData) -> str | None:
    """Base64 PNG of an accumulate render, or None without matplotlib."""
    if importlib.util.find_spec("matplotlib") is None:
        return None
    from . import viz

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        path = f.name
    try:
        viz.render_frame(data, "accumulate", path)
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    finally:
        os.remove(path)


def _run_denoise(data: EventData, filter_name: str, window: int) -> tuple[EventData, dict]:
    if filter_name == "baf":
        out = FILTERS[filter_name](data, time_window_us=window)
    else:
        out = FILTERS[filter_name](data, refractory_us=window)
    stats = {
        "operation": f"denoise ({filter_name}, window={window} us)",
        "events_in": len(data.events),
        "events_out": len(out.events),
        "retention": metrics.retention(data, out),
    }
    return out, stats


def _run_corrupt(
    data: EventData, types: str, severity: str, coverage: float, seed: int
) -> tuple[EventData, dict]:
    names = sorted(TYPES) if types == "all" else [s.strip() for s in types.split(",")]
    unknown = [n for n in names if n not in TYPES]
    if unknown:
        raise ValueError(f"unknown corruption types {unknown}; choose from {sorted(TYPES)}")
    episodes = make_schedule(
        data.duration_us, names, severity=severity, coverage=coverage, seed=seed
    )
    out = apply_schedule(data, episodes, seed=seed)
    stats = {
        "operation": f"corrupt ({severity}, coverage={coverage}, seed={seed})",
        "events_in": len(data.events),
        "events_out": len(out.events),
        "corrupted_events": int((out.meta["corruption"] != 0).sum()),
        "episodes": len(episodes),
    }
    return out, stats


def create_app():
    """Build the FastAPI app (imports the ui extra lazily)."""
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import FileResponse, HTMLResponse

    app = FastAPI(title="evlab", docs_url=None, redoc_url=None)
    workdir = tempfile.mkdtemp(prefix="evlab-serve-")
    results: dict[str, str] = {}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (resources.files("evlab") / "static" / "index.html").read_text("utf-8")

    @app.post("/api/process")
    async def process(
        file: UploadFile = File(...),
        op: str = Form(...),
        filter_name: str = Form("baf"),
        window: int = Form(5000),
        types: str = Form("all"),
        severity: str = Form("high"),
        coverage: float = Form(0.45),
        seed: int = Form(0),
    ):
        raw = await file.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file too large for the web UI; use the evlab CLI")
        name = os.path.basename(file.filename or "clip")
        stem, ext = os.path.splitext(name)
        src = os.path.join(workdir, f"in-{uuid.uuid4().hex}{ext.lower()}")
        with open(src, "wb") as f:
            f.write(raw)
        try:
            data = formats.load(src)
            if op == "denoise":
                out, stats = _run_denoise(data, filter_name, window)
            elif op == "corrupt":
                out, stats = _run_corrupt(data, types, severity, coverage, seed)
            else:
                raise ValueError(f"unknown operation '{op}' (denoise, corrupt)")
        except ValueError as e:
            raise HTTPException(400, str(e)) from e
        finally:
            os.remove(src)

        token = uuid.uuid4().hex
        dst = os.path.join(workdir, f"out-{token}.npz")
        formats.save(out, dst)
        results[token] = dst
        stats.update(
            {
                "resolution": f"{data.width} x {data.height}",
                "duration_s": round(data.duration_us / 1e6, 3),
            }
        )
        return {
            "stats": stats,
            "preview_before": _preview_png(data),
            "preview_after": _preview_png(out),
            "token": token,
            "filename": f"{stem}-{op}.npz",
        }

    @app.get("/api/result/{token}")
    def result(token: str, name: str = "result.npz"):
        path = results.get(token)
        if path is None or not os.path.exists(path):
            raise HTTPException(404, "result expired or unknown token")
        return FileResponse(path, filename=os.path.basename(name), media_type="application/zip")

    return app
