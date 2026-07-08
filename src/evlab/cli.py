"""evlab command line interface."""

from __future__ import annotations

import json

import click
import numpy as np

from . import formats, metrics
from .filters import FILTERS
from .representations import voxel_grid


@click.group()
@click.version_option(package_name="evlab")
def main():
    """Inspect, clean, convert, and benchmark event-camera data."""


@main.command()
@click.argument("path", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def info(path, as_json):
    """Summarize an event file (count, resolution, duration, rate)."""
    data = formats.load(path)
    stats = metrics.summary(data)
    if as_json:
        click.echo(json.dumps(stats, indent=2))
        return
    click.echo(f"{path}")
    click.echo(f"  events      : {stats['num_events']:,}")
    click.echo(f"  resolution  : {stats['width']} x {stats['height']}")
    click.echo(f"  duration    : {stats['duration_s']:.3f} s")
    click.echo(f"  event rate  : {stats['event_rate_hz']:,.0f} ev/s")
    if stats["num_events"]:
        click.echo(f"  polarity ON : {stats['polarity_balance']:.1%}")
        click.echo(
            f"  active px   : {stats['active_pixels']:,}"
            f" ({stats['active_pixel_fraction']:.1%} of sensor)"
        )


@main.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.argument("dst", type=click.Path(dir_okay=False))
def convert(src, dst):
    """Convert between event file formats (npz, csv, txt, aedat4-in)."""
    data = formats.load(src)
    formats.save(data, dst)
    click.echo(f"wrote {len(data.events):,} events -> {dst}")


@main.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.argument("dst", type=click.Path(dir_okay=False))
@click.option(
    "--filter",
    "filter_name",
    type=click.Choice(sorted(FILTERS)),
    default="baf",
    show_default=True,
    help="Denoising filter to apply.",
)
@click.option(
    "--window",
    type=int,
    default=5000,
    show_default=True,
    help="Time window in microseconds (baf) / refractory period (refractory).",
)
def denoise(src, dst, filter_name, window):
    """Denoise an event stream and write the result."""
    data = formats.load(src)
    if filter_name == "baf":
        out = FILTERS[filter_name](data, time_window_us=window)
    else:
        out = FILTERS[filter_name](data, refractory_us=window)
    formats.save(out, dst)
    kept = metrics.retention(data, out)
    click.echo(f"kept {len(out.events):,}/{len(data.events):,} events ({kept:.1%}) -> {dst}")


@main.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.argument("dst", type=click.Path(dir_okay=False))
@click.option("--bins", type=int, default=10, show_default=True, help="Temporal bins.")
def voxel(src, dst, bins):
    """Build a (bins, H, W) voxel grid and save it as .npy."""
    data = formats.load(src)
    grid = voxel_grid(data, bins=bins)
    np.save(dst, grid)
    click.echo(f"voxel grid {grid.shape} (sum={grid.sum():+.1f}) -> {dst}")


@main.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.argument("dst", type=click.Path(dir_okay=False))
@click.option(
    "--mode",
    type=click.Choice(["accumulate", "time-surface", "gif"]),
    default="accumulate",
    show_default=True,
)
@click.option(
    "--window", type=float, default=33.0, show_default=True, help="Frame window in ms (gif mode)."
)
@click.option(
    "--tau", type=float, default=30.0, show_default=True, help="Time-surface decay in ms."
)
def visualize(src, dst, mode, window, tau):
    """Render events to a PNG (accumulate/time-surface) or GIF."""
    from . import viz

    data = formats.load(src)
    if mode == "gif":
        n = viz.render_gif(data, dst, window_ms=window)
        click.echo(f"wrote {n} frames -> {dst}")
    else:
        viz.render_frame(data, mode, dst, tau_us=tau * 1000)
        click.echo(f"wrote {mode} image -> {dst}")


@main.command()
@click.argument("reference", type=click.Path(exists=True, dir_okay=False))
@click.argument("other", type=click.Path(exists=True, dir_okay=False))
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def benchmark(reference, other, as_json):
    """Compare two event streams (e.g. raw vs. denoised)."""
    ref = formats.load(reference)
    oth = formats.load(other)
    result = metrics.compare(ref, oth)
    if as_json:
        click.echo(json.dumps(result, indent=2))
        return
    click.echo(f"retention        : {result['retention']:.1%}")
    click.echo(f"event rate (ref) : {result['event_rate_ref_hz']:,.0f} ev/s")
    click.echo(f"event rate (new) : {result['event_rate_other_hz']:,.0f} ev/s")
    click.echo(f"structure (ref)  : {result['structure_ref']:.3f}")
    click.echo(f"structure (new)  : {result['structure_other']:.3f}")


if __name__ == "__main__":
    main()
