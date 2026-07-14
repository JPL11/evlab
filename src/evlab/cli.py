"""evlab command line interface."""

from __future__ import annotations

import json

import click
import numpy as np

from . import formats, metrics
from .filters import FILTERS
from .representations import voxel_grid
from .corrupt import SEVERITIES, TYPES, apply_schedule, load_recipe, make_schedule, save_recipe
from .synth import GENERATORS


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
@click.argument("dst", type=click.Path(dir_okay=False))
@click.option(
    "--pattern", type=click.Choice(sorted(GENERATORS)), default="moving-bar", show_default=True
)
@click.option("--resolution", default="240x240", show_default=True, help="WIDTHxHEIGHT.")
@click.option("--duration", type=float, default=1.0, show_default=True, help="Seconds.")
@click.option("--signal-rate", type=float, default=20000, show_default=True, help="Signal ev/s.")
@click.option("--noise-rate", type=float, default=5000, show_default=True, help="Noise ev/s.")
@click.option("--seed", type=int, default=0, show_default=True)
def synth(dst, pattern, resolution, duration, signal_rate, noise_rate, seed):
    """Generate a synthetic stream with ground-truth signal/noise labels."""
    width, _, height = resolution.partition("x")
    data = GENERATORS[pattern](
        width=int(width),
        height=int(height),
        duration_us=int(duration * 1e6),
        signal_rate_hz=signal_rate,
        noise_rate_hz=noise_rate,
        seed=seed,
    )
    formats.save(data, dst)
    n_sig = int(data.meta["signal"].sum())
    click.echo(f"wrote {len(data.events):,} events ({n_sig:,} signal) -> {dst}")


@main.command("denoise-bench")
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.option(
    "--filter", "filter_name", type=click.Choice(sorted(FILTERS)), default="baf", show_default=True
)
@click.option(
    "--window",
    type=int,
    default=5000,
    show_default=True,
    help="Time window (baf) / refractory period, microseconds.",
)
@click.option("--json", "as_json", is_flag=True, help="Emit machine-readable JSON.")
def denoise_bench(src, filter_name, window, as_json):
    """Score a denoising filter against a labeled stream (see `evlab synth`)."""
    from .filters import MASKS

    data = formats.load(src)
    if filter_name == "baf":
        mask = MASKS[filter_name](data, time_window_us=window)
    else:
        mask = MASKS[filter_name](data, refractory_us=window)
    score = metrics.denoise_score(data, mask)
    if as_json:
        click.echo(json.dumps(score, indent=2))
        return
    click.echo(f"{filter_name} (window={window} us) on {src}")
    click.echo(f"  precision     : {score['precision']:.1%}")
    click.echo(f"  recall        : {score['recall']:.1%}")
    click.echo(f"  f1            : {score['f1']:.3f}")
    click.echo(f"  noise removed : {score['noise_removed']:.1%}")


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


@main.command()
@click.argument("src", type=click.Path(exists=True, dir_okay=False))
@click.argument("dst", type=click.Path(dir_okay=False))
@click.option(
    "--types",
    default="all",
    show_default=True,
    help="Comma-separated corruption types, or 'all'. Types: " + ", ".join(sorted(TYPES)),
)
@click.option(
    "--severity", type=click.Choice(["low", "high"]), default="high", show_default=True
)
@click.option(
    "--coverage",
    type=float,
    default=0.45,
    show_default=True,
    help="Fraction of the recording covered by corruption episodes.",
)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option(
    "--recipe",
    "recipe_in",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Replay a previously emitted recipe instead of drawing a schedule.",
)
@click.option(
    "--emit-recipe",
    "recipe_out",
    type=click.Path(dir_okay=False),
    default=None,
    help="Write the episode schedule (exact onsets, params, seed) as JSON.",
)
def corrupt(src, dst, types, severity, coverage, seed, recipe_in, recipe_out):
    """Inject physically modeled sensor corruptions with exact ground truth.

    Six failure modes (hot pixels, flicker, background-activity bursts, dead
    regions, readout congestion, polarity faults) are injected as timed
    episodes. Every output event is labeled clean or by corruption type, and
    injected timestamps are snapped to the recording's clock grid so the
    injection itself is not detectable. Use --emit-recipe / --recipe to make
    a benchmark exactly reproducible.
    """
    data = formats.load(src)
    if recipe_in:
        episodes, seed = load_recipe(recipe_in)
    else:
        names = sorted(TYPES) if types == "all" else [s.strip() for s in types.split(",")]
        for name in names:
            if name not in TYPES:
                raise click.BadParameter(f"unknown type '{name}'; choose from {sorted(TYPES)}")
        episodes = make_schedule(
            data.duration_us, names, severity=severity, coverage=coverage, seed=seed
        )
    result = apply_schedule(data, episodes, seed=seed)
    if not dst.endswith(".npz"):
        raise click.BadParameter("dst must be .npz to preserve ground-truth labels")
    formats.save(result, dst)
    if recipe_out:
        save_recipe(result.meta["schedule"], seed, recipe_out)
    labels = result.meta["corruption"]
    n_cor = int((labels != 0).sum())
    click.echo(
        f"wrote {len(result.events):,} events ({n_cor:,} corrupted, "
        f"{len(episodes)} episodes) -> {dst}"
    )
