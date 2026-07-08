"""evlab: inspect, clean, convert, and benchmark event-camera data."""

from .formats import EventData, from_arrays, load, save

__version__ = "0.1.0a1"
__all__ = ["EventData", "from_arrays", "load", "save", "__version__"]
