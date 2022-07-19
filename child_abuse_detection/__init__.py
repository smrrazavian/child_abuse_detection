# type: ignore[attr-defined]
"""child abuse detection based on python and bag of words."""

from importlib import metadata as importlib_metadata


def get_version() -> str:
    """gets the version of project."""
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()
