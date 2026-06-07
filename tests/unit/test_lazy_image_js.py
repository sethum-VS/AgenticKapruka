"""Tests for Alpine lazyImage static script."""

from __future__ import annotations

from pathlib import Path

LAZY_IMAGE_JS = Path(__file__).resolve().parent.parent.parent / "static" / "js" / "lazy-image.js"


def test_lazy_image_js_registers_alpine_component() -> None:
    """lazy-image.js defines lazyImage with intersection observer and fade-in state."""
    source = LAZY_IMAGE_JS.read_text()

    assert 'Alpine.data("lazyImage"' in source
    assert "IntersectionObserver" in source
    assert "inView" in source
    assert "loaded" in source
    assert "onLoad" in source
    assert "animate-pulse" not in source
