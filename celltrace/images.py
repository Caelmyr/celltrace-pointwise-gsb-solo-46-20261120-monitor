from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePath
import warnings

import numpy as np
from PIL import Image, UnidentifiedImageError

from .model import Capture, Issue, MAX_SOURCE_PIXELS, MAX_UPLOAD_BYTES


Image.MAX_IMAGE_PIXELS = MAX_SOURCE_PIXELS


@dataclass(frozen=True)
class SensorImage:
    pixels: np.ndarray
    format: str
    bit_depth: int
    digest: str
    byte_size: int

    @property
    def width(self):
        return self.pixels.shape[1]

    @property
    def height(self):
        return self.pixels.shape[0]

    @property
    def maximum(self):
        return (1 << self.bit_depth) - 1


def decode_source(content: bytes) -> SensorImage:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise Issue("Source images must be nonempty and no larger than 22 MiB.", 413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(content)) as image:
                fmt = image.format
                if fmt not in ("PNG", "TIFF"):
                    raise Issue("Use original grayscale PNG or TIFF images.")
                if getattr(image, "n_frames", 1) != 1:
                    raise Issue("Multi-page images need to be separated before upload.")
                if min(image.size) < 32 or image.width * image.height > MAX_SOURCE_PIXELS:
                    raise Issue("Source images must be at least 32×32 and contain at most 8 million pixels.")
                if image.getexif().get(274, 1) != 1:
                    raise Issue("Reset image orientation metadata before upload.")
                depth = source_depth(image)
                target = np.uint8 if depth == 8 else np.uint16
                values = np.array(image)
                if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer):
                    raise Issue("The image must contain integer grayscale sensor samples.")
                if values.min() < 0 or values.max() > (1 << depth) - 1:
                    raise Issue("Source samples exceed the supported bit range.")
                array = np.asarray(values, dtype=target)
    except Issue:
        raise
    except (OSError, TypeError, ValueError, SyntaxError, EOFError, UnidentifiedImageError,
            Image.DecompressionBombError, Image.DecompressionBombWarning):
        raise Issue("The image could not be decoded as a supported grayscale PNG or TIFF.") from None
    array.setflags(write=False)
    return SensorImage(array, fmt, depth, sha256(content).hexdigest(), len(content))


def source_depth(image: Image.Image) -> int:
    if image.format == "PNG":
        if image.mode == "L":
            return 8
        if image.mode in ("I", "I;16", "I;16B", "I;16L"):
            return 16
    else:
        tags = image.tag_v2
        bits = tags.get(258, (8,))
        bits = bits[0] if isinstance(bits, tuple) and len(bits) == 1 else bits
        samples = tags.get(277, 1)
        photo = tags.get(262, 1)
        sample_format = tags.get(339, (1,))
        if isinstance(sample_format, tuple) and len(sample_format) == 1:
            sample_format = sample_format[0]
        if samples == 1 and photo == 1 and sample_format == 1 and bits in (8, 16):
            return bits
    raise Issue("Use single-channel 8-bit or unsigned 16-bit grayscale samples with black at zero.")


def safe_name(name: str, default: str) -> str:
    normalized = PurePath(name.replace("\\", "/")).name.strip()
    clean = "".join(char for char in normalized if ord(char) >= 32 and ord(char) != 127)
    return clean[:120] or default


def verify_capture(image: SensorImage, capture: Capture):
    if (
        image.digest != capture.digest
        or image.byte_size != capture.byte_size
        or image.width != capture.width
        or image.height != capture.height
        or image.bit_depth != capture.bit_depth
        or image.format != capture.format
    ):
        raise Issue("Original bytes do not match the stored capture metadata.", 409)


def png_bytes(image: Image.Image) -> bytes:
    target = BytesIO()
    image.save(target, format="PNG", compress_level=4)
    return target.getvalue()


def tiff_sensor(array: np.ndarray) -> bytes:
    target = BytesIO()
    Image.fromarray(array.astype(np.uint16)).save(target, format="TIFF", compression="tiff_deflate")
    return target.getvalue()


def luma_display(values: np.ndarray, lower: float, upper: float, gamma: float) -> np.ndarray:
    clean = np.nan_to_num(values, nan=lower, posinf=upper, neginf=lower)
    scaled = np.clip((clean - lower) / (upper - lower), 0, 1)
    return np.rint(255 * np.power(scaled, gamma)).astype(np.uint8)


def set_invalid_pattern(rgb: np.ndarray, valid: np.ndarray):
    y, x = np.ogrid[:valid.shape[0], :valid.shape[1]]
    mark = ((x // 6 + y // 6) % 2).astype(bool)
    rgb[~valid & mark] = (39, 31, 49)
    rgb[~valid & ~mark] = (24, 21, 30)


def grayscale_preview(values: np.ndarray, valid: np.ndarray,
                      lower: float, upper: float, gamma: float) -> Image.Image:
    lum = luma_display(values, lower, upper, gamma)
    if valid.all():
        return Image.fromarray(lum, mode="L").convert("RGB")
    rgb = np.repeat(lum[..., None], 3, axis=2)
    set_invalid_pattern(rgb, valid)
    return Image.fromarray(rgb, mode="RGB")


def continuous_palette(values: np.ndarray, bounds: tuple[float, float],
                       colors: tuple[tuple[int, int, int], ...]) -> np.ndarray:
    lower, upper = bounds
    index = np.linspace(lower, upper, len(colors))
    return np.stack(
        [np.interp(values, index, [color[channel] for color in colors])
         for channel in range(3)],
        axis=-1,
    ).astype(np.uint8)


def mixed_overlay(base: Image.Image, mask: np.ndarray, color: tuple, opacity: float) -> Image.Image:
    pixels = np.array(base, dtype=np.uint8)
    pixels[mask] = np.rint(pixels[mask] * (1 - opacity) + np.asarray(color) * opacity).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def compact_thumbnail(source: Image.Image, longest: int = 310) -> Image.Image:
    thumbnail = source.copy()
    thumbnail.thumbnail((longest, longest), Image.Resampling.LANCZOS)
    return thumbnail


def bounds_for_mask(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.flatnonzero(np.any(mask, axis=1))
    cols = np.flatnonzero(np.any(mask, axis=0))
    if rows.size == 0:
        return None
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def histogram_rows(values: np.ndarray, bounds: tuple[float, float], bins: int) -> list[dict]:
    finite = values[np.isfinite(values)]
    low, high = bounds
    counts, edges = np.histogram(np.clip(finite, low, high), bins=bins, range=bounds)
    return [
        {"from": round(float(edges[index]), 5), "to": round(float(edges[index + 1]), 5), "pixels": int(count)}
        for index, count in enumerate(counts)
    ]


def distribution(values: np.ndarray) -> dict:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0, "mean": None, "min": None, "max": None,
            "median": None, "p10": None, "p90": None, "mad": None,
        }
    p10, median, p90 = np.percentile(finite, (10, 50, 90))
    return {
        "count": int(finite.size),
        "mean": round(float(np.mean(finite)), 6),
        "min": round(float(np.min(finite)), 6),
        "max": round(float(np.max(finite)), 6),
        "median": round(float(median), 6),
        "p10": round(float(p10), 6),
        "p90": round(float(p90), 6),
        "mad": round(float(np.median(np.abs(finite - median))), 6),
    }
