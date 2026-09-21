import base64
import binascii
import io
import warnings

from PIL import Image, UnidentifiedImageError

from .errors import ServiceError

MIME_FORMATS = {"image/png": "PNG", "image/jpeg": "JPEG", "image/webp": "WEBP"}


def validate_images(images: list[str]) -> None:
    """Permit verified inline images only: no server-side remote URL fetching."""
    for image in images:
        try:
            header, encoded = image.split(",", 1)
            if header not in {f"data:{mime};base64" for mime in MIME_FORMATS}:
                raise ValueError("unsupported data URL")
            decoded = base64.b64decode(encoded, validate=True)
            if len(decoded) > 8 * 1024 * 1024:
                raise ValueError("image exceeds 8 MiB")
            with warnings.catch_warnings():
                warnings.simplefilter("error", Image.DecompressionBombWarning)
                with Image.open(io.BytesIO(decoded)) as opened:
                    if opened.format != MIME_FORMATS[header[5:-7]]:
                        raise ValueError("MIME type does not match image")
                    if opened.width * opened.height > 16_000_000:
                        raise ValueError("image exceeds 16 megapixels")
                    opened.verify()
        except (
            ValueError,
            binascii.Error,
            OSError,
            UnidentifiedImageError,
            Image.DecompressionBombError,
            Image.DecompressionBombWarning,
        ) as exc:
            raise ServiceError(
                422,
                "invalid_image",
                "Use valid PNG, JPEG or WebP base64 data URLs, at most 8 MiB / 16 MP per image",
            ) from exc
