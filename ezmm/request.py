from typing import Optional

import aiohttp
import asyncio
import certifi
import ssl

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/123.0.0.0 Safari/537.36",
}

ssl_context = ssl.create_default_context(cafile=certifi.where())

async def is_maybe_image_url(url: str, session: aiohttp.ClientSession) -> bool:
    """Returns True iff the URL points at an accessible _pixel_ image file
    or if the content type is a binary download stream."""
    try:
        headers = await fetch_headers(url, session, timeout=3)
        content_type = headers.get('Content-Type') or headers.get('content-type')
        if content_type.startswith("image/"):
            # Surely an image
            return (not "svg" in content_type and
                    not "eps" in content_type)
        else:
            # If the content is a binary download stream, it may encode an image
            # but also something else. This is a case of "maybe an image"
            return content_type == "binary/octet-stream"

    except Exception:
        return False


async def is_maybe_video_url(url: str, session: aiohttp.ClientSession) -> bool:
    """Returns True iff the URL points at an accessible video file/stream."""
    try:
        headers = await fetch_headers(url, session, timeout=3)
        content_type = headers.get('Content-Type') or headers.get('content-type')
        if content_type.startswith("video/") or content_type == "application/vnd.apple.mpegurl":
            # Surely a video
            return True
        else:
            # If the content is a binary download stream, it may encode a video
            # but also something else. This is a case of "maybe a video"
            return content_type == "binary/octet-stream"

    except Exception:
        return False


async def fetch_headers(url, session: aiohttp.ClientSession, **kwargs) -> dict:
    async with session.head(url, ssl=ssl_context, headers=HEADERS, **kwargs) as response:
        response.raise_for_status()
        return dict(response.headers)


async def request_static(url: str,
                         session: aiohttp.ClientSession,
                         get_text: bool = True,
                         **kwargs) -> Optional[str | bytes]:
    """Downloads the static page from the given URL using aiohttp. If `get_text` is True,
    returns the HTML as text. Otherwise, returns the raw binary content (e.g. an image)."""
    # TODO: Handle web archive URLs
    if url:
        url = str(url)
        try:
            async with session.get(url, timeout=10, headers=HEADERS, allow_redirects=True,
                                   raise_for_status=True, ssl=ssl_context, **kwargs) as response:
                if get_text:
                    return await response.text()  # HTML string
                else:
                    return await stream(response)  # Binary data
        except asyncio.TimeoutError:
            pass  # Server too slow
        except UnicodeError:
            pass  # Page not readable
        except (aiohttp.ClientOSError, aiohttp.ClientConnectorError):
            pass  # Page not available anymore
        except aiohttp.ClientResponseError as e:
            if e.status in [403, 404, 429, 500, 502, 503]:
                # 403: Forbidden access
                # 404: Not found
                # 429: Too many requests
                # 500: Server error
                # 502: Bad gateway
                # 503: Service unavailable (e.g. rate limit)
                pass
            else:
                print(f"\rFailed to retrieve page.\n\t{type(e).__name__}: {e}")
        except Exception as e:
            print(f"\rFailed to retrieve page at {url}.\n\tReason: {type(e).__name__}: {e}")


async def stream(response: aiohttp.ClientResponse, chunk_size: int = 1024) -> bytes:
    data = bytearray()
    async for chunk in response.content.iter_chunked(chunk_size):
        data.extend(chunk)
    return bytes(data)  # Convert to immutable bytes if needed
