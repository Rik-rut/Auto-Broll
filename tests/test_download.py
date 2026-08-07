from __future__ import annotations

import io
import json
from pathlib import Path

import httpx
from PIL import Image as PILImage

from broll_agent.config import Settings
from broll_agent.download_searxng import (
    DOWNLOADED_PREFIX,
    download_for_opportunity,
    download_images,
    search_images,
    select_candidates_with_llm,
)
from broll_agent.models import BrollOpportunity


def _opportunity() -> BrollOpportunity:
    return BrollOpportunity(
        id="001",
        slug="puppy-chewing-a-shoe",
        timestamp_start=41.2,
        timestamp_end=47.8,
        context_text="...and this is where he chewed straight through my shoe.",
        reasoning="Concrete, visual anecdote.",
        image_prompt="a golden retriever puppy chewing a leather shoe",
        searxng_query="puppy chewing shoe",
    )


def _png_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, color=(200, 100, 50)).save(buffer, format="PNG")
    return buffer.getvalue()


def _jpeg_bytes(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    PILImage.new("RGB", size, color=(50, 100, 200)).save(buffer, format="JPEG")
    return buffer.getvalue()


def _search_client(searxng_payload: dict) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/search"
        assert request.url.params["format"] == "json"
        assert request.url.params["categories"] == "images"
        return httpx.Response(200, json=searxng_payload)

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_search_images_parses_candidates_and_respects_pool(searxng_payload) -> None:
    cfg = Settings(_env_file=None)
    with _search_client(searxng_payload) as client:
        candidates = search_images("puppy chewing shoe", cfg, client=client)
    # result without img_src is dropped; resolutions are parsed
    assert [c.img_src for c in candidates] == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.png",
        "https://cdn.example.com/d.jpg",
    ]
    assert (candidates[0].width, candidates[0].height) == (1920, 1080)
    assert (candidates[1].width, candidates[1].height) == (800, 600)


def test_search_images_caps_pool_size(searxng_payload) -> None:
    cfg = Settings(_env_file=None, download_candidate_pool_size=2)
    with _search_client(searxng_payload) as client:
        candidates = search_images("puppy", cfg, client=client)
    assert len(candidates) == 2


def test_select_filters_unknown_urls_and_caps_count(fake_llm_factory) -> None:
    cfg = Settings(_env_file=None, download_images_per_broll=2)
    from broll_agent.download_searxng import ImageCandidate

    candidates = [
        ImageCandidate(img_src=f"https://cdn.example.com/{name}.jpg") for name in ("a", "b", "c")
    ]
    llm = fake_llm_factory(
        [
            json.dumps(
                [
                    {"img_src": "https://cdn.example.com/b.jpg", "reason": "sharp"},
                    {"img_src": "https://evil.example.com/not-in-pool.jpg", "reason": "x"},
                    {"img_src": "https://cdn.example.com/b.jpg", "reason": "dupe"},
                    {"img_src": "https://cdn.example.com/a.jpg", "reason": "big"},
                    {"img_src": "https://cdn.example.com/c.jpg", "reason": "extra"},
                ]
            )
        ]
    )
    chosen = select_candidates_with_llm(_opportunity(), candidates, cfg, llm)
    assert [c.img_src for c in chosen] == [
        "https://cdn.example.com/b.jpg",
        "https://cdn.example.com/a.jpg",
    ]
    assert llm.calls == 1


def test_select_falls_back_to_pool_order_on_bad_json(fake_llm_factory) -> None:
    cfg = Settings(_env_file=None, download_images_per_broll=2)
    from broll_agent.download_searxng import ImageCandidate

    candidates = [
        ImageCandidate(img_src=f"https://cdn.example.com/{name}.jpg") for name in ("a", "b", "c")
    ]
    llm = fake_llm_factory(["garbage", "still garbage"])
    chosen = select_candidates_with_llm(_opportunity(), candidates, cfg, llm)
    assert [c.img_src for c in chosen] == [
        "https://cdn.example.com/a.jpg",
        "https://cdn.example.com/b.jpg",
    ]
    assert llm.calls == 2


def test_download_images_backfills_past_failures(tmp_path: Path) -> None:
    cfg = Settings(
        _env_file=None, download_images_per_broll=2, download_min_width=640, download_min_height=640
    )
    big_png = _png_bytes((700, 700))
    small_png = _png_bytes((100, 100))
    big_jpeg = _jpeg_bytes((800, 900))

    def handler(request: httpx.Request) -> httpx.Response:
        match request.url.path:
            case "/error":
                return httpx.Response(500)
            case "/html":
                return httpx.Response(
                    200, content=b"<html>not an image</html>", headers={"content-type": "text/html"}
                )
            case "/small":
                return httpx.Response(200, content=small_png, headers={"content-type": "image/png"})
            case "/good.png":
                return httpx.Response(200, content=big_png, headers={"content-type": "image/png"})
            case "/good.jpg":
                return httpx.Response(
                    200, content=big_jpeg, headers={"content-type": "image/jpeg; charset=binary"}
                )
            case _:
                return httpx.Response(404)

    urls = [
        "https://fake/error",
        "https://fake/html",
        "https://fake/small",
        "https://fake/good.png",
        "https://fake/good.jpg",
    ]
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        saved = download_images(urls, cfg, tmp_path, client=client)
    assert [p.name for p in saved] == ["downloaded_01.png", "downloaded_02.jpg"]
    assert all(path.exists() for path in saved)
    assert count_downloaded(tmp_path) == 2


def count_downloaded(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.name.startswith(DOWNLOADED_PREFIX))


def test_download_for_opportunity_end_to_end(
    tmp_path: Path, searxng_payload, fake_llm_factory
) -> None:
    cfg = Settings(
        _env_file=None,
        searxng_base_url="http://searxng.test",
        download_images_per_broll=1,
        download_min_width=100,
        download_min_height=100,
    )
    big_png = _png_bytes((700, 700))
    search_url = "http://searxng.test/search"

    def handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith(search_url):
            return httpx.Response(200, json=searxng_payload)
        if request.url.host == "cdn.example.com":
            return httpx.Response(200, content=big_png, headers={"content-type": "image/png"})
        return httpx.Response(404)

    llm = fake_llm_factory(
        [json.dumps([{"img_src": "https://cdn.example.com/b.png", "reason": "best"}])]
    )
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        saved = download_for_opportunity(_opportunity(), cfg, llm, tmp_path, http=client)
    assert [p.name for p in saved] == ["downloaded_01.png"]
    assert llm.calls == 1
