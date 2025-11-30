from shutil import copyfile

import aiohttp
import numpy as np
import pytest

from ezmm import MultimodalSequence, Video, download_video


def test_video():
    vid = Video("in/mountains.mp4")
    print(vid)


def test_video_equality():
    # Duplicate image file
    copyfile("in/mountains.mp4", "in/mountains_copy.mp4")

    vid1 = Video("in/mountains.mp4")
    vid2 = Video("in/mountains_copy.mp4")
    assert vid1 == vid2
    assert vid1 is not vid2


def test_videos_in_sequence():
    vid1 = Video("in/mountains.mp4")
    vid2 = Video("in/snow.mp4")
    seq = MultimodalSequence("The videos", vid1, vid2, "show scenes in the Alps.")
    print(seq)
    videos = seq.videos
    assert len(videos) == 2
    assert vid1 in videos
    assert vid2 in videos
    assert vid1 in seq
    assert vid2 in seq


def test_binary():
    with open("in/mountains.mp4", "rb") as f:
        binary_data = f.read()
    vid = Video(binary_data=binary_data)
    print(vid)


def test_base64():
    vid = Video("in/mountains.mp4")
    print(len(vid.get_base64_encoded()))


# TODO: Move downloading skills to ScrapeMM
@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8",
    "https://devstreaming-cdn.apple.com/videos/streaming/examples/adv_dv_atmos/main.m3u8",
    "https://video.bsky.app/watch/did%3Aplc%3Alvs2rrkrj6usatuglfukwoea/bafkreibdgmt4y3z62opupxdykw53ftvkyoprzxuztzocxqfe2hjskziq44/playlist.m3u8",
])
async def test_download_m3u8(url):
    async with aiohttp.ClientSession() as session:
        vid = await download_video(url, session)
        assert isinstance(vid, Video)


# TODO: Move downloading skills to ScrapeMM
@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/a7/How_to_make_video.webm/How_to_make_video.webm.1080p.vp9.webm",
    "https://video.twimg.com/amplify_video/1982791531959332864/vid/avc1/720x892/ByktcD0YE_6cV89u.mp4",
])
async def test_download_mp4_webm(url):
    async with aiohttp.ClientSession() as session:
        vid = await download_video(url, session)
        assert isinstance(vid, Video)


@pytest.mark.parametrize("path", ["in/mountains.mp4", "in/snow.mp4"])
@pytest.mark.parametrize("n_frames", [1, 5, 10])
def test_frame_sampling(path: str, n_frames: int):
    vid = Video(path)
    frames = vid.sample_frames(n_frames=n_frames)
    assert len(frames) == n_frames
    assert isinstance(frames[0], np.ndarray)


@pytest.mark.parametrize("path", ["in/mountains.mp4"])
def test_metadata(path: str):
    def assert_metadata():
        assert vid.width
        assert vid.height
        assert vid.frame_count
        assert vid.fps
        assert vid.duration

    vid = Video(path)
    assert_metadata()
    vid.close()
    assert_metadata()
    vid.sample_frames()
    assert_metadata()

