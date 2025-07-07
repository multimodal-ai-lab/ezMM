from shutil import copyfile

import aiohttp
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


@pytest.mark.asyncio
@pytest.mark.parametrize("url", [
    "https://upload.wikimedia.org/wikipedia/commons/transcoded/a/a7/How_to_make_video.webm/How_to_make_video.webm.1080p.vp9.webm",
    "https://demo.unified-streaming.com/k8s/features/stable/video/tears-of-steel/tears-of-steel.ism/.m3u8"
])
async def test_download_video(url):
    async with aiohttp.ClientSession() as session:
        vid = await download_video(url, session)
        print(vid)
        assert isinstance(vid, Video)
