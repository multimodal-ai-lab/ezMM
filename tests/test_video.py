from shutil import copyfile

import numpy as np
import pytest

from ezmm import MultimodalSequence, Video


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
    vid.sample_frames()
    assert_metadata()


def test_embedding():
    vid1 = Video("in/mountains.mp4")
    vid2 = Video("in/snow.mp4")
    print(vid1.embedding)
    print(vid2.embedding)
    print(vid1.cos_sim(vid2))
