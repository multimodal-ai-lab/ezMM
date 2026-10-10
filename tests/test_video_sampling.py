"""Tests of the FFmpeg-based video frame sampling for embeddings (no model needed)."""
import subprocess
import sys
import types

import imageio_ffmpeg
import numpy as np
import pytest

from ezmm import embedding
from ezmm.embedding import MAX_FRAME_PIXELS, probe_video, sample_video_frames

FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def ffmpeg(*args):
    subprocess.run([FFMPEG, "-y", "-nostdin", "-v", "error", *map(str, args)], check=True)


def make_video(path, seconds=3, size="160x120", gop=25, extra=()):
    """Writes a small H.264 test video (25 fps) with a keyframe every `gop` frames."""
    ffmpeg("-f", "lavfi", "-i", f"testsrc2=size={size}:rate=25", "-t", seconds, *extra, "-c:v", "libx264",
           "-preset", "ultrafast", "-g", gop, "-pix_fmt", "yuv420p", path)
    return path


def assert_valid(frames, n_frames):
    assert len(frames) == n_frames
    for frame in frames:
        assert isinstance(frame, np.ndarray) and frame.dtype == np.uint8
        assert frame.ndim == 3 and frame.shape[2] == 3
        assert frame.shape == frames[0].shape
        assert frame.shape[0] * frame.shape[1] <= MAX_FRAME_PIXELS


@pytest.mark.parametrize("path", ["in/mountains.mp4", "in/snow.mp4"])
def test_repo_videos(path):
    frames = sample_video_frames(path)
    assert_valid(frames, 4)  # 1 fps of a ~4.4 s video
    assert frames[0].shape == (852, 480, 3)  # 720x1280 (portrait), downscaled, even size
    assert not np.array_equal(frames[0], frames[-1])
    assert frames[0].flags.writeable


def test_rate_and_limit():
    assert_valid(sample_video_frames("in/snow.mp4", fps=0.5), 2)
    assert_valid(sample_video_frames("in/snow.mp4", max_frames=3), 3)
    assert_valid(sample_video_frames("in/snow.mp4", fps=0.01), 1)


def test_long_video_is_limited(tmp_path):
    path = make_video(tmp_path / "long.mp4", seconds=40, size="64x48", gop=25)
    frames = sample_video_frames(path, max_frames=8)
    assert_valid(frames, 8)


def test_sparse_keyframes(tmp_path):
    """Videos with too few keyframes get fully decoded."""
    path = make_video(tmp_path / "one_keyframe.mp4", seconds=4, gop=1000)
    frames = sample_video_frames(path)
    assert_valid(frames, 4)
    assert len({frame.tobytes() for frame in frames}) == 4


def test_large_and_rotated_video(tmp_path):
    path = make_video(tmp_path / "large.mp4", seconds=1, size="1920x1080")
    assert sample_video_frames(path)[0].shape == (480, 852, 3)
    rotated = tmp_path / "rotated.mp4"
    ffmpeg("-display_rotation", 90, "-i", path, "-c", "copy", rotated)
    assert probe_video(rotated)["size"] == (1080, 1920)
    assert sample_video_frames(rotated)[0].shape == (852, 480, 3)


def test_no_video_stream(audio_only_video):
    assert probe_video(audio_only_video) is None
    assert sample_video_frames(audio_only_video) == []
    assert sample_video_frames("in/tone.wav") == []
    assert sample_video_frames("in/does_not_exist.mp4") == []


def test_reserved_color_metadata(tmp_path, monkeypatch):
    """Videos with reserved (invalid) colour metadata, which FFmpeg's filters reject."""
    path = make_video(tmp_path / "source.mp4")
    reserved = tmp_path / "reserved.mp4"
    ffmpeg("-i", path, "-c", "copy", "-bsf:v",
           "h264_metadata=colour_primaries=3:transfer_characteristics=3:matrix_coefficients=3", reserved)
    assert probe_video(reserved)["reserved_color"]
    assert_valid(sample_video_frames(reserved), 3)

    # Without detecting the reserved values upfront, the sampler retries with fixed metadata
    info = probe_video(reserved)
    monkeypatch.setattr(embedding, "probe_video", lambda _: {**info, "reserved_color": False})
    assert_valid(sample_video_frames(reserved), 3)


def test_mixed_interlacing(tmp_path):
    """Concatenated progressive and interlaced segments in an MPEG-TS container."""
    progressive, interlaced = tmp_path / "progressive.ts", tmp_path / "interlaced.ts"
    make_video(progressive, seconds=2, size="320x240", extra=["-x264-params", "repeat-headers=1"])
    ffmpeg("-f", "lavfi", "-i", "testsrc2=size=320x240:rate=50", "-t", 2, "-vf", "tinterlace=interleave_top",
           "-r", 25, "-c:v", "libx264", "-preset", "ultrafast", "-g", 25, "-pix_fmt", "yuv420p",
           "-flags", "+ildct+ilme", "-x264-params", "tff=1:repeat-headers=1", interlaced)
    playlist = tmp_path / "concat.txt"
    playlist.write_text(f"file '{progressive.name}'\nfile '{interlaced.name}'\nfile '{progressive.name}'\n")
    mixed = tmp_path / "mixed.ts"
    ffmpeg("-f", "concat", "-safe", 0, "-i", playlist, "-c", "copy", mixed)
    frames = sample_video_frames(mixed)
    assert len(frames) >= 5  # ~6 s
    assert_valid(frames, len(frames))


def test_corrupted_video(tmp_path):
    path = make_video(tmp_path / "corrupt.mp4", seconds=6, size="320x240")
    data = bytearray(path.read_bytes())
    middle = len(data) // 2
    data[middle:middle + 2048] = bytes((i * 37 + 11) % 256 for i in range(2048))
    path.write_bytes(bytes(data))
    frames = sample_video_frames(path)
    assert len(frames) >= 3
    assert_valid(frames, len(frames))


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    """Replaces torch and sentence_transformers by fakes, recording the model loading calls."""
    calls = []

    class FakeSentenceTransformer:
        cached = True

        def __init__(self, name, local_files_only=False, **kwargs):
            calls.append(local_files_only)
            if local_files_only and not self.cached:
                raise OSError("We couldn't connect to 'https://huggingface.co' ... couldn't find them in the cache")

    torch = types.SimpleNamespace(float32="float32", bfloat16="bfloat16")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "sentence_transformers",
                        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer))
    monkeypatch.setattr(embedding, "is_available", lambda: True)
    monkeypatch.setattr(embedding, "_model", None)
    monkeypatch.setenv("EZMM_DEVICE", "cpu")
    return FakeSentenceTransformer, calls


def test_model_loaded_from_cache(fake_sentence_transformers):
    model_class, calls = fake_sentence_transformers
    assert isinstance(embedding.get_model(), model_class)
    assert calls == [True]  # No online call


def test_model_downloaded_if_not_cached(fake_sentence_transformers):
    model_class, calls = fake_sentence_transformers
    model_class.cached = False
    assert isinstance(embedding.get_model(), model_class)
    assert calls == [True, False]  # Falls back to the online load
