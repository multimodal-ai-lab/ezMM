import pytest


@pytest.fixture(autouse=True, scope="function")
def run_before_each_test(tmp_path):
    """Lets each test run on a fresh, isolated item registry."""
    from ezmm.common import item_registry
    item_registry.close()
    item_registry.clear_cache()
    item_registry.set_path(tmp_path / "registry")
    yield
    item_registry.close()
    item_registry.clear_cache()


@pytest.fixture
def audio_only_video(tmp_path):
    """An MP4 file without video stream (e.g., from downloading an audio-only HLS variant)."""
    import subprocess

    import imageio_ffmpeg
    path = tmp_path / "audio_only.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-i", "in/tone.wav", "-c:a", "aac", str(path)],
                   check=True)
    return path
