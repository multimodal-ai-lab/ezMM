import base64
from pathlib import Path

import pytest

from ezmm import Audio, Item, MultimodalSequence


def test_audio():
    audio = Audio("in/tone.wav")
    assert audio.kind == "audio"
    assert audio.reference == f"<audio:{audio.id}>"


def test_metadata():
    audio = Audio("in/tone.wav")
    assert audio.duration == pytest.approx(1.0, abs=0.01)
    assert audio.sample_rate == 16000
    assert audio.channels == 1
    assert audio.bitrate > 0
    assert "audio" in audio.mime_type


def test_binary():
    data = Path("in/tone.wav").read_bytes()
    audio = Audio(binary_data=data)
    assert audio.file_path.suffix == ".wav"  # Format is detected from the content
    assert audio.duration == pytest.approx(1.0, abs=0.01)


def test_binary_with_mime_type():
    audio = Audio(binary_data=Path("in/tone.wav").read_bytes(), mime_type="audio/wav")
    assert audio.file_path.suffix == ".wav"


def test_base64():
    audio = Audio("in/tone.wav")
    assert base64.b64decode(audio.get_base64_encoded()) == Path("in/tone.wav").read_bytes()


def test_html():
    audio = Audio("in/tone.wav")
    assert "<audio" in audio.as_html()


def test_audio_in_sequence():
    audio = Audio("in/tone.wav")
    seq = MultimodalSequence("Listen to this:", audio)
    assert seq.has_audios()
    assert seq.audios == [audio]
    assert MultimodalSequence(str(seq)) == seq
    assert Item.from_reference(audio.reference) is audio
