from pathlib import Path

import numpy as np
import soundfile as sf


def generate_sine_wave(
    frequency: float, duration: float, sample_rate: int
) -> np.ndarray:
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = 0.3 * np.sin(2 * np.pi * frequency * t)
    return wave


def generate_silence(duration: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(sample_rate * duration))


def generate_speech_like_audio(duration: float, sample_rate: int) -> np.ndarray:
    audio_segments: list[np.ndarray] = []
    time_elapsed = 0.0

    speech_frequencies = [200, 300, 400, 250, 350, 280, 320]
    freq_index = 0

    while time_elapsed < duration:
        if np.random.random() > 0.3:
            segment_duration = np.random.uniform(0.5, 2.0)
            frequency = speech_frequencies[freq_index % len(speech_frequencies)]
            segment = generate_sine_wave(frequency, segment_duration, sample_rate)
            freq_index += 1
        else:
            segment_duration = np.random.uniform(0.2, 0.8)
            segment = generate_silence(segment_duration, sample_rate)

        audio_segments.append(segment)
        time_elapsed += segment_duration

    audio = np.concatenate(audio_segments)
    audio = audio[: int(duration * sample_rate)]

    return audio


def create_mock_audio_files(output_dir: str = "data"):
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Test File 1: Short mono audio (10 seconds, 16kHz)
    audio = generate_speech_like_audio(duration=10.0, sample_rate=16000)
    sf.write(output_path / "short_mono_16k.wav", audio, 16000)

    # Test File 2: Short stereo audio (10 seconds, 44.1kHz)
    audio_mono = generate_speech_like_audio(duration=10.0, sample_rate=44100)
    audio_left = audio_mono
    audio_right = audio_mono * 0.9
    audio_stereo = np.column_stack([audio_left, audio_right])
    sf.write(output_path / "short_stereo_44k.wav", audio_stereo, 44100)

    # Test File 3: Medium length (30 seconds, 16kHz, mono)
    audio = generate_speech_like_audio(duration=30.0, sample_rate=16000)
    sf.write(output_path / "medium_mono_16k.wav", audio, 16000)

    # Test File 4: Long audio (2 minutes, 16kHz, mono)
    audio = generate_speech_like_audio(duration=120.0, sample_rate=16000)
    sf.write(output_path / "long_mono_16k.wav", audio, 16000)

    # Test File 5: Edge case - Very short (3 seconds)
    audio = generate_speech_like_audio(duration=3.0, sample_rate=16000)
    sf.write(output_path / "edge_short_3s.wav", audio, 16000)

    # Test File 6: With mostly silence
    audio = np.concatenate(
        [
            generate_speech_like_audio(duration=2.0, sample_rate=16000),
            generate_silence(duration=5.0, sample_rate=16000),
            generate_speech_like_audio(duration=2.0, sample_rate=16000),
        ]
    )
    sf.write(output_path / "mostly_silence.wav", audio, 16000)


if __name__ == "__main__":
    create_mock_audio_files()
