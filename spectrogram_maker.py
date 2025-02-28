import argparse
import os
from typing import List, Optional, Tuple

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Settings
sr = 44100  # sample rate
max_duration = 4.5  # seconds

# Parameters for spectrogram
window_duration = 0.03  # 0.03 seconds
window_length = int(window_duration * sr)  # number of samples in the window
noverlap = int(0.99 * window_length)  # overlap
nfft = 512  # Changed from 2048 to 512 to match MATLAB implementation
hop_length = window_length - noverlap  # hop length


def time_stretch(audio: np.ndarray, rate: float = 1.0) -> np.ndarray:
    """Apply time stretching to audio.

    Args:
        audio: Audio signal.
        rate: Stretching rate. If rate > 1, then the signal is sped up.
              If rate < 1, then the signal is slowed down.

    Returns:
        np.ndarray: Time-stretched audio.
    """
    return librosa.effects.time_stretch(audio, rate=rate)


def pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    """Apply pitch shifting to audio.

    Args:
        audio: Audio signal.
        sr: Sample rate.
        n_steps: Number of semitones to shift. Positive values shift up, negative values shift down.

    Returns:
        np.ndarray: Pitch-shifted audio.
    """
    return librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)


def add_noise(audio: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
    """Add random noise to audio.

    Args:
        audio: Audio signal.
        noise_level: Level of noise to add.

    Returns:
        np.ndarray: Audio with added noise.
    """
    noise = np.random.randn(len(audio)) * noise_level
    return audio + noise


def apply_augmentation(audio: np.ndarray, sr: int, augment: bool = False) -> np.ndarray:
    """Apply random augmentation to audio.

    Args:
        audio: Audio signal.
        sr: Sample rate.
        augment: Whether to apply augmentation.

    Returns:
        np.ndarray: Augmented audio.
    """
    if not augment:
        return audio

    # Randomly select augmentation type
    aug_type = np.random.choice(["time", "pitch", "noise", "none"])

    if aug_type == "time":
        # Random time stretching between 0.8 and 1.2
        rate = np.random.uniform(0.8, 1.2)
        return time_stretch(audio, rate)
    elif aug_type == "pitch":
        # Random pitch shift between -2 and 2 semitones
        n_steps = np.random.uniform(-2, 2)
        return pitch_shift(audio, sr, n_steps)
    elif aug_type == "noise":
        # Random noise level between 0.001 and 0.01
        noise_level = np.random.uniform(0.001, 0.01)
        return add_noise(audio, noise_level)
    else:
        return audio


def process_audio_file(
    wav_path: str, output_dir: Optional[str] = None, augment: bool = False
) -> Tuple[str, bool]:
    """Process a single audio file to create a spectrogram.

    Args:
        wav_path: Path to the audio file.
        output_dir: Directory to save the spectrogram. If None, save in the same directory as the audio file.
        augment: Whether to apply audio augmentation.

    Returns:
        Tuple[str, bool]: Path to the saved spectrogram and success flag.
    """
    try:
        # Load audio file with librosa
        x, _ = librosa.load(wav_path, sr=sr, mono=True)
        duration = len(x) / sr

        # Check duration and pad if necessary
        if duration <= max_duration:
            total_length = int(max_duration * sr)
            current_length = len(x)
            pad_amount = total_length - current_length
            # Pad symmetrically
            pad_before = pad_amount // 2
            pad_after = pad_amount - pad_before
            x_padded = np.pad(x, (pad_before, pad_after), mode="constant")
        else:
            # If duration is more than max, skip this file
            print(f"Skipping {wav_path} because its duration > {max_duration} seconds.")
            return "", False

        # Apply augmentation if requested
        if augment:
            x_padded = apply_augmentation(x_padded, sr, augment)

        # Compute spectrogram using STFT
        S = librosa.stft(
            x_padded,
            n_fft=nfft,
            hop_length=hop_length,
            win_length=window_length,
            window="hann",
        )
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # Plot spectrogram
        plt.figure(figsize=(6, 4))
        # y-axis in kHz
        librosa.display.specshow(
            S_db, sr=sr, hop_length=hop_length, x_axis="time", y_axis="hz"
        )
        plt.ylim(0, 8000)  # 0 to 8 kHz
        plt.xlim(0, max_duration)  # 0 to 4.5 seconds
        plt.clim(-160, -20)  # fix color axis as in MATLAB caxis([-160 -20])
        plt.axis("off")  # hide axes
        plt.tight_layout(pad=0)

        # Save figure
        # Append '_spec.png' to original filename
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.basename(wav_path)
            out_filename = os.path.join(output_dir, base_filename + "_spec.png")
        else:
            out_filename = wav_path + "_spec.png"

        plt.savefig(out_filename, bbox_inches="tight", pad_inches=0)
        plt.close()
        return out_filename, True
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return "", False


def find_wav_files(data_folder: str) -> List[str]:
    """Find all WAV files in a directory recursively.

    Args:
        data_folder: Directory to search for WAV files.

    Returns:
        List[str]: List of paths to WAV files.
    """
    wav_files = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.join(root, file))
    return wav_files


def main():
    """Main function to process audio files and create spectrograms."""
    parser = argparse.ArgumentParser(description="Create spectrograms from audio files")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="soundwel",
        help="Directory containing audio files",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory to save spectrograms (default: same as input)",
    )
    parser.add_argument(
        "--augment", action="store_true", help="Apply audio augmentation"
    )
    args = parser.parse_args()

    # Data folder path (using relative path)
    data_folder = args.input_dir
    output_dir = args.output_dir

    # Find all WAV files
    wav_files = find_wav_files(data_folder)
    print(f"Found {len(wav_files)} .wav files.")

    # Process each file with progress bar
    success_count = 0
    for wav_path in tqdm(wav_files, desc="Processing audio files"):
        out_filename, success = process_audio_file(wav_path, output_dir, args.augment)
        if success:
            success_count += 1
            tqdm.write(f"Saved spectrogram: {out_filename}")

    print(f"Successfully processed {success_count} out of {len(wav_files)} files.")


if __name__ == "__main__":
    main()
