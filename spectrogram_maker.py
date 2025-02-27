import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

# Settings
sr = 44100  # sample rate
max_duration = 4.5  # seconds

# Parameters for spectrogram
window_duration = 0.03  # 0.03 seconds
window_length = int(window_duration * sr)  # number of samples in the window
noverlap = int(0.99 * window_length)  # overlap
nfft = 512  # Changed from 2048 to 512 to match MATLAB implementation
hop_length = window_length - noverlap  # hop length

# Data folder path (using relative path)
dataFolder = 'soundwel'

# Walk through the directory recursively to find .wav files
wav_files = []
for root, _, files in os.walk(dataFolder):
    for file in files:
        if file.lower().endswith('.wav'):
            wav_files.append(os.path.join(root, file))

print(f'Found {len(wav_files)} .wav files.')

for wav_path in wav_files:
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
            x_padded = np.pad(x, (pad_before, pad_after), mode='constant')
        else:
            # If duration is more than max, skip this file
            print(f'Skipping {wav_path} because its duration > {max_duration} seconds.')
            continue

        # Compute spectrogram using STFT
        S = librosa.stft(x_padded, n_fft=nfft, hop_length=hop_length, win_length=window_length, window='hann')
        S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)

        # Plot spectrogram
        plt.figure(figsize=(6, 4))
        # y-axis in kHz
        librosa.display.specshow(S_db, sr=sr, hop_length=hop_length, x_axis='time', y_axis='hz')
        plt.ylim(0, 8000)  # 0 to 8 kHz
        plt.xlim(0, max_duration)  # 0 to 4.5 seconds
        plt.clim(-160, -20)  # fix color axis as in MATLAB caxis([-160 -20])
        plt.axis('off')  # hide axes
        plt.tight_layout(pad=0)

        # Save figure
        # Append '_spec.png' to original filename
        out_filename = wav_path + '_spec.png'
        plt.savefig(out_filename, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f'Saved spectrogram: {out_filename}')
    except Exception as e:
        print(f'Error processing {wav_path}: {e}')