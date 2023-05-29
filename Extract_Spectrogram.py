# 2023年月29日15时08分38秒
import os
import glob
import pandas
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt

import scipy.io.wavfile
from scipy.fftpack import fft
from scipy import signal
import sys
import scipy.io



def spectrogram(filepath):
    sr, x = scipy.io.wavfile.read(filepath)
    ## Parameters: 10ms step, 30ms window
    nstep = int(sr * 0.01)
    nwin = int(sr * 0.03)
    nfft = nwin

    window = np.hamming(nwin)

    ## will take windows x[n1:n2].  generate
    ## and loop over n2 such that all frames
    ## fit within the waveform
    nn = range(nwin, len(x), nstep)

    X = np.zeros((len(nn), nfft // 2))

    for i, n in enumerate(nn):
        xseg = x[n - nwin:n]
        z = np.fft.fft(window * xseg, nfft)
        X[i, :] = np.log(np.abs(z[:nfft // 2]))

    return X


def spectrogram_plot(X):
    plt.figure(figsize=(12, 4))
    plt.imshow(X.T, interpolation='nearest',
               origin='lower',
               aspect='auto')

    plt.show()


def log_spectrogram(audio, sample_rate, window_size=20,step_size=10, eps=1e-10):
    nperseg = int(round(window_size * sample_rate / 1e3))
    noverlap = int(round(step_size * sample_rate / 1e3))
    freqs, times, spec = signal.spectrogram(audio,
                                    fs=sample_rate,
                                    window='hann',
                                    nperseg=nperseg,
                                    noverlap=noverlap,
                                    detrend=False)
    return freqs, times, np.log(spec.T.astype(np.float32) + eps)

def scipy_log_spectrogram(filepath):
    samples, sample_rate = librosa.load(filepath)
    freqs, times, spectrogram = log_spectrogram(samples, sample_rate)
    return freqs, times, spectrogram

def scipy_log_spectrogram_plot(freqs, times, spectrogram):
    plt.figure(figsize=(12, 4))
    plt.imshow(spectrogram.T, aspect='auto', origin='lower',
               extent=[times.min(), times.max(), freqs.min(), freqs.max()])
    plt.yticks(freqs[::16])
    plt.xticks(times[::16])
    plt.title('Spectrogram')
    plt.ylabel('Freqs in Hz')
    plt.xlabel('Seconds')
    plt.show()

def mel_power_spectrogram(filepath):
    samples, sample_rate = librosa.load(filepath)
    freqs, times, spectrogram = log_spectrogram(samples, sample_rate)
    # Plotting Mel Power Spectrogram
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S

def mel_power_spectrogram_plot(log_S, sample_rate):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_S, sr=sample_rate, x_axis='time', y_axis='mel')
    plt.title('Mel power spectrogram ')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()


def delta2_mfcc(filepath):
    samples, sample_rate = librosa.load(filepath)
    freqs, times, spectrogram = log_spectrogram(samples, sample_rate)
    # Plotting Mel Power Spectrogram
    S = librosa.feature.melspectrogram(samples, sr=sample_rate, n_mels=128)
    # Convert to log scale (dB). We'll use the peak power (max) as reference.
    log_S = librosa.power_to_db(S, ref=np.max)
    # Plotting MFCC
    mfcc = librosa.feature.mfcc(S=log_S, n_mfcc=13)
    # Let's pad on the first and second deltas while we're at it
    delta2_mfcc = librosa.feature.delta(mfcc, order=2)
    return delta2_mfcc

def delta2_mfcc_plot(delta2_mfcc):
    plt.figure(figsize=(12, 4))
    librosa.display.specshow(delta2_mfcc)
    plt.ylabel('MFCC coeffs')
    plt.xlabel('Time')
    plt.title('MFCC')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # spec = spectrogram('./test.wav')
    # print(spec.shape)
    # print(spec)
    # spectrogram_plot(spec)

    # freqs, times, spec2 = scipy_log_spectrogram('./test.wav')
    # print(spec2.shape)
    # print(spec2)
    # scipy_log_spectrogram_plot(freqs, times, spec2)

    # spec3 = mel_power_spectrogram('./test.wav')
    # print(spec3.shape)
    # print(spec3)
    # mel_power_spectrogram_plot(spec3, 16000)

    spec4 = delta2_mfcc('./test.wav')
    print(spec4.shape)
    print(spec4)
    delta2_mfcc_plot(spec4)

