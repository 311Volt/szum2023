import librosa
import librosa.feature
import scipy.signal
import PIL.Image
import tensorflow as tf
import audiofile
import numpy as np


def create_spectrogram(signal):
    rate = 16000
    topdb = 70

    spec = librosa.feature.melspectrogram(
        y=signal, sr=rate, n_fft=512, hop_length=256,
        window=scipy.signal.windows.blackmanharris
    )

    specdB = librosa.power_to_db(spec, ref=np.max, top_db=topdb)
    specdB = 1.0 + (specdB / topdb)
    return np.clip(specdB, 0, 1)


def create_spectrogram_from_audio_file(filename):
    signal, rate = audiofile.read(filename)
    signal16k = librosa.resample(signal, orig_sr=rate, target_sr=16000)
    return create_spectrogram(signal16k)


def read_spectrogram(filename):
    img = PIL.Image.open(filename).convert("L")
    return np.asarray(img, dtype=np.uint8).astype(np.float_) / 255.0


def show_spectrogram(spectrogram, title):
    PIL.Image.fromarray((spectrogram * 255.0).astype(np.uint8)).show(title)


def entry():
    show_spectrogram(create_spectrogram_from_audio_file("002945.wav"), "fromaudio")
    show_spectrogram(read_spectrogram("000022.png"), "frompng")


if __name__ == "__main__":
    entry()
