import math

import audiofile
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import librosa
import librosa.feature
import PIL.Image
import sys

def dBFS(val):
    return 20.0 * math.log10(val)

def rms(arr):
    return math.sqrt(np.mean(np.square(arr)))

if __name__ == '__main__':
    rangeStart = 0
    rangeEnd = 200000
    if len(sys.argv) >= 3:
        rangeStart = int(sys.argv[1])
        rangeEnd = int(sys.argv[2])
    print("processing sampleids from {} to {}".format(rangeStart, rangeEnd))
    for sampleid in range(rangeStart, rangeEnd):
        filename = "wav/{:06d}.wav".format(sampleid)
        outputFilename = "mel/{:06d}.png".format(sampleid)
        if not os.path.exists(filename):
            continue
        signal, rate = audiofile.read(filename)

        if sampleid % 100 == 0:
            print("{} rms={:.1f} dB, rate={}".format(filename, dBFS(rms(signal)), rate))

        spec = librosa.feature.melspectrogram(
            y=signal, sr=rate, n_fft=512, hop_length=256,
            window=scipy.signal.windows.blackmanharris
        )
        topdb = 70

        specdB = librosa.power_to_db(spec, ref=np.max, top_db=topdb)
        specdB = 1.0 + (specdB/topdb)
        img = PIL.Image.fromarray((np.clip(specdB, 0, 1) * 254).astype(np.uint8))
        img.save(outputFilename)

        # fig, ax = plt.subplots()
        # img = librosa.display.specshow(specdB, x_axis='time', y_axis='mel', sr=rate, fmax=8000, ax=ax)
        # fig.colorbar(img, ax=ax, format='%+2.0f dB')
        # ax.set(title="mel spectrogram: {}".format(specdB.shape))
        #
        # plt.show()
        # if idx > 3:
        #     break