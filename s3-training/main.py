import librosa
import librosa.feature
import scipy.signal
import PIL.Image
import pandas as pd
import os
import audiofile
import numpy as np


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras


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


def binarize_labels(labels):
    int_labels = []
    for label in labels:
        if label == 'male':
            int_labels.append(0)
        elif label == 'female':
            int_labels.append(1)

    return int_labels


def prepare_dataset(split_number, type_name):
    csv_path = "./split{}_{}.csv".format(split_number, type_name)

    df_train = pd.read_csv(csv_path, dtype={"sampleid": "string"})
    image_paths = ['./mel_unnormalized/' + fname + '.png' for fname in df_train['sampleid']]
    class_labels = df_train['gender'].tolist()
    int_labels = binarize_labels(class_labels)

    assert (len(int_labels) == len(class_labels))

    image_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    label_ds = tf.data.Dataset.from_tensor_slices(int_labels)

    return tf.data.Dataset.zip((image_ds, label_ds))


def load_and_preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1, dtype=tf.dtypes.uint8)
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # normalize pixel values
    img = tf.cast(img, tf.float16) / 255.
    img = tf.squeeze(img)
    img = tf.transpose(img)
    paddings = tf.constant([[0, 100, ], [0, 0]])
    # 'constant_values' is 0.
    # rank of 't' is 2.
    img = tf.pad(img, paddings, "CONSTANT")
    img = img[:100, :]
    return img, label


def entry():
    # show_spectrogram(create_spectrogram_from_audio_file("002945.wav"), "fromaudio")
    # show_spectrogram(read_spectrogram("000022.png"), "frompng")

    ds_train = prepare_dataset(1, 'train') \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .prefetch(tf.data.AUTOTUNE)

    ds_val = prepare_dataset(1, 'val') \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(4096) \
        .prefetch(tf.data.AUTOTUNE)

    lstm_layer = keras.layers.LSTM(64, input_shape=(None, 128))
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(10),
            keras.layers.Dense(1),
        ]
    )

    model.summary()

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer="adam",
        metrics=["accuracy"],
    )

    print(ds_train)

    model.fit(ds_train, epochs=60, validation_data=ds_val)

    # for idx, (x, y) in enumerate(ds_train):
    #     # model.fit(x=x, y=y, validation_data=ds_val, batch_size=128)
    #     #if idx % 10 == 0:
    #         #model.fit(x=x, y=y, validation_data=ds_val, batch_size=128)
    #     model.fit(x=x, y=y, batch_size=128)


if __name__ == "__main__":
    entry()
