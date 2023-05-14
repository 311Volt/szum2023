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

def sampleid_filename(split_number, sampleid):
    if split_number == 1:
        return "./mel_unnormalized/{}.png".format(sampleid)
    return "./mel/{}.png".format(sampleid)

def prepare_dataset(split_number, type_name):
    csv_path = "./split{}_{}.csv".format(split_number, type_name)

    df_train = pd.read_csv(csv_path, dtype={"sampleid": "string"})
    image_paths = [sampleid_filename(split_number, fname) for fname in df_train['sampleid']]
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
    img = tf.cast(img, tf.float16) * (1.0 / 255.0)
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

    split_number = 3

    ds_train = prepare_dataset(split_number, 'train') \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .cache()
        # .prefetch(tf.data.AUTOTUNE)

    ds_test = prepare_dataset(split_number, 'test') \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .prefetch(tf.data.AUTOTUNE)

    ds_val = prepare_dataset(split_number, 'val') \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .prefetch(tf.data.AUTOTUNE)

    lstm_layer = keras.layers.LSTM(256, input_shape=(None, 128))
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(24, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ]
    )

    model.summary()
    keras.optimizers.Adam()
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer='adam',
        metrics=["accuracy"],
    )

    savecb = keras.callbacks.ModelCheckpoint(
        "saved-model-epoch{epoch:03d}-{val_accuracy:.2f}.hdf5",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max')

    print(ds_train)

    history = model.fit(ds_train, epochs=40, validation_data=ds_val, callbacks=savecb)
    model.save("")
    print(history.history)
    print("loss, accuracy: " + str(model.evaluate(ds_test)))



if __name__ == "__main__":
    entry()
