import re
import sys

import librosa
import librosa.feature
import scipy.signal
import PIL.Image
import pandas as pd
import os
import audiofile
import numpy as np
import sklearn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import keras_tuner


def create_spectrogram(signal):
    rate = 16000
    topdb = 70

    spec = librosa.feature.melspectrogram(
        y=signal, sr=rate, n_fft=512, hop_length=256,
        window=scipy.signal.windows.blackmanharris
    )

    specdB = librosa.power_to_db(spec, ref=np.max, top_db=topdb)
    specdB = 1.0 + (specdB / topdb)
    spec1 = np.transpose(np.clip(specdB, 0, 1).astype(np.float16))
    return spec1.reshape((1, spec1.shape[0], spec1.shape[1]))


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


def augmentation_amount(sampleid, enable=True):
    if not enable:
        return 0.0
    hv = hash(sampleid)
    db = (hv % 35) - 10  # -24 to +10 dB
    # 1 dB = 1/70
    diff = db / 70.0
    return diff


def prepare_dataset(split_number, type_name, enable_augmentation):
    csv_path = "./split{}_{}.csv".format(split_number, type_name)

    df_train = pd.read_csv(csv_path, dtype={"sampleid": "string"})
    image_paths = [sampleid_filename(split_number, fname) for fname in df_train['sampleid']]
    aug_amounts = [augmentation_amount(int(sampleid), enable_augmentation) for sampleid in df_train['sampleid']]
    class_labels = df_train['gender'].tolist()
    int_labels = binarize_labels(class_labels)

    assert (len(int_labels) == len(class_labels))

    image_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    augamt_ds = tf.data.Dataset.from_tensor_slices(aug_amounts)
    label_ds = tf.data.Dataset.from_tensor_slices(int_labels)

    return tf.data.Dataset.zip((image_ds, augamt_ds, label_ds))


def load_and_preprocess_image(path, augamt, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1, dtype=tf.dtypes.uint8)
    # img = tf.image.convert_image_dtype(img, dtype=tf.float32)  # normalize pixel values
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)

    img = tf.clip_by_value(img - augamt, 0.0, 1.0)
    img = tf.cast(img, tf.float16)

    img = tf.squeeze(img)
    img = tf.transpose(img)

    paddings = tf.constant([[0, 100, ], [0, 0]])
    # 'constant_values' is 0.
    # rank of 't' is 2.
    img = tf.pad(img, paddings, "CONSTANT")
    img = img[:100, :]
    return img, label


def build_tuned_model(hp: keras_tuner.HyperParameters):
    model = keras.Sequential()
    model.add(
        keras.layers.LSTM(
            hp.Int("lstm_units", min_value=160, max_value=320, step=32),
            input_shape=(None, 128)
        )
    )
    dense_activ = hp.Choice("dense_activation", ['relu', 'sigmoid'])
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Dense(
        hp.Int("h1size", min_value=24, max_value=96),
        activation=dense_activ
    ))
    model.add(keras.layers.Dense(
        hp.Int("h2size", min_value=8, max_value=24),
        activation=dense_activ
    ))
    model.add(keras.layers.Dense(1, activation='sigmoid'))

    optimizer = keras.optimizers.Adam(learning_rate=hp.Float("lrate", min_value=0.0001, max_value=0.01, sampling='log'))

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=optimizer,
        metrics=[
            "accuracy"
        ],
    )
    return model


def entry():
    # show_spectrogram(create_spectrogram_from_audio_file("002945.wav"), "fromaudio")
    # show_spectrogram(read_spectrogram("000022.png"), "frompng")

    # ds_test = prepare_dataset(1, 'test', False) \
    #     .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
    #     .batch(128) \
    #     .prefetch(tf.data.AUTOTUNE)
    #
    # imported_model = keras.models.load_model("tuned_model.hdf5")
    # imported_model.summary()
    #
    # imported_model.evaluate(ds_test)
    # return None

    if len(sys.argv) > 1:
        imported_model = keras.models.load_model("gcmodel.hdf5")

        in_filename = sys.argv[1]
        inputsnd = create_spectrogram_from_audio_file(in_filename)
        result = imported_model.predict(inputsnd)[0][0]
        print("result={:.06f}\n{} is {}".format(
            result,
            in_filename,
            "MALE" if result < 0.5 else "FEMALE"
        ))
        return None

    split_number = 2

    ds_train = prepare_dataset(split_number, 'train', True) \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .cache()
    # .prefetch(tf.data.AUTOTUNE)

    ds_test = prepare_dataset(split_number, 'test', False) \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .prefetch(tf.data.AUTOTUNE)

    ds_val = prepare_dataset(split_number, 'val', False) \
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE) \
        .batch(128) \
        .prefetch(tf.data.AUTOTUNE)

    lstm_layer = keras.layers.LSTM(512, input_shape=(None, 128))
    model = keras.models.Sequential(
        [
            lstm_layer,
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid'),
        ]
    )

    model.summary()
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.0013),
        metrics=[
            "accuracy"
        ],
    )

    # tuner = keras_tuner.BayesianOptimization(
    #     hypermodel=build_tuned_model,
    #     objective="val_accuracy",
    #     max_trials=20
    # )
    #
    # tuner.search_space_summary()
    # tuner.search(ds_train, epochs=6, validation_data=ds_val)
    # tuner.results_summary()
    #
    #
    # best_model: keras.models.Sequential = tuner.get_best_models(num_models=2)[0]
    # best_model.summary()
    # best_model.save("tuned_model.hdf5")
    #
    # return None


    savecb = keras.callbacks.ModelCheckpoint(
        "saved-model-epoch{epoch:03d}-{val_accuracy:.2f}.hdf5",
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='max')

    print(ds_train)

    history = model.fit(ds_train, epochs=40, validation_data=ds_val, callbacks=savecb)
    model.save("thicc_bitch.hdf5")
    print(history.history)
    print("loss, accuracy: " + str(model.evaluate(ds_test)))


if __name__ == "__main__":
    entry()
