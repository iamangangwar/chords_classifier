import json
import math
import matplotlib.pyplot as plt
import librosa
from pathlib import Path
import numpy as np
import sounddevice as sd
import streamlit
import wavio
import tensorflow as tf
import os

FILE_PATH = os.path.dirname(os.path.abspath(__file__))

def create_spectrogram(voice_sample):
    """
    Creates and saves a spectrogram plot for a sound sample.
    Parameters:
        voice_sample (str): path to sample of sound
    Return:
        fig
    """

    in_fpath = Path(voice_sample.replace('"', "").replace("'", ""))
    original_wav, sampling_rate = librosa.load(str(in_fpath))

    # Plot the signal read from wav file
    fig = plt.figure()
    plt.subplot(211)
    plt.title(f"Spectrogram of recorded audio")

    plt.plot(original_wav)
    plt.xlabel("Sample")
    plt.ylabel("Amplitude")

    plt.subplot(212)
    plt.specgram(original_wav, Fs=sampling_rate)
    plt.xlabel("Time")
    plt.ylabel("Frequency")
    # plt.savefig(voice_sample.split(".")[0] + "_spectogram.png")
    return fig

def read_audio(file):
    with open(file, "rb") as audio_file:
        audio_bytes = audio_file.read()
    return audio_bytes

def record(duration=5, fs=48000):
    sd.default.samplerate = fs
    sd.default.channels = 1
    myrecording = sd.rec(int(duration * fs))
    sd.wait(duration)
    return myrecording

def save_record(path_myrecording, myrecording, fs):
    wavio.write(path_myrecording, myrecording, fs, sampwidth=2)
    return None


def save_mfcc(recording_path, n_mfcc=13, n_fft=2048, hop_length=512, n_segments=5):

    JSON_PATH = os.path.join(FILE_PATH, "samples/data.json")
    SAMPLE_RATE = 44100
    DURATION = 2  # measured in seconds
    SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

    # dictionary to store data
    data = {
        'mfcc': 0
    }

    n_samples_per_segment = int(SAMPLES_PER_TRACK / n_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(n_samples_per_segment / hop_length)


    # load audio file
    try:
        signal, sr = librosa.load(recording_path, sr=SAMPLE_RATE)
    except:
        streamlit.error("Error! File didn't load!!")
        

    # process segments extracting mfcc and storing data
    for s in range(n_segments):
        start_sample = n_samples_per_segment * s
        finish_sample = start_sample + n_samples_per_segment

        signal_part = signal[start_sample:finish_sample]
        mfcc = librosa.feature.mfcc(y=signal_part,
                                    sr=sr,
                                    n_fft=n_fft,
                                    n_mfcc=n_mfcc,
                                    hop_length=hop_length)
        mfcc = mfcc.T

        # store mfcc for segment if it has the expected length
        if len(mfcc) == expected_num_mfcc_vectors_per_segment:
            data['mfcc'] = mfcc.tolist()

    with open(JSON_PATH, "w") as fp:
        json.dump(data, fp, indent=4)

def predict():
    model = tf.keras.models.load_model(os.path.join(FILE_PATH, "saved_model/dcnn_model"))

    inputs = np.zeros(10)
    error = False
    try:
        with open(os.path.join(FILE_PATH, "samples/data.json"), "r") as fp:
            data = json.load(fp)

            # convert lists into numpy arrays
            inputs = np.array(data["mfcc"])

        inputs = inputs[np.newaxis,..., np.newaxis]
        prediction = model.predict(inputs)
        prediction = prediction*(7/8)
        prediction = np.delete(prediction, 3)

        pred_index = np.argmax(prediction)
        chords = ["E", "F", "G", "D", "A", "C", "B"]
        return error, prediction, pred_index, chords

    except:
        streamlit.error("Error! File not found, please record again!!")
        error = True
        
    return error, -1, -1, -1