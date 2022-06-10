import streamlit as st
from functions import *
import librosa
import librosa.display as display
import matplotlib.pyplot as plt
import os

st.title("Guitar Chords Classifier")
st.text("This application classifies open major chords played on a guitar using a DCNN \n(Deep Convolutional Neural Network)")
st.header("Record the audio")
FILE_PATH = os.path.dirname(os.path.abspath(__file__))
rec_path = os.path.join(FILE_PATH, "samples/audio.wav")

if st.button("Click to Record"):
    record_state = st.text("Recording...")
    duration = 2
    fs = 44100
    recording = record(duration, fs)
    save_record(rec_path, recording, fs)
    save_mfcc(recording_path=rec_path, n_segments=10, n_mfcc=40)

    st.text("Spectrogram")
    fig = create_spectrogram(rec_path)
    st.pyplot(fig)

    signal, sr = librosa.load(os.path.join(FILE_PATH, "samples/audio.wav"), sr=44100)
    mfcc = librosa.feature.mfcc(y=signal, sr=sr, n_fft=2048, hop_length=512, n_mfcc=40)
    fig, ax = plt.subplots()
    display.specshow(data=mfcc, sr=44100, hop_length=512)
    plt.xlabel("Time")
    plt.ylabel("MFCCs")
    plt.colorbar()
    st.text("MFCCs")
    st.pyplot(fig)

st.header("Predict the Chord")
if st.button("Click to Predict"):
    error, prediction, pred_index, chords = predict()

    if ~error:
        fig = plt.figure(figsize = (10, 5))
        plt.bar(chords, prediction)
        plt.xlabel("Chord")
        plt.ylabel("Probability")
        st.pyplot(fig)

        st.text("The predicted audio is {}% chord {}".format(prediction[pred_index]*100, chords[pred_index]))
        os.remove(rec_path)
        os.remove(os.path.join(FILE_PATH, "samples/data.json"))
        st.text("Audio file deleted from the server for privacy purpose.")