#!/usr/bin/env python3

import os
import csv
import h5py

import librosa

import wave
import torch
import torchaudio

import numpy as np

def min_max_audio_lenght(audio_data):
    """
        Computes the max and min lenght (in number of samples) of the samples in an audio dataset

        Arguments:
        ----------
        audio_data: dict
            Dictionary containing the samples of a dataset where the keys are the IDs of the samples and
            the values are dictionaries with the following keys: 'ImagePath', 'RawDataPath'

        Returns:
        --------
        max_lenght: float
        min_lenght: float
    """
    min_lenght, max_lenght = float("infinity"), 0
    for sample_id in range(len(audio_data)):
        waveform, sample_rate = torchaudio.load(audio_data[sample_id]['RawDataPath'])
        lenght = waveform.shape[1]
        if (lenght > max_lenght):
            max_lenght = lenght
        if (lenght < min_lenght):
            min_lenght = lenght

    return min_lenght, max_lenght

# Zero pad audio data of different lenght in order to have a fixed lenght for every sample
def zero_pad_audio(waveform, sample_rate, audio_lenght, single_channel=False):
    """
        Add zero pad at the beginning and the end of an audio sample in order to obtain a new sample
        of duration audio_duration.
        IMPORTANT HYPOTHESIS: audio_duration >= current audio sample duration

        Arguments:
        ----------
        waveform: np.array
            Audio sample to segment
        audio_lenght: int
             Number of samples wanted in the final audio sample
        single_channel: bool
            True if the audio sample has only one channel

        Returns:
        --------
        new_audio: tensor, array
            New zero padded audio of lenght audio_lenght
    """
    nb_samples_waveform = None
    if (len(waveform.shape) == 1): # One channel audio
        assert (audio_lenght >= waveform.shape[0])
        nb_samples_waveform = waveform.shape[0]
        single_channel = True
    else:
        assert (audio_lenght >= waveform.shape[1])
        nb_samples_waveform = waveform.shape[1]

    # Number of values to add to the current audio sample to have an audio sample of duration audio_duration
    nb_values_to_add = audio_lenght - nb_samples_waveform

    # Zero padding
    if ((nb_values_to_add % 2) == 1):
        padding = (int(nb_values_to_add//2), int(nb_values_to_add//2) + 1)
    else:
        padding = (int(nb_values_to_add//2), int(nb_values_to_add//2))
    if (single_channel):
        waveform = np.pad(waveform, padding, mode='constant', constant_values=0)
    else:
        channel_1 = np.pad(waveform[0], padding, mode='constant', constant_values=0)
        channel_2 = np.pad(waveform[1], padding, mode='constant', constant_values=0)
        waveform = np.array([channel_1, channel_2])

    return waveform


def get_information_wav(wav_path):
    """
        Returns some useful information about a wav file

        Arguments:
        ----------
        wav_path: str
            Path to the wav file that we want to analyze

        Returns:
        --------
        channels: int
            Number of channels in the input audio
        SampleRate: float
        bit_type: int
            Number of bits used to encoded the amplitude values of the audio
        frames: int
            Number of samples in the audio
        duration: float
    """
    # Loading the file
    f = wave.open(wav_path)

    #Audio head parameters
    params = f.getparams()
    channels = f.getnchannels()
    sampleRate = f.getframerate()
    bit_type = f.getsampwidth() * 8
    frames = f.getnframes()
    duration = frames / float (sampleRate)
    f.close()

    return channels, sampleRate, bit_type, frames, duration

def segment_signal(signal, L, hop_length=None):
        """
            This way to treat the input signal to create input embeddings is a naive way that I found myself. It
            is the same way used in the paper "Tiny Transformers for Environmental Sound Classification at the Edge"
            for the raw audio

            Arguments:
            ----------
            signal: np.array
                Signal to segment
            L: int
                length of each segment.
            hop_length: int
        """
        # Number of samples in the signal
        if (len(signal.shape) > 1):
            signal_nb_samples = signal.shape[1]
        else:
            signal_nb_samples = signal.shape[0]

        # Padding with zeros if the number of samples is not divisible by L
        if (signal_nb_samples % L != 0):
            nb_zeros_add = L - (signal_nb_samples % L)
            if (len(signal.shape) > 1):
                padding = np.zeros((signal.shape[0], nb_zeros_add))
                signal = np.concatenate((signal, padding), axis=1)
            else:
                padding = np.zeros((nb_zeros_add))
                signal = np.concatenate((signal, padding))

        # Splitting the signal
        if (hop_length is None):
            hop_length = L
        if (len(signal.shape) > 1):
            segmented_signal = [None for _ in range(signal.shape[0])]
            for channel_nb in range(signal.shape[0]):
                segmented_signal_channel = librosa.util.frame(signal[channel_nb], frame_length=L, hop_length=hop_length, axis=0)
                segmented_signal[channel_nb] = segmented_signal_channel
            segmented_signal = np.array(segmented_signal)
        else:
            segmented_signal = librosa.util.frame(signal, frame_length=L, hop_length=hop_length, axis=0)
        segment_signal = np.array(segmented_signal)

        return segment_signal
