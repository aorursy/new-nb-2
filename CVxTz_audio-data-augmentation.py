# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.
#Import stuff



import numpy as np

import random

import itertools

import librosa

import IPython.display as ipd

import matplotlib.pyplot as plt



def load_audio_file(file_path):

    input_length = 16000

    data = librosa.core.load(file_path)[0] #, sr=16000

    if len(data)>input_length:

        data = data[:input_length]

    else:

        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

def plot_time_series(data):

    fig = plt.figure(figsize=(14, 8))

    plt.title('Raw wave ')

    plt.ylabel('Amplitude')

    plt.plot(np.linspace(0, 1, len(data)), data)

    plt.show()
data = load_audio_file("../input/train/audio/off/1df483c0_nohash_0.wav")

plot_time_series(data)
#Hear it ! 

ipd.Audio(data, rate=16000)
# Adding white noise 

wn = np.random.randn(len(data))

data_wn = data + 0.005*wn

plot_time_series(data_wn)

# We limited the amplitude of the noise so we can still hear the word even with the noise, 

#which is the objective

ipd.Audio(data_wn, rate=16000)
# Shifting the sound

data_roll = np.roll(data, 1600)

plot_time_series(data_roll)

ipd.Audio(data_roll, rate=16000)
# stretching the sound

def stretch(data, rate=1):

    input_length = 16000

    data = librosa.effects.time_stretch(data, rate)

    if len(data)>input_length:

        data = data[:input_length]

    else:

        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")



    return data





data_stretch =stretch(data, 0.8)

print("This makes the sound deeper but we can still hear 'off' ")

plot_time_series(data_stretch)

ipd.Audio(data_stretch, rate=16000)



data_stretch =stretch(data, 1.2)

print("Higher frequencies  ")

plot_time_series(data_stretch)

ipd.Audio(data_stretch, rate=16000)
# You can now plug all those transformations in your keras data generator and see your LB rank go up :D