import scipy.io as sio
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import glob
#from tqdm import tqdm

freq = 300
secs = 10
max_length = 61


path = 'balanced_training2017/*.mat' # download the original .mat files from https://physionet.org/content/challenge-2017/1.0.0/

def zero_pad(data, length):
    extended = np.zeros(length)
    signalength = np.min([length, data.shape[0]])
    extended[:signalength] = data[:signalength]
    return extended

def spectrogram(data, fs=300, nperseg=64, noverlap=32):
        f, t, Sxx = signal.spectrogram(data, fs=fs, nperseg=nperseg, noverlap=noverlap)
        Sxx = np.transpose(Sxx, [0, 2, 1])
        Sxx = np.abs(Sxx)
        mask = Sxx > 0
        Sxx[mask] = np.log(Sxx[mask])
        return f, t, Sxx

for file in glob.glob(path):
    tag = '.'.join(os.path.basename(file).split('.')[:-1])
    data = sio.loadmat(file)['val'][0]
    data = zero_pad(data, max_length * freq)
    _, t, Sxx = spectrogram(np.expand_dims(data, axis=0))
    plt.figure(figsize=(15, 5))
    xticks_array = np.arange(0, Sxx[0].shape[0], 100)
    xticks_labels = [round(t[label]) for label in xticks_array]
    plt.xticks(xticks_array, labels=xticks_labels)
    plt.xlabel('Time (s)', fontsize=25)
    plt.ylabel('Frequency (Hz)', fontsize=25)
    plt.imshow(np.transpose(Sxx[0]), aspect='auto', cmap='jet')
    plt.gca().invert_yaxis()
    plt.savefig('balanced_training2017_images/'+tag+'.png') # save to an output dir called "training2017_images"
    plt.close('all')
