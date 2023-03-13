from data_parser import *
import matplotlib.pyplot as plt
import soundfile as sf
from sklearn_som.som import SOM
from scipy.spatial import Delaunay

def mean_filter(signal, window_width):
    signal_copy = np.concatenate((
                                np.ones(shape = (window_width//2,1)) * signal[0],
                                signal, 
                                np.ones(shape = (window_width - window_width//2,1)) * signal[-1]),
                                axis = 0)

    for i in range(signal.size):
        signal[i] = np.mean(signal_copy[i + window_width//2: i + window_width])

    return signal

def median_filter(signal, window_width):
    signal_copy = np.concatenate((
                            np.ones(shape = (window_width//2,1)) * signal[0],
                            signal, 
                            np.ones(shape = (window_width - window_width//2,1)) * signal[-1]),
                            axis = 0)
    
    for i in range(signal.size):
        signal[i] = np.median(signal_copy[i + window_width//2: i + window_width])

    return signal