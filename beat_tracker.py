from data_parser import *
from distance_functions import *
from global_variables import *
from dsp_functions import *
from filters import *
from metrics import *

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

from sklearn_extra.cluster import KMedoids

def beatTracker(inputFile):
    audio, _ = lr.load(inputFile, 44100)

    H_vanilla = np.abs(lr.stft(audio, n_fft = WINDOW_SIZE, hop_length = HOP_LENGTH))

    H_vanilla = min_max_scaling(H_vanilla)

    mwmd_peaks_27 = find_beats(H_vanilla, spectral_flux, 33)

    peak_indexes = find_indexes_of_spikes(mwmd_peaks_27)[:,1] + 2
    IOI_tests = find_IOI(peak_indexes)

    cluster = KMedoids(n_clusters=20).fit(IOI_tests)
    x_medoids = cluster.cluster_centers_
    induced_tempo_kmedoids = np.sort(x_medoids[:,0])[0]

    if induced_tempo_kmedoids < 33 or induced_tempo_kmedoids > 190:
        induced_tempo_kmedoids = 33.0

    cleaned_peaks = clean_extra_peaks(mwmd_peaks_27, induced_tempo_kmedoids)
    filled_peaks = fill_beats(cleaned_peaks, induced_tempo_kmedoids)

    predicted_peaks_to_test = find_indexes_of_spikes(filled_peaks)
    predicted_peaks_to_test = predicted_peaks_to_test[:,1] * HOP_LENGTH/FS

    return predicted_peaks_to_test, _