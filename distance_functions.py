from data_parser import *
from global_variables import *



def logscale_spectrogram(H):
    H[H < MAGNITUDE_LOWER_BOUND] = MAGNITUDE_LOWER_BOUND
    H = 20*np.log10(H / MAGNITUDE_LOWER_BOUND)
    H /= MAXIMUM_LOG_MAGNITUDE

    return H

def mwmd(H1, H2):
    return 1 - np.mean(np.sum(np.power(np.abs(H1 - H2), H1 + 1e-8)))

def mse(H1, H2):
    return np.mean(np.sum(np.power(H1 - H2, 2)))

def mae(H1, H2):
    return np.mean(np.abs(H1 - H2))

def spectral_flux(H1, H2):
    return np.sum(np.max(H1 - H2, 0))

def spectral_flux_mwmd(H1, H2):
    pass

def cosine_dissimilarity(H1, H2):
    return np.dot(H1, H2) / (np.sqrt(np.sum(np.power(H1, 2))) * np.sqrt(np.sum(np.power(H2, 2))))