from data_parser import *
from metrics import *
from global_variables import *
from dsp_functions import *
from filters import *

import matplotlib.pyplot as plt
import soundfile as sf

if __name__ == "__main__":

    parser = ballroom_danceset_parser(HOP_LENGTH)

    song_to_test = 0

    audio, annotations = parser[song_to_test]

    H_vanilla = np.abs(lr.stft(audio, n_fft = WINDOW_SIZE, hop_length = HOP_LENGTH))

    H_vanilla = min_max_scaling(H_vanilla)

    mae_peaks_estimated = find_beats(H_vanilla, mae)
    mse_peaks_estimated = find_beats(H_vanilla, mse)
    mwmd_peaks_estimated = find_beats(H_vanilla, mwmd)
    spectral_flux_peaks_estimated = find_beats(H_vanilla, spectral_flux)

    peak_indexes = find_indexes_of_spikes(mwmd_peaks_estimated)[:,1] + 2
    IOI_tests = find_IOI(peak_indexes)

    plt.plot(IOI_tests)
    plt.show()
    plt.close()

    return_audio_with_beats_as_clicks(audio, find_indexes_of_spikes(mae_peaks_estimated)[:,1] + 2, click_duration=0.05, amplitude=0.3, song_name="mae_vanilla_27.wav")
    return_audio_with_beats_as_clicks(audio, find_indexes_of_spikes(mse_peaks_estimated)[:,1] + 2, click_duration=0.05, amplitude=0.3, song_name="mse_vanilla_27.wav")
    return_audio_with_beats_as_clicks(audio, find_indexes_of_spikes(mwmd_peaks_estimated)[:,1] + 2, click_duration=0.05, amplitude=0.3, song_name="mwmd_vanilla_27.wav")
    return_audio_with_beats_as_clicks(audio, find_indexes_of_spikes(spectral_flux_peaks_estimated)[:,1] + 2, click_duration=0.05, amplitude=0.3, song_name="spectral_flux_vanilla_27.wav")

    concatenated_onsets = np.concatenate((np.expand_dims(annotations, axis = 1), mae_peaks_estimated, mse_peaks_estimated, spectral_flux_peaks_estimated, mwmd_peaks_estimated), axis = 1)

    plt.pcolormesh(concatenated_onsets[1000:1200,:].T)
    plt.show()
    plt.close()