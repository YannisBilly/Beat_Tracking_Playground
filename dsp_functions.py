import soundfile as sf
from global_variables import *


def find_maxmin(signal, window_width = 3, type = "min"):
    signal_copy = np.concatenate((
                                np.ones(shape = (window_width//2,1)) * signal[0],
                                signal, 
                                np.ones(shape = (window_width - window_width//2,1)) * signal[-1]),
                                axis = 0)
    
    peaks = np.zeros_like(signal)

    if type == "min":
        for i in range(signal.size):
            found_invalid_point = False

            for j in range(i - window_width//2, i + window_width//2):
                if not(signal_copy[i + window_width//2] < signal_copy[j + window_width//2]) and ((i + window_width//2) != (j + window_width//2)):
                    peaks[i] = 0
                    found_invalid_point = True
                    break
            
            if not(found_invalid_point):
                peaks[i] = 1
    else:
        for i in range(signal.size):
            found_invalid_point = False

            for j in range(i - window_width//2, i + window_width//2):
                if not(signal_copy[i + window_width//2] > signal_copy[j + window_width//2])  and ((i + window_width//2) != (j + window_width//2)):
                    peaks[i] = 0
                    found_invalid_point = True
                    break
            
            if not(found_invalid_point):
                peaks[i] = 1

    return peaks

def return_audio_with_beats_as_clicks(audio, clicks, click_duration, amplitude, song_name = "test.wav"):
    sine_beep = amplitude * np.sin(2 * np.pi * 440 / FS * np.arange(int(click_duration * FS)))

    copy_of_audio = np.copy(audio)

    size_of_beep_in_samples = sine_beep.size

    for click_index in clicks:
        # find sample that this will be centered
        click_center_in_samples = int(click_index * HOP_LENGTH)

        start_sample = 0 if click_center_in_samples - size_of_beep_in_samples//2 < 0 else click_center_in_samples - size_of_beep_in_samples//2
        end_sample = copy_of_audio.size if click_center_in_samples + size_of_beep_in_samples//2 >= copy_of_audio.size else click_center_in_samples + size_of_beep_in_samples//2

        copy_of_audio[start_sample : end_sample] += sine_beep[0:len(copy_of_audio[start_sample : end_sample])]

    sf.write(song_name, copy_of_audio, samplerate=44100)

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def min_max_scaling(H, new_min = 0, new_max = 1):
    return (H - H.min()) / (H.max() - H.min() + 1e-8) * (new_max - new_min) + new_min

def plain_difference(H1, H2):
    return H1 - H2

def first_order_differences(H, loss = plain_difference):
    output = np.zeros(shape = (H.shape[-1],1))

    for i in range(1, H.shape[-1]):
        output[i] = loss(H[:, i], H[:,i-1])

    return output[1:]

def find_first_maximum(signal):
    for i in range(1, signal.size - 1):
        if (signal[i-1] < signal[i] and signal[i] > signal[i+1]):
            return i
        
    return 0

def find_indexes_of_spikes(concatenated_onsets):
    number_of_features_used = concatenated_onsets.shape[-1]

    indexes_of_spikes = np.zeros(shape = (1, 2))

    for i in range(number_of_features_used):
        indexes_for_current_feature_peaks = np.argwhere(concatenated_onsets[:,i] == 1)

        for index in indexes_for_current_feature_peaks:
            temp_indexes = np.zeros(shape = (1,2))
            temp_indexes[0,0] = i
            temp_indexes[0,1] = index

            indexes_of_spikes = np.concatenate((indexes_of_spikes, temp_indexes), axis = 0)

    return indexes_of_spikes[1:, :]

def find_beats(spectrogram, similarity_function, window_for_maximum_find = 27):
    onset_detection_function = first_order_differences(spectrogram, similarity_function)

    # autocor_odf = autocorr(np.squeeze(min_max_scaling(onset_detection_function)))
    # period_estimate_in_samples = find_first_maximum(autocor_odf_mse[1:])+1
    # print(f"Estimated Tempo: {60/(period_estimate_in_samples * (HOP_LENGTH/44100))}")

    peaks_estimated = find_maxmin(onset_detection_function, window_for_maximum_find, "max")

    return peaks_estimated

def calculate_triangle_area(position_of_vertices_matrix):
    vector_1 = position_of_vertices_matrix[1,:] - position_of_vertices_matrix[0,:]
    vector_2 = position_of_vertices_matrix[2,:] - position_of_vertices_matrix[0,:]

    pass

def find_IOI(indexes_of_beats, latency_factors = 4):
    """
    latency: the step for forward differences to calculate
    """

    output = np.zeros(shape = (1,1))

    for latency in range(1,latency_factors+1):
        for i in range(indexes_of_beats.shape[0] - latency):
            output = np.concatenate((output, 
                                     np.array([[indexes_of_beats[i + latency] - indexes_of_beats[i]]])), 
                                     axis = 0)

    return output[1:,:]