import soundfile as sf
from global_variables import *


def find_maxmin(signal, window_width = 3, type = "min"):
    """
    Function to find local minima or maxima, by outputing 1 if the sample at t is the maximum in [t - window_width // 2, t + window_width // 2].
    Inputs:
        signal: the onset detection function
        window_width: the total width of the window
        type: "min" for estimating the minimum, "max" for estimating the maximum
    """
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
    """
    Function to generate a sinusoid for each beat in clicks, by adding a sinusoid to audio.
    Inputs:
        audio: the audio for which the beat tracking system works
        clicks: bin indexes of clicks
        amplitude: the amplitude for the sinusoid that will be the beat
        song_name: the generated songs name
    """
    sine_beep = amplitude * np.sin(2 * np.pi * 440 / FS * np.arange(int(click_duration * FS)))

    copy_of_audio = np.copy(audio)

    size_of_beep_in_samples = sine_beep.size

    for click_index in clicks:
        click_center_in_samples = int(click_index * HOP_LENGTH)

        start_sample = 0 if click_center_in_samples - size_of_beep_in_samples//2 < 0 else click_center_in_samples - size_of_beep_in_samples//2
        end_sample = copy_of_audio.size if click_center_in_samples + size_of_beep_in_samples//2 >= copy_of_audio.size else click_center_in_samples + size_of_beep_in_samples//2

        copy_of_audio[start_sample : end_sample] += sine_beep[0:len(copy_of_audio[start_sample : end_sample])]

    sf.write(song_name, copy_of_audio, samplerate=44100)

def autocorr(x):
    """
    Autocorrelation function for tempo estimation. It isn't used
    """
    result = np.correlate(x, x, mode='full')
    return result[result.size//2:]

def min_max_scaling(H, new_min = 0, new_max = 1):
    """
    Minimum maximum scaling of H to [new_min, n_max]
    """
    return (H - H.min()) / (H.max() - H.min() + 1e-8) * (new_max - new_min) + new_min

def plain_difference(H1, H2):
    return H1 - H2

def first_order_differences(H, loss = plain_difference):
    """
    Onset detection function estimation by using the function loss.
    Inputs:
        H: the spectrogram
        loss: the similarity function to be used
    """
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
    """
    Function to estimate the index of the beats in concatenated_onsets, which is a vector of zeros and ones, where a one denotes the presence of a beat.
    """
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
    """
    A wrapper for the beat-tracking system, without post-processing
    """
    onset_detection_function = first_order_differences(spectrogram, similarity_function)

    peaks_estimated = find_maxmin(onset_detection_function, window_for_maximum_find, "max")

    return peaks_estimated

def find_IOI(indexes_of_beats):
    """
    Function to find Inter Onset Intervals
    """

    output = np.zeros(shape = (1,1))

    for j in range(indexes_of_beats.shape[0]):
        for i in range(j, indexes_of_beats.shape[0]):
            if i != j and i+j < indexes_of_beats.shape[0]:
                output = np.concatenate((output, 
                                        np.array([[indexes_of_beats[i + j] - indexes_of_beats[i]]])), 
                                        axis = 0)


    return output[1:,:]

def clean_extra_peaks(peaks, tempo):
    """
    Post processing cleanup of 'extra' beats
    """
    output = np.copy(peaks)

    for i in range(peaks.shape[0]):
        if (peaks[i] == 1):
            for j in range(i+1, peaks.shape[0]):
                if (peaks[j] == 1):
                    if j - i < tempo*0.8:
                        output[j] = 0
                    break

    return output

def fill_beats(peaks, tempo):
    """
    Post-processing 'filler' of missing beats
    """
    output = np.copy(peaks)

    for i in range(peaks.shape[0]):
        if (peaks[i] == 1):
            for j in range(i+1, peaks.shape[0]):
                if (peaks[j] == 1):
                    if j-i > 1.6*tempo and j-i < 2.4*tempo:
                        middle_index = int((j+i)/2)

                        output[middle_index] = 1
                    break

    return output