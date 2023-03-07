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

import matplotlib.pyplot as plt
import soundfile as sf
from sklearn_extra.cluster import KMedoids
import csv
from tqdm import tqdm

if __name__ == "__main__":
    parser = ballroom_danceset_parser(HOP_LENGTH)

    total_songs = len(parser.path_to_songs)

    with open("general_results_mwmd.csv", "w") as file:
        writer = csv.writer(file)

        writer.writerow(["Album", "Song", "Total beats", "Total predicted beats", "True Positives", "False Positives", "False Negatives", "Drag no", "Rush no", "Rush mean", "Rush std", "Drag mean", "Drag std"])

        for song_to_test in tqdm(range(total_songs)):
            audio, annotations_in_frames, original_annotations = parser[song_to_test]

            H_vanilla = np.abs(lr.stft(audio, n_fft = WINDOW_SIZE, hop_length = HOP_LENGTH))

            H_vanilla = min_max_scaling(H_vanilla)

            mwmd_peaks_27 = find_beats(H_vanilla, mwmd, 33)

            peak_indexes = find_indexes_of_spikes(mwmd_peaks_27)[:,1] + 2
            IOI_tests = find_IOI(peak_indexes)

            # Find three potential more than 40bpm and less than 200
            lower_estimated_tempo_threshold = (60/200) / (HOP_LENGTH/FS)
            upper_estimated_tempo_threshold = (60/30) / (HOP_LENGTH/FS)

            cluster = KMedoids(n_clusters=20).fit(IOI_tests)
            x_medoids = cluster.cluster_centers_
            induced_tempo_kmedoids = np.sort(x_medoids[:,0])[0]

            if induced_tempo_kmedoids < 25 or induced_tempo_kmedoids > 350:
                induced_tempo_kmedoids = 33.0

            cleaned_peaks = clean_extra_peaks(mwmd_peaks_27, induced_tempo_kmedoids)
            filled_peaks = fill_beats(cleaned_peaks, induced_tempo_kmedoids)

            gt_peaks_to_test = original_annotations
            predicted_peaks_to_test = find_indexes_of_spikes(filled_peaks)
            predicted_peaks_to_test = predicted_peaks_to_test[:,1] * HOP_LENGTH/FS

            TP = true_positives(gt_peaks_to_test, predicted_peaks_to_test, 5e-2) 
            FP = false_positives(gt_peaks_to_test, predicted_peaks_to_test, 5e-2)
            FN = false_negatives(gt_peaks_to_test, predicted_peaks_to_test, 5e-2)

            rush_mean, rush_std = rushing_time(gt_peaks_to_test, predicted_peaks_to_test, 5e-2)
            rush_mean = rush_mean.round(7)
            rush_std = rush_std.round(7)
            drag_mean, drag_std = dragging_time(gt_peaks_to_test, predicted_peaks_to_test, 5e-2)
            drag_mean = drag_mean.round(7)
            drag_std = drag_std.round(7)

            number_of_drags = calculate_number_of_drags(gt_peaks_to_test, predicted_peaks_to_test, 5e-2)
            number_of_rushes = calculate_number_of_rushes(gt_peaks_to_test, predicted_peaks_to_test, 5e-2)

            total_gt_beats = len(original_annotations)
            total_predicted_beats = len(predicted_peaks_to_test)

            album_name = parser.path_to_songs[song_to_test].split("data/songs/")[-1].split("/")[0]
            song_name = parser.path_to_songs[song_to_test].split("data/songs/")[-1].split("/")[-1]

            to_write = [album_name, song_name,
                        total_gt_beats,
                        total_predicted_beats,
                        TP, FP, FN,
                        number_of_drags, number_of_rushes,
                        rush_mean, rush_std,
                        drag_mean, drag_std]

            writer.writerow(to_write)