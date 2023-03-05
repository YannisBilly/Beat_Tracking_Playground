import os
import numpy as np
import librosa as lr
import matplotlib.pyplot as plt

class ballroom_danceset_parser():
    def __init__(self, hop_size = 4096):
        self.hop_size = hop_size
        self.path_to_songs = None
        self.song_and_annotations = None

        print("Generating relative paths to songs...")
        self.generate_path_to_songs()

        print("Generating the respective annotations for each chunk...")
        self.generate_class_labels()

    def generate_path_to_songs(self):
        with open("data/songs/allBallroomFiles") as file:
            lines = [line.rstrip() for line in file]

        self.path_to_songs = ["data/songs/" + x[2:] for x in lines]
        self.path_to_songs = self.path_to_songs[0:10]
    def __len__(self):
        return len(self.song_and_annotations)

    def __getitem__(self, index):
        return self.song_and_annotations[index][0], np.array(self.song_and_annotations[index][1])

    def generate_class_labels(self):

        self.song_and_annotations = []

        annotation_list = os.listdir("data/annotations/")
        annotation_list.remove("README.md")
        annotation_list = ["data/annotations/" + x for x in annotation_list]

        for song in self.path_to_songs:
            annotations_of_chunks = []
            # generate the name of song to read annotations
            song_name = song.split("/")[-1]
            song_name = song_name.split(".wav")[0]

            for annotation_name in annotation_list:
                if song_name in annotation_name:
                    annotation_temp = annotation_name
                    break

            audio, _ = lr.load(song, sr = 44100)
            song_length = audio.shape[0]

            with open(annotation_temp) as file:
                annotations_in_seconds = [float(line.split(" ")[0]) for line in file]
                
            annotations_in_samples = [int(44100*x) for x in annotations_in_seconds]

            for start_index in range(0, song_length - self.hop_size, self.hop_size):
                found_a_beat = False
                for anno_in_samples in annotations_in_samples:
                    if start_index <= anno_in_samples and anno_in_samples < start_index + self.hop_size:          
                        annotations_of_chunks.append(1)
                        found_a_beat = True
                        break
                if not(found_a_beat):
                    annotations_of_chunks.append(0)

            self.song_and_annotations.append([audio, annotations_of_chunks])

    def plot_onsets_with_annotations(self, onset_detection_function, annotations, name = "test"):
        indexes_where_annotation = np.argwhere(annotations == 1)

        plt.figure(figsize=(15,10))
        for index in indexes_where_annotation:
            plt.axvline(index)

        plt.plot(onset_detection_function)

        plt.savefig(f"plots/{name}.png")
        plt.show()
        plt.close()