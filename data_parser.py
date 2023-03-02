import os
import torch
from torch.utils.data import Dataset
import torchaudio as taudio
import numpy as np
from tqdm import tqdm

class ballroom_danceset_parser(Dataset):
    def __init__(self, chunk_size = 0.1):
        """
        Input:
            chunk_size (float): the length of a chunk for MelSpec calculation in samples
                It will be 0.1 seconds because w
        """
        self.chunk_size = int(chunk_size*44100)
        self.path_to_songs = None
        self.annotations_of_chunks = None
        self.songs_and_timestamps = None

        print("Generating relative paths to songs...")
        self.generate_path_to_songs()

        print("Generating starting timestamps for each chunk...")
        self.generate_timestamps_for_audio()

        print("Generating the respective annotations for each chunk...")
        self.generate_class_labels()

        # make a list of (load start time, load duration)
    def __len__(self):
        return len(self.annotations_of_chunks)

    def __getitem__(self, index):
        audio, _ = taudio.backend.sox_io_backend.load(self.songs_and_timestamps[index][0], frame_offset = self.songs_and_timestamps[index][1], num_frames = self.chunk_size)
        annotation = torch.Tensor([self.annotations_of_chunks[index]]) 
        return audio, annotation

    def generate_path_to_songs(self):
        with open("data/songs/allBallroomFiles") as file:
            lines = [line.rstrip() for line in file]

        self.path_to_songs = ["data/songs/" + x[2:] for x in lines]

    def generate_timestamps_for_audio(self):
        self.songs_and_timestamps = []

        for song in self.path_to_songs:
            metadata = taudio.info(song)
            song_length = metadata.num_frames

            for start_index in range(0, song_length - self.chunk_size, self.chunk_size):
                self.songs_and_timestamps.append([song, start_index])

    def generate_class_labels(self):
        self.annotations_of_chunks = []

        annotation_list = os.listdir("data/annotations/")
        annotation_list.remove("README.md")
        annotation_list = ["data/annotations/" + x for x in annotation_list]

        for song in self.path_to_songs:
            # generate the name of song to read annotations
            song_name = song.split("/")[-1]
            song_name = song_name.split(".wav")[0]

            for annotation_name in annotation_list:
                if song_name in annotation_name:
                    annotation_temp = annotation_name
                    break

            metadata = taudio.info(song)
            song_length = metadata.num_frames

            with open(annotation_temp) as file:
                annotations_in_seconds = [float(line.split(" ")[0]) for line in file]
                
            annotations_in_samples = [int(44100*x) for x in annotations_in_seconds]

            for start_index in range(0, song_length - self.chunk_size, self.chunk_size):
                found_a_beat = False
                for anno_in_samples in annotations_in_samples:
                    if start_index <= anno_in_samples and anno_in_samples < start_index + self.chunk_size:          
                        self.annotations_of_chunks.append(1)
                        found_a_beat = True
                        break
                if not(found_a_beat):
                    self.annotations_of_chunks.append(0)
