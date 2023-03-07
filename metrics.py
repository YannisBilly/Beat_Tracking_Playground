import numpy as np

def true_positives(annotations, estimated_beats, window_of_precision = 1e-2):
    TP = 0 

    for i in range(estimated_beats.shape[0]):
        if len(np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)) >= 1:
            TP += 1

    return TP

def false_positives(annotations, estimated_beats, window_of_precision):
    FP = 0 

    for i in range(estimated_beats.shape[0]):
        if len(np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)) < 1:
            FP += 1

    return FP

def false_negatives(annotations, estimated_beats, window_of_precision):
    FN = 0 

    for i in range(annotations.shape[0]):
        if len(np.argwhere(np.abs(annotations[i] - estimated_beats) < window_of_precision)) < 1:
            FN += 1

    return FN

def f1_score(annotations, estimated_beats, false_negative_weigth):
    pass

def rushing_time(annotations, estimated_beats, window_of_precision):

    rushing_times = []

    # if true positive, calculate statistics
    for i in range(estimated_beats.shape[0]):
        if len(np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)) >= 1:
            index_of_tp = np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)[0][0]

            if estimated_beats[i] < annotations[index_of_tp]:
                rushing_times.append(annotations[index_of_tp] - estimated_beats[i])

    if len(rushing_times) < 1:
        rushing_times.append(0)

    return np.mean(np.array(rushing_times)), np.std(np.array(rushing_times))

def dragging_time(annotations, estimated_beats, window_of_precision):

    dragging_times = []

    # if true positive, calculate statistics
    for i in range(estimated_beats.shape[0]):
        if len(np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)) >= 1:
            index_of_tp = np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)[0][0]

            if estimated_beats[i] > annotations[index_of_tp]:
                dragging_times.append(estimated_beats[i] - annotations[index_of_tp])

    if len(dragging_times) < 1:
        dragging_times.append(0)

    return np.mean(np.array(dragging_times)), np.std(np.array(dragging_times))

def calculate_number_of_drags(annotations, estimated_beats, window_of_precision):
    number_of_drags = 0

    # if true positive, calculate statistics
    for i in range(estimated_beats.shape[0]):
        if len(np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)) >= 1:
            index_of_tp = np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)[0][0]

            if estimated_beats[i] > annotations[index_of_tp]:
                number_of_drags += 1

    return number_of_drags

def calculate_number_of_rushes(annotations, estimated_beats, window_of_precision):
    number_of_rushes = 0

    for i in range(estimated_beats.shape[0]):
        if len(np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)) >= 1:
            index_of_tp = np.argwhere(np.abs(estimated_beats[i] - annotations) < window_of_precision)[0][0]

            if estimated_beats[i] < annotations[index_of_tp]:
                number_of_rushes += 1
    return number_of_rushes