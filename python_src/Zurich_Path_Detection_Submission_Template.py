# kgolan@student.ethz.ch, 20-923-082
# fabzhang@student.ethz.ch, 17-939-711
# ricardde@student.ethz.ch, 18-916-098


import sys
from Lilygo.Recording import Recording
from Lilygo.Dataset import Dataset
import json
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
from scipy import signal
from scipy.fft import fftshift
import numpy as np
from math import sqrt
from scipy.fft import fft
import matplotlib.pyplot as plt
import pickle
from scipy.signal import medfilt


sys.path.append("..")


# filename = sys.argv[1]  # e.g. 'data/someDataTrace.json'
# IMPORTANT: To allow grading, the two arguments no_labels and mute must be set True in the constructor when loading the data
# trace = Recording(filename, no_labels=False, mute=False)
boardLocation = 0  # <- here goes your detected board location
pathIdx = 0  # <- here goes your detected path index
stepCount = 0  # <- here goes your detected stepcount
activities = (
    []
)  # <- here goes a list of your detected activities, order does not matter


#
# Your algorithm goes here
# Make sure, you do not use the gps data and are tolerant for missing data (see task set). Your program must not crash when single smartphone data traces are missing.
#

### Board Location Detection Functions ###


def get_mag(trace, x, y, z):
    assert x in trace.data and y in trace.data and z in trace.data
    assert len(trace.data[x].values) == len(trace.data[y].values) and len(
        trace.data[x].values
    ) == len(trace.data[z].values)
    magn = [
        sqrt(a**2 + trace.data[y].values[i] ** 2 + trace.data[z].values[i] ** 2)
        for i, a in enumerate(trace.data[x].values)
    ]
    return magn


def rolling_window_stride(array, window_size, freq):
    shape = (array.shape[0] - window_size + 1, window_size)
    strides = (array.strides[0],) + array.strides
    rolled = np.lib.stride_tricks.as_strided(array, shape=shape, strides=strides)
    return rolled[np.arange(0, shape[0], freq)]


def powerSpectrum(x, magn, thresh, filt_order, cutoff, band_type, title, xlim, ylim):
    Fs = int(x.samplerate)
    t = np.arange(0, x.total_time, 1 / x.samplerate)
    # print(t)
    gated_magn = [0 if i < thresh else i for i in magn]

    butterworth = signal.butter(
        N=filt_order, Wn=cutoff, btype=band_type, output="sos", fs=Fs
    )
    filt_magn = signal.sosfilt(butterworth, magn)

    # powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(filt_magn, Fs=Fs, NFFT = 256, noverlap = 128, pad_to = 256)
    powerSpectrum, _, _, _ = plt.specgram(
        filt_magn, Fs=Fs, NFFT=32, noverlap=8, pad_to=32
    )

    return powerSpectrum


def board_loc_preprocessing(trace):
    ax = trace.data["ax"]
    amagn = get_mag(trace, "ax", "ay", "az")
    trace.data["amagn"] = Dataset.fromLists(
        "Accelerometer magnitude", amagn, trace.data["ax"].timestamps
    )

    amagnar = np.asarray(amagn)
    windows = rolling_window_stride(amagnar, 200, 100)

    perc75 = 0
    iqr = 0
    rms = 0

    for i in range(len(windows)):
        perc75 = perc75 + np.percentile(windows[i], 75)
        iqr = iqr + np.percentile(windows[i], 75) - np.percentile(windows[i], 25)
        rms = np.sqrt(np.mean(windows[i] ** 2))

    perc75 = perc75 / np.shape(windows)[0]
    iqr = iqr / np.shape(windows)[0]
    rms = rms / np.shape(windows)[0]
    with plt.ioff():
        PS = powerSpectrum(ax, amagn, 1.1, 20, 10, "lowpass", "Acc Mag", "std", 1)

        pss = PS.mean(1)
        pss.flatten()
        features = np.concatenate((np.array([perc75, iqr, rms]), pss))
        features.flatten()

    return features.reshape(1, 20)


def boardLocation_detector(trace):
    preprocessed_trace = board_loc_preprocessing(trace)
    model = XGBClassifier()
    model.load_model(fname="group15_xgb_board_loc.json")
    return model.predict(preprocessed_trace)[0]


### Step Count Algorithm Functions ###


def step_count_algo(trace):
    acc_x = trace.data["ax"].values
    acc_y = trace.data["ay"].values
    acc_z = trace.data["az"].values

    acc_mag = np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))

    def ma_threshold_sc(acc_mag):
        butterworth_filter = signal.butter(
            N=6, Wn=2, btype="lowpass", output="sos", fs=200
        )
        filtered_mag = signal.sosfilt(butterworth_filter, acc_mag)
        acc_mag_ma = np.convolve(filtered_mag, np.ones(15) / 15, mode="full")
        # plt.plot(filtered_mag, label = ["Filtered Accelerometer Magnitude"])
        peaks = signal.find_peaks(acc_mag_ma)
        peak_magnitudes = [acc_mag_ma[i] for i in peaks[0]]
        upperl, lowerl = np.percentile(peak_magnitudes, [95, 5])
        peak_filter = [
            peak for peak in peak_magnitudes if peak >= lowerl and peak <= upperl
        ]
        return len(peak_filter)

    final_step_count = ma_threshold_sc(acc_mag)
    return final_step_count


### Activity Detection Functions ###


def compute_basic_stats(window_array: np.ndarray):
    avg = np.average(window_array)
    std = np.std(window_array)
    median = np.median(window_array)
    fft_coefs = fft(window_array, norm="forward")
    avg_energy = np.average(
        np.asarray([np.sum([i**2 for i in row]) / len(row) for row in fft_coefs])
    )
    min_max_diff = np.average(
        np.asarray([np.max(row) - np.min(row) for row in window_array])
    )
    iqr = np.average(
        np.asarray(
            [np.percentile(row, 75) - np.percentile(row, 25) for row in window_array]
        )
    )
    rms = np.average(
        np.asarray([np.sqrt(np.mean([x**2 for x in row])) for row in window_array])
    )

    return np.asarray([avg, std, median, avg_energy, min_max_diff, iqr, rms])


def rolling_window(arr: np.ndarray, window_size: int, overlap: float):
    arr = np.asarray(arr)
    overlap_step = int(overlap * window_size)
    window_step = window_size - overlap_step
    new_shape = arr.shape[:-1] + (
        (arr.shape[-1] - overlap_step) // window_step,
        window_size,
    )
    new_strides = arr.strides[:-1] + (window_step * arr.strides[-1],) + arr.strides[-1:]
    return np.lib.stride_tricks.as_strided(arr, shape=new_shape, strides=new_strides)


def feature_extraction_activity_detection(trace):
    ax = np.asarray(trace.data["ax"].values)
    ay = np.asarray(trace.data["ay"].values)
    az = np.asarray(trace.data["az"].values)
    amag = np.sqrt(np.square(ax) + np.square(ay) + np.square(az))

    ax_windows = rolling_window(ax, 512, 0.25)
    ay_windows = rolling_window(ay, 512, 0.25)
    az_windows = rolling_window(az, 512, 0.25)
    amag_windows = rolling_window(amag, 512, 0.25)

    acc_data = [ax_windows, ay_windows, az_windows, amag_windows]

    feat_vector = []
    for data in acc_data:
        feat_vector.append(compute_basic_stats(data))
    return np.asarray(feat_vector, dtype=float).flatten().reshape(1, 28)


def activity_detection(trace):
    preprocessed_trace = feature_extraction_activity_detection(trace)
    trained_model = pickle.load(open("group15_activity_detector.pkl", "rb"))
    fitted_encoder = pickle.load(open("group15_activity_encoder.pkl", "rb"))
    activity_detection = trained_model.predict(preprocessed_trace)
    activity_detection = fitted_encoder.inverse_transform(activity_detection)[0]
    return activity_detection


### Path Detection ###


def slopes(trace):
    a = trace.data["altitude"].values
    a = medfilt(a, 999)
    l = len(a)
    tot = a[l - 1] - a[0]
    a1 = a[int((l - 1) / 5)] - a[0]
    a2 = a[int(2 * (l - 1) / 5)] - a[int((l - 1) / 5)]
    a3 = a[int(3 * (l - 1) / 5)] - a[int(2 * (l - 1) / 5)]
    a4 = a[int(4 * (l - 1) / 5)] - a[int(3 * (l - 1) / 5)]
    a5 = a[int(5 * (l - 1) / 5)] - a[int(4 * (l - 1) / 5)]
    slopes = np.array([tot, a1, a2, a3, a4, a5])
    return slopes


def path_detection(trace):
    preprocessed_trace = slopes(trace).reshape(1, 6)
    path_detector = pickle.load(open("group15_path_detector.pkl", "rb"))
    predicted_path = path_detector.predict(preprocessed_trace)[0]
    return int(predicted_path)


def main():
    filename = sys.argv[1]  # e.g. 'data/someDataTrace.json'
    ## Board Location Detector
    trace = Recording(filename, no_labels=True, mute=True)
    boardLocation = boardLocation_detector(trace)
    stepCount = step_count_algo(trace)
    activities = activity_detection(trace)
    pathIdx = path_detection(trace)
    print(boardLocation)
    print(pathIdx)
    print(stepCount)
    print(activities)

    # IMPORTANT: To allow grading, the two arguments no_labels and mute must be set True in the constructor when loading the data


if __name__ == "__main__":
    main()
