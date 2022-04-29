# This is the template for the submission. If you want, you can develop your algorithm in a Jupyter Notebook and copy the code here for submission. Don't forget to test according to specification below

# Team members (e-mail, legi):
# kgolan@student.ethz.ch, 
# examplestudent2@ethz.ch, 12-345-679
# examplestudent3@ethz.ch, 12-345-670

import sys
from Lilygo.Recording import Recording
from Lilygo.Dataset import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage.filters import uniform_filter1d


def walk_detection_algo(traces: dict):
    pass
    
    

def step_count_algo(traces: dict):
    traces.DataIntegrityCheck()
    print("\r\nAvailable data traces:")
    print(list(traces.data.keys())) # Prints the various features recorded during walk. 
    ## NOTE: every key here is an instance of the Lilygo Dataset class and can be plotted as a timeseries. 

    if not(traces.labels is None):
        print("\r\nLabels: ")
        print(traces.labels)
    
    ax = traces.data["ax"]
    print(type(ax))
    print(ax.__dict__.keys()) # returns class attributes

    print("Data Summary:\n")
    print(f"Name of sensor: '{ax.title}'")
    print(f"Sample rate: {int(ax.samplerate)} Hz")
    print(f"Recording length: {ax.total_time} seconds")
    print(f"Timestamp of recording: {ax.raw_timestamps[0][1]}")

    # traces.plot([['ax', 'ay', 'az'], ['gx', 'gy', 'gz'], ['mx', 'my', 'mz'], ['altitude'], ['speed'], ['temperature']], 
            #  ylabels=['Accelerometer [g]', 'Gyroscope', 'Magnetometer', 'Elevation (m)', 'speed', 'IMU temperature'],
            #  labels=[['Acc X', 'Acc Y', 'Acc Z'], ['Gyro X', 'Gyro Y', 'Gyro Z'], ['Mag X', 'Mag Y', 'Mag Z'], ['altitude'], ['speed'], ['temperature']])
    acc_x = ax.values
    acc_y = traces.data["ay"].values
    acc_z = traces.data["az"].values

    acc_mag = np.sqrt(np.square(acc_x) + np.square(acc_y) + np.square(acc_z))


    def ma_threshold_sc(acc_mag):
        butterworth_filter = signal.butter(N = 6, Wn = 2, btype = "lowpass", output = "sos", fs = 200)
        filtered_mag = signal.sosfilt(butterworth_filter, acc_mag)
        acc_mag_ma = np.convolve(filtered_mag, np.ones(15)/15, mode='full')
        plt.figure()
        plt.subplot(211)
        plt.plot(acc_mag, label = ["Accelerometer Magnitude"])
        plt.subplot(212)
        # plt.plot(filtered_mag, label = ["Filtered Accelerometer Magnitude"])
        plt.plot(acc_mag_ma, label = ["Accelerometer Magnitude Moving Average"])
        peaks = signal.find_peaks(acc_mag_ma)
        peak_magnitudes = [acc_mag_ma[i] for i in peaks[0]]
        plt.subplot(211)
        plt.show()
        upperl, lowerl = np.percentile(peak_magnitudes, [95 , 5])
        peak_filter = [peak for peak in peak_magnitudes if peak >= lowerl and peak <= upperl]
        print("Detected Step Count:", len(peak_filter))
        return len(peak_filter)
    
    def normalized_autocorrelation_sc(acc_mag):
        def normalization(vector):
            """
        Picked up from https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector 
            """
            return vector / (np.linalg.norm(vector) + 1e-16)

        def rolling_window(a, window):
            """
        Picked up from https://stackoverflow.com/questions/6811183/rolling-window-for-1d-arrays-in-numpy 
            """
            shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
            strides = a.strides + (a.strides[-1],)
            return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)
            

        normalized_acc_mag  = normalization(acc_mag)
        ac_normalized_acc_mag = None

    def stft_sc_method(acc_mag):
        NotImplemented

    def ml_sc(acc_mag):
        NotImplemented

    # plt.plot(acc_mag, label = ["Accelerometer Magnitude"])
    
    # print("Detected Step Count:", len(peaks[0]))

    print("""
    TA Dataset
    released_0: 41 steps (walking, arm still)
    released_1: 101 steps (walking, arm swinging)
    released_2: 44 steps (running)
    released_3: 47 steps (walking downhill)
    released_4: 45 steps (walking, tie shoelaces)
    released_5: 33 steps (walking/climbing stairs)

    Our recordings
    recording_01 = 60 steps
    recording_02 = 55 steps
    recording_03 = 110 steps
    recording_04 = 100 steps
    recording_05 = 40 steps
    recording_06 = 38 steps
    recording_07 = 30 steps
    recording_08 = 53 steps
    recording_09 = 53 steps
    """)


    final_step_count = ma_threshold_sc(acc_mag)
    return final_step_count
    


    


def main():
    filename = sys.argv[1] # e.g. 'data/someDataTrace.json'
    print(filename)
    
    # IMPORTANT: To allow grading, the two arguments no_labels and mute must be set True in the constructor when loading the data
    trace = Recording(filename, no_labels=True, mute=True)
    step_count_algo(trace)
    stepCount = step_count_algo(trace) # <- here goes your detected stepcount
    print(stepCount)


if __name__ == '__main__':
    main()


#
# Your algorithm goes here
# Make sure, you only use data from the LilyGo Wristband, namely the following 10 keys (as in trace.data[key]):
# 3-axis accelerometer: key in [ax, ay, az]
# 3-axis gyro: key in [gx, gy, gz]
# 3-axis magnetometer: key in [mx, my, mz] 
# IMU temperature: key==temperature
#



#Print result as Integer, do not change!


# Test this file before submission with the data we provide to you

# 1. In the console or Anaconda Prompt execute:
# python --version

# Output should look something like (displaying your python version, which must be 3.8.x):
# Python 3.8.10
# If not, check your python installation or command 

# 2. In the console execute:
# python [thisfilename.py] path/to/datafile.json

# Output should be an integer corresponding to the step count you calculated
