import main
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)  # Change this to logging.INFO or DEBUG to hide debug messages


def raw_file(fn):
    raw_file = pd.read_csv(fn, sep="\t", decimal=".", on_bad_lines='skip', skiprows=37, low_memory=False)
    return raw_file


def plot_graph(frequency, power, time, signal, linear_function, signal_minus_slope):
    # Check if running in debug mode
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)

    if __debug__:
        # Plot on the first subplot
        ax1.plot(time, signal)
        ax1.set_title('raw_signal')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Amplitude (nm)')
        ax1.grid()

        # Plot on the second subplot
        ax2.plot(time, linear_function)
        ax2.set_title('fitted_linear_func')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Amplitude (nm)')
        ax2.grid()

        ax3.plot(time, signal_minus_slope)
        ax3.set_title('sinal_without_slope')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude (nm)')
        ax3.grid()

        ax4.plot(frequency, power)
        ax4.set_title('FFT')
        ax4.set_xlabel('Frequency (Hz)')
        plt.xlim(0, 25)
        plt.xticks(np.arange(0, max(frequency) + 1, 1))
        # plt.yticks(ticks=plt.yticks()[0], labels=plt.yticks()[0] * 10E9)  # multiply y-Axis by 10E9 -> now in nm
        ax4.set_ylabel('Amplitude (nm)')
        ax4.grid()

        plt.show()


# constants
fn = "2023_03_07_z_very-stable_002.dat"
channel_to_observe = "Z (m)"
time_aquisition = 0.02  # change from const value, to reading from header

# coarse file loading
df = main.NanonisLongTermDataToFFT(fn)
whole_file = df.get_file()

header_index = df.end_of_header_start_data_index()  # get header index
header = df.get_header(header_index)  # get all header
data = raw_file(fn)
n_signal_values = len(data[channel_to_observe])
vec_signal_Z = data[channel_to_observe].to_numpy(float)
vec_time = np.arange(0, n_signal_values * time_aquisition, time_aquisition)
array_2D = np.stack((vec_time, vec_signal_Z), axis=1)  # convert 2 vectors to vertical 2D-array

# calculate and subtract slope from signal (preparation for FFT)
linear_regression = main.lin_regression(array_2D)

# FFT
frequency, power, time, signal, linear_function, signal_minus_slope = main.fft_signal(array_2D, linear_regression[0],
                                                                                      linear_regression[1])
plot_graph(frequency, power, time, signal, linear_function, signal_minus_slope)
