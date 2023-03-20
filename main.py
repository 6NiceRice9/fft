from typing import Any, Tuple

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

class NanonisLongTermDataToFFT:
    # Configure the logging level
    logging.basicConfig(level=logging.INFO)  # Change this to logging.INFO or DEBUG to hide debug messages

    def __init__(self, fname: str):
        self.fname = fname
        self.separator = "\t"
        self.decimal = ","
        self.end_of_header_start_data = "[DATA]"
        self._raw_file = None

    def get_file(self):
        self._raw_file = pd.read_csv(self.fname, sep=self.separator, decimal=self.decimal,  on_bad_lines='skip')
        logging.debug(f"\n raw_pandas_file: \n"
                      f"{self._raw_file}")
        return self._raw_file

    def end_of_header_start_data_index(self):
        with open(self.fname, 'r') as f:
            for i, line in enumerate(f):
                if self.end_of_header_start_data in line:
                    end_of_header_start_data_index = i
                    break
            logging.debug(f"\n end_of_header_start_data_index: \n"
                          f"{end_of_header_start_data_index}")
            return end_of_header_start_data_index

    def get_header(self, index):
        get_header = self._raw_file[:index]
        logging.debug(f"\n pandas_dataframe_header: \n"
                      f"{get_header}")
        return get_header

    def get_array(self, index):
        raw_body = self._raw_file[index:]
        get_array = raw_body.to_numpy().astype(float)

        logging.debug(f"\n raw_body: \n"
                      f"{raw_body}")
        return get_array


def lin_regression(array_2D):
    slope, intercept = np.polyfit(array_2D[:, 0], array_2D[:, 1], 1)  # x, y , 1 = linear fit
    logging.debug(f"\n"
                  f"slope: {slope}\n"
                  f"intercept: {intercept}")
    return slope, intercept


def fft_signal(array_2D, slope, intercept):
    # Sampling Data
    time = array_2D[:, 0]
    signal = array_2D[:, 1]

    # linear function y = m*x + b
    linear_function = slope * time + intercept

    # subtracts slope from the signal -> preparing for FFT
    signal_minus_slope = signal - linear_function

    # Calculate the FFT
    fft_signal = np.fft.fft(signal_minus_slope)

    # Calculate the frequency values
    num_samples = len(time)
    sampling_interval = time[1] - time[0]
    freqency = np.fft.fftfreq(num_samples, sampling_interval)

    # Calculate the amplitude
    power = np.abs(fft_signal)

    # calculated frequency and amplitude (power), additionally: linear_function, signal_without_slope
    # raw data: time, signal
    return freqency, power, time, signal, linear_function, signal_minus_slope


def plot_graph():
    # Check if running in debug mode
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)

    if __debug__:
        # Plot on the first subplot
        ax4.plot(frequency, power)
        ax4.set_title('Sine Function')
        ax4.set_xlabel('Frequency (Hz)')
        ax4.set_ylabel('Amplitude (nm)')
        ax4.grid()

        # Plot on the second subplot
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

        # Plot on the second subplot
        ax3.plot(time, signal_minus_slope)
        ax3.set_title('sinal_without_slope')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Amplitude (nm)')
        ax3.grid()

        plt.show()



    else:
        print("Not in debug mode")

    # # Plot the frequency vs amplitude
    # plt.plot(frequency, power)
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude (nm)')
    # plt.title('FFT of a Stable Tip ')
    # # plt.xlim(0, 50)
    # plt.yticks(ticks=plt.yticks()[0], labels=plt.yticks()[0] * 10E9)  # multiply y-Axis by 10E9 -> now in nm
    # plt.grid()
    # plt.show()


if __name__ == "__main__":
    file = NanonisLongTermDataToFFT("6Hz_Z_signal_clean001.dat")

    whole_file = file.get_file()
    header_index = file.end_of_header_start_data_index()
    header = file.get_header(header_index)
    data = file.get_array(header_index)
    linear_regression = lin_regression(data)
    frequency, power, time, signal, linear_function, signal_minus_slope = fft_signal(data, linear_regression[0],
                                                                                     linear_regression[1])
    plot_graph()

    # header = file.get_header()
    # var = print(type(file.end_of_header_start_data_index))
    # file.get_header()

