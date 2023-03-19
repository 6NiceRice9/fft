from typing import Any

import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

from numpy import ndarray


class NanonisLongTermDataToFFT:
    # Configure the logging level
    logging.basicConfig(level=logging.INFO)  # Change this to logging.INFO or DEBUG to hide debug messages

    def __init__(self, fname: str):
        self._end_of_header_start_data_index = None
        self.fname = fname
        self.separator = "\t"
        self.decimal = ","
        self.end_of_header_start_data = "[DATA]"
        self._raw_file = None
        self._get_header = None
        self.get_array = None
        self.time = None
        self.np_flatten_signal = None

    def get_file(self):
        self._raw_file = pd.read_csv(self.fname, sep=self.separator, decimal=self.decimal)
        return self._raw_file

    def end_of_header_start_data_index(self):
        with open(self.fname, 'r') as f:
            for i, line in enumerate(f):
                if self.end_of_header_start_data in line:
                    self._end_of_header_start_data_index = i
                    break
            return self._end_of_header_start_data_index
    def get_header(self):
        self.get_header = self._raw_file[:self.end_of_header_start_data_index]
        logging.debug(f"\n pandas_dataframe_head: \n"
                      f"{self.get_header}")
        return self._get_header

    def get_array(self):
        df_raw_body = self._raw_file[self.end_of_header_start_data_index + 2:]
        self.get_array = df_raw_body.to_numpy().astype(float)

        logging.debug(f"\n raw_body: \n"
                      f"{df_raw_body}")
        return self.get_array

    def linear_regression(self: np.ndarray[float]) -> tuple[ndarray, ndarray]:
        # noinspection PyTupleAssignmentBalance
        slope, intercept = np.polyfit(self.df_np_body[:, 0], self.df_np_body[:, 1], 1)  # x, y , 1 = linear fit
        logging.debug(f"\n"
                      f"slope: {slope}\n"
                      f"intercept: {intercept}")
        return slope, intercept

    def fft_signal(self):
        # Sampling Data
        time = self.get_array[:, 0]
        signal = self.get_array[:, 1]

        # linear function y = m*x + b
        linear_function = linear_regression[0] * self.time + linear_regression[1]

        # subtracts slope from the signal -> preparing for FFT
        self.np_flatten_signal = signal - linear_function

        # Calculate the FFT
        fft_signal = np.fft.fft(self.np_flatten_signal)

        # Calculate the frequency values
        num_samples = len(self.time)
        sampling_interval = time[1] - time[0]
        self.freq = np.fft.fftfreq(num_samples, sampling_interval)

        # Calculate the amplitude
        self.power = np.abs(fft_signal)

        return freq, power

    def plot_graph(self):
        # Check if running in debug mode
        if __debug__:
            plt.plot(self.time, self.np_flatten_signal)
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (nm)')
            plt.title('???')
            # plt.xlim(0, 50)
            # plt.yticks(ticks=plt.yticks()[0], labels=plt.yticks()[0] * 10E9)  # multiply y-Axis by 10E9 -> now in nm
            plt.grid()
            plt.show()
        else:
            print("Not in debug mode")

        # Plot the frequency vs amplitude
        plt.plot(self.fft_signal[0], self.fft_signal[1])
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude (nm)')
        plt.title('FFT of a Stable Tip ')
        # plt.xlim(0, 50)
        plt.yticks(ticks=plt.yticks()[0], labels=plt.yticks()[0] * 10E9)  # multiply y-Axis by 10E9 -> now in nm
        plt.grid()
        plt.show()


if __name__ == "__main__":
    file = NanonisLongTermDataToFFT("6Hz_Z_signal_clean001.dat")
    aa = file.get_file()
   var = file._end_of_header_start_data_index()
    # file.get_header()
