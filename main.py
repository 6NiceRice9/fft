import numpy as np
import pandas as pd
import logging
import re

class Nanonis_long_term_data_to_fft:
    # Configure the logging level
    logging.basicConfig(level=logging.DEBUG)  # Change this to logging.INFO to hide debug messages

    # constant values

    def __init__(self, fname, sep="\t"):
        self.fname = fname
        # self.data_clean = data_clean
        self.start_read_from_here = "[DATA]"
        self.start_read_form_index = None

    def load_file(self):
        with open(self.fname, 'r') as f:
            for i, line in enumerate(f):
                if self.start_read_from_here in line:
                    self.start_read_form_index = i
                    break

        df_raw_header = pd.read_csv(self.fname, sep="\t", decimal=",", nrows=self.start_read_form_index + 1)
        df_raw_body = pd.read_csv(self.fname, sep="\t", decimal=",", skiprows=range(1, self.start_read_form_index + 2))

        logging.debug(f"Temporary argument for debugging:"
                      f"\n raw_header:"
                      f"\n{df_raw_header}"
                      f"\n raw_body:"
                      f"\n{df_raw_body}")
        return df_raw_header, df_raw_body


if __name__ == "__main__":
    tfile = Nanonis_long_term_data_to_fft("6Hz_Z_signal_clean001.dat").load_file()

