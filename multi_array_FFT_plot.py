import main
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt

fn = "2023_03_07_z_very-stable_002.dat"
df = main.NanonisLongTermDataToFFT(fn)

whole_file = df.get_file()

def raw_file(fn):
    raw_file = pd.read_csv(fn, sep="\t", decimal=".",  on_bad_lines='skip', skiprows=37, low_memory=False)
    return raw_file
header_index = df.end_of_header_start_data_index() # get header index
header = df.get_header(header_index)    # get all header
data = raw_file(fn)

#data = df.get_array(header_index)
#linear_regression = main.lin_regression(data)
#frequency, power, time, signal, linear_function, signal_minus_slope = main.fft_signal(data, linear_regression[0], linear_regression[1])
#plot_graph()


