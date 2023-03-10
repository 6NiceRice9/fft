import numpy as np
import pandas as pd
import io


def load_file():
    data = pd.read_csv("long_term_signal.dat")


    return data

def slope():
    data = load_file()
    return data

# var = np.fft()

dt= slope()
#if __name__ == "__main__":
#    dt = slope()
#    print("done")