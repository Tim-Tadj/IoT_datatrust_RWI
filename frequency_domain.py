# this script finds the cosine transform coefficients of a time window
# the result is stored in a list of dictionary
# iterate through the list to itereate through time windows and index the dictionary 
# to get the coefficients of a particular metric and mote id in that time window 
# e.g. listOFwindows[0]["temperature"][1] returns the coefficient array for the first time window, "temperature" metric, of mote 1
# the result is saved as a pkl file so that it can be loaded later

from numpy import ndarray
import pandas as pd
from tqdm import tqdm
import os
import pickle as pkl
import concurrent.futures
from scipy.fftpack import dct
import time


def compress_transform(cos_transform, max_len, n=5):
    shaped = cos_transform.reshape((max_len//n, len(cos_transform)//(max_len//n)))
    return shaped.mean(axis=0)


def get_cosine_transform(time_window:pd.DataFrame, max_len = 200) -> ndarray:
    result = []
    

    for moteIDnum in range(1, 55):
        sample_arr = (time_window[time_window.moteid == moteIDnum])
        sample_arr = sample_arr[METRIC].to_numpy()
        cos_transform = dct(sample_arr, type=2, n=max_len)

        result.append(cos_transform)
    # down-scale into averaged bands
    for i in range(54):
        result[i] = compress_transform(result[i], max_len, n=10)
    return result



def save(mylist, name):
    if input("Save as pkl? (Y/n)\n~").lower() == "y":
        print("SAVING...")
        start_t = time.time()
        with open(name + ".pkl", 'wb') as f:
            pkl.dump(mylist, f)
        print("DONE! in", "{:.1f}".format(time.time()-start_t) + "s")
    else:
        print("exiting")


def main(time_windows, metric):
    global METRIC
    METRIC = metric
    listOFwindows = []
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for out in tqdm(executor.map(get_cosine_transform, time_windows), total=len(time_windows)):
    #         listOFwindows.append(out)
    for window in tqdm(time_windows):
        out = get_cosine_transform(window)
        listOFwindows.append(out)
    return listOFwindows


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    sample_spaces = []
    with open("timewindows.pkl", "rb") as f:
        sample_spaces = pkl.load(f)
    save(main(sample_spaces), "frequency_domain", "temperature")