# this script creates time windows according to the size and increment parameters of
# the make_sample_spaces(size, increment, df) function
# the result is a list of timewindow dataframes of the sorted dataset
# iterate through the list to get each time window
# e.g. listOFwindows[0] returns the dataframe for the first time window
# the result is saved as a pkl file so that it can be loaded later

# note this script creates  these time windows within the training set it makes
import pandas as pd
from tqdm import tqdm
import datetime
import os
import sys
import pickle as pkl
import concurrent.futures
os.chdir(os.path.dirname(__file__))
import time

SORTED_DATA = None
TP_SIZE = None

def save(mylist, name):
    if input("Save as pkl? (Y/n)\n~").lower() == "y":
        print("SAVING...")
        start_t = time.time()
        with open(name + ".pkl", 'wb') as f:
            pkl.dump(mylist, f)
        print("DONE! in", "{:.1f}".format(time.time()-start_t) + "s")
    else:
        print("exiting")


def get_samplespaces(sorted_data, window_start, window_end):
    return sorted_data[(sorted_data.datetime >= (window_start)) & (sorted_data.datetime < (window_end))]


def make_sample_spaces(sorted_data:pd.DataFrame, size:datetime, increment:datetime ) -> list:
    sorted_data["datetime"] = pd.to_datetime(sorted_data["datetime"])
    min_time = sorted_data["datetime"].min()
    max_time = sorted_data["datetime"].max()

    i = min_time
    timepoints  = []
    while i < max_time:
        timepoints.append(i)
        i+=increment

    samplespaces = []

    for window_start in tqdm(timepoints):
        window = get_samplespaces(sorted_data, window_start, window_start+size)
        if len(window)>500:
            samplespaces.append(window)
    return samplespaces


def main(sorted_data, size, increment):

    samplespaces = make_sample_spaces(sorted_data, size, increment)
    print("n# of samples:", len(samplespaces))

    total_bytes = 0
    for i in samplespaces:
        total_bytes += sys.getsizeof(i)
    print(total_bytes/1000000, "MB" )
    return samplespaces

if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    hour = datetime.timedelta(hours=1)
    minute = datetime.timedelta(minutes=1)
    print("loading data from file..")
    dataset_in = pd.read_csv("data.csv")
    result = main(dataset_in, 2*hour, 1*hour)
    save(result, "timewindows") #saves as a list of dataframes
    