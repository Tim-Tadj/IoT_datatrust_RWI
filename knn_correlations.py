# this script finds the cosine transform coefficients of a time window
# the result is stored in a list of dictionary
# iterate through the list to itereate through time windows and index the dictionary
# to get the coefficients of a particular metric and mote id in that time window
# e.g. listOFwindows[0]["temperature"][1] returns the coefficient array for the first time window, "temperature" metric, of mote 1
# the result is saved as a pkl file so that it can be loaded later

import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import pickle as pkl
import concurrent.futures

def get_adjacent_motes_all(nearcorr: pd.DataFrame) -> dict:
    return   {moteIDnum: nearcorr[(nearcorr.moteid == moteIDnum)].moteid_corr.tolist() for moteIDnum in range(1, 59)}


def get_adjacent_mote_correlation(mote_df: pd.DataFrame, nearestmotes_dfs: list) -> list:

    mote_x = mote_df[[METRIC]]

    corrlist = []
    for nearest_mote in nearestmotes_dfs:
        mote_y = nearest_mote[[METRIC]]
        merged = pd.merge_asof(
            mote_x, mote_y, left_index=True, right_index=True, direction="nearest")
        corr_val = merged.corr().iloc[0, 1]
        # default correlation to 0 if result is NaN
        corrlist.append(0 if np.isnan(corr_val) else corr_val)

    if corrlist == []:
        corrlist = [0 for _ in range(7)]

    return corrlist


def get_knn_corr(wdw) -> list:
    samplespace = wdw

    result = []

    for moteIDnum in range(1, 55):

        mote_df = samplespace[samplespace.moteid == moteIDnum].set_index("datetime")
        nearest_mote_IDs = ADJACENT_MOTES[moteIDnum] 
        nearestmotes_dfs = [(samplespace[samplespace.moteid == i].set_index("datetime")) for i in nearest_mote_IDs]
        result.append(get_adjacent_mote_correlation(mote_df, nearestmotes_dfs))
    return result


def save(list_data: list, path):
    with open(path + ".pkl", "wb") as file:
        pkl.dump(list_data, file)


def main(timewindows, neighbour_map, metric):
    global ADJACENT_MOTES
    global METRIC
    METRIC = metric
    ADJACENT_MOTES = get_adjacent_motes_all(neighbour_map)
    resulting_windows = []


    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     for out in tqdm(executor.map(get_knn_corr, timewindows), total=len(timewindows)):
    #         resulting_windows.append(out)
    for window in tqdm(timewindows):
        out = get_knn_corr(window)
        resulting_windows.append(out)
    ADJACENT_MOTES = None
    return resulting_windows
    


if __name__ == "__main__":

    os.chdir(os.path.dirname(__file__))
    neighbour_map = pd.read_csv(
        "neighbour_map.csv")
    sample_spaces = []
    with open("timewindows.pkl", "rb") as f:
        sample_spaces = pkl.load(f)

    main_result = main(sample_spaces, neighbour_map)

    save(main_result, "knn_correlations", "temperature")
