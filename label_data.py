import pandas as pd
import os
# set working directory to file location
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# read in data.csv file that was downloaded
df=pd.read_csv("data.csv")

# label original data as trustworthy 1 or untrustworthy 0
labelled_original_data = df[["datetime", "moteid", "temperature"]]
labelled_original_data = labelled_original_data.assign(trustscore=1)

# set trustscore of outliers to 0
# outlers are defined as values that are 3 standard deviations away from the median
upper_limit = labelled_original_data["temperature"].median() + 2 * labelled_original_data["temperature"].std()
lower_limit = labelled_original_data["temperature"].median() - 2 * labelled_original_data["temperature"].std()
boolean_series = (labelled_original_data["temperature"] > upper_limit) | (labelled_original_data["temperature"] < lower_limit)
labelled_original_data.loc[boolean_series, "trustscore"] = 0
labelled_original_data.head()

#proportion of untrustworthy data in original data
num_untrustworthy = labelled_original_data[labelled_original_data.trustscore == 0].count()[0]
num_total = labelled_original_data.count()[0]
print("proportion of untrustworthy in original data is", num_untrustworthy/num_total)

#all synthetic randwalk data is untrustworthy
labelled_randwalk_data = df[["datetime", "moteid", "randwalk"]]
labelled_randwalk_data = labelled_randwalk_data.assign(trustscore=0)

#all synthetic drift data is untrustworthy
labelled_drift_data = df[["datetime", "moteid", "drift"]]
labelled_drift_data = labelled_drift_data.assign(trustscore=0)

# save all labelled data to csv
labelled_original_data.to_csv("labelled_original_data.csv", index=False)
labelled_randwalk_data.to_csv("labelled_randwalk_data.csv", index=False)
labelled_drift_data.to_csv("labelled_drift_data.csv", index=False)