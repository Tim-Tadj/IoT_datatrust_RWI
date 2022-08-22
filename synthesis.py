import numpy as np 
import pandas as pd
import os
from tqdm import tqdm
import random
from multiprocessing import Pool
from multiprocessing import cpu_count


def is_unique_column(series: pd.Series) -> bool: # use nparray to efficiently check this
    array = series.to_numpy()
    return(array[0] == array).all()

def split_days(mote: pd.DataFrame) -> dict: #make a list of dataframes given a mote, outputs list of dataframes
    #check if this is valid to do
    if not is_unique_column(mote.moteid):
        print("ERR: mote is not unique")
        return
    #sort by date first

    setofdates = list(set(mote.date))
    setofdates.sort()
    date_a_sets = []
    for strdate in setofdates:
        date_a_sets.append(mote[ mote.date == strdate])
    return date_a_sets

#struct for storing required values for rwi
class partition:
    def __init__(self):
        self.res = 0 #size of partition
        self.position = 0  #index of start of partition in df
        self.position_end = 0 #index of end of partition in df
        self.ystart = 0 #value of start of partition
        self.yend = 0 #value of end of partition

#random walk infitting algorithm (pretty fast)
class rwi(partition):
    def __init__(self, stepdev:float, scale:float, noise:float, num_points=12):
        self.scale = scale
        self.noise = noise
        self.stepdev = stepdev
        self.num_points = num_points
        self.window_points = [partition() for i in range(num_points-1)]


    def choose_points(self, data:pd.Series, point_selection='even') -> list:
        #find the start and end index in the dataframe for this day
        start, end = data.index.values[0], data.index.values[-1]
        length = end - start
        
        if length < 1: #if the series is empty return an empy list
            return []
        
        #include start and end point in the approximation
        rand_points = [start, length+start]

        #choose 10 evenly spaced points between start and end
        if point_selection == 'even':
            for i in range(1, self.num_points-1):
                rand_points.append(int(start + (length/self.num_points)*i))
            rand_points.sort()

        elif point_selection == 'random':
            # choose 10 more random points to use to approximate the signal
            numpoints = self.num_points - 2
            choice_range = range(start+1, end)
            # choose random points with replacement (no dupicates)
            unique_rand_points = random.sample(choice_range, numpoints)
            rand_points.extend(unique_rand_points)
            rand_points.sort()
        else:
            print("ERR: point selection not valid")
            return []

        for i, point in enumerate(rand_points):
            # bounds checking
            if rand_points[-1] == point:
                break
            elif rand_points[i+1] == point:
                continue

            # struct to keep track of parameters of a sample between approximation points
            part = partition()
            part.res = rand_points[i+1] - point
            part.position = point
            part.position_end = rand_points[i+1]
            part.ystart = data.values[part.position-start] 
            part.yend = data.values[part.position_end-start]

            self.window_points[i] = part
        return self.window_points


    # Define random walk infit algorithm
    def rwi(self, start: float, end: float, res: int, scale: float, noise: float, stddev: float) -> np.ndarray:
        
        # determine step size by the size of the segment we are replacing and the stddev of the data
        step_size = stddev/np.sqrt(res)
        # do guassian walk
        rand_arr = np.random.normal(loc=0, scale=step_size, size=res)
        random_walk = np.cumsum(rand_arr)
        # scale everything by a factor
        scaled_walk = random_walk* scale
        # add gaussian noise according to the stddev of random walk
        noise_arr = np.random.normal(0, scaled_walk.std() * noise, res)
        noisy_random_walk = scaled_walk + noise_arr
        # rotate to connect start and end points of noisy random walk
        y = noisy_random_walk
        x = range(len(y))
        # find angle to move to that joins start and finish
        start_theta = np.arctan( (y[-1] - y[0]) / res)
        fin_theta = np.arctan((end-start)/res)
        theta = -(fin_theta-start_theta)
        # make vector to rotate
        d = np.hstack((np.vstack(x), np.vstack(y)))
        # rotate vector
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        xr = np.dot(d, R)
        # return rotated vector
        return xr[:, 1] + start

    def rwi_between_points(self):
        # call rwi function to generate random walk between points
        output = []
        for part in self.window_points:
            output.append(self.rwi(part.ystart, part.yend, part.res, self.scale, self.noise, self.stepdev))
        return output


# creates "synthetic" column
def rwi_all_days(dataset:pd.DataFrame, column:str, stepdev: float, scale: float, noise: float, num_points=12, point_selection='even') -> pd.DataFrame:
    column_added = []
    if "datetime" in dataset.columns:
        dataset["datetime"] = pd.to_datetime(dataset["datetime"])

    if 'datetime' not in dataset.columns:
        dataset['datetime'] = pd.to_datetime(dataset['date'] + ' ' + dataset['time'])
        column_added.append('datetime')
    elif 'date' not in dataset.columns:
        dataset['date'] = dataset['datetime'].dt.date
        column_added.append('date')

    datas = [dataset[dataset.moteid == i] for i in range(1, 55)]
    synthetic_dataset = dataset.copy()
    synthetic_dataset["synthetic"] = dataset[column]
    # for each mote and day in the dataset infit the data using RWI(random walk infitting)
    for motedata in tqdm(datas):
        days = split_days(mote=motedata)
        for day in days:
            # skip if the day has less points than num of windows
            if len(day[column]) <= num_points:
                continue
            # use a rwi object to run the rwi algorithm for a day for a mote
            rwi_obj = rwi(stepdev, scale, noise, num_points)
            rwi_obj.choose_points(day[column], point_selection=point_selection)
            synthetic_temperatures = rwi_obj.rwi_between_points()
            # add the synthetic temperatures to the dataframe
            for i, part in enumerate(rwi_obj.window_points):
                synthetic_dataset.loc[part.position:part.position_end-1, "synthetic"] = synthetic_temperatures[i]
    
    #remove added columns
    for col in column_added:
        synthetic_dataset.drop(col, axis=1, inplace=True)
    return synthetic_dataset

def rwi_parallel(inputdata:zip) -> pd.DataFrame:
    inputdata = list(inputdata)
    motedata_pool:pd.DataFrame = inputdata[0]
    motes_present:list = inputdata[1]
    column:str = inputdata[2]
    stepdev:float = inputdata[3]
    scale:float = inputdata[4]
    noise: float = inputdata[5]
    num_points = inputdata[6]
    point_selection = inputdata[7]
    # for each mote and day in the dataset infit the data using RWI(random walk infitting)
    for motenum in motes_present:
        days = split_days(mote=motedata_pool[motedata_pool.moteid==motenum])
        for day in days:
            # skip if the day has less points than num of windows
            if len(day[column]) <= num_points:
                continue
            # use a rwi object to run the rwi algorithm for a day for a mote
            rwi_obj = rwi(stepdev, scale, noise, num_points)
            rwi_obj.choose_points(day[column], point_selection=point_selection)
            synthetic_temperatures = rwi_obj.rwi_between_points()
            # add the synthetic temperatures to the dataframe
            for i, part in enumerate(rwi_obj.window_points):
                motedata_pool.loc[part.position:part.position_end-1, "synthetic"] = synthetic_temperatures[i]
    return motedata_pool


def rwi_all_days_parallel(dataset:pd.DataFrame, column:str, stepdev: float, scale: float, noise: float, num_points=12, point_selection='even', cpu_amount=4) -> pd.DataFrame:
    column_added = []
    if "datetime" in dataset.columns:
        dataset["datetime"] = pd.to_datetime(dataset["datetime"])

    if 'datetime' not in dataset.columns:
        dataset['datetime'] = pd.to_datetime(dataset['date'] + ' ' + dataset['time'])
        column_added.append('datetime')
    elif 'date' not in dataset.columns:
        dataset['date'] = dataset['datetime'].dt.date
        column_added.append('date')

    dataset["synthetic"] = dataset[column]
    # for each mote and day in the dataset infit the data using RWI(random walk infitting)
    # use multiprocessing to speed up the process
    #split motenums into chunks of size cpu count
    columnsInDataset = list(dataset.columns)
    include_columns = ["moteid", "date", "synthetic", column]
    exclude_columns = [x for x in columnsInDataset if x not in include_columns]
    num_arr = np.arange(1, 55)
    pooled_motenums = np.array_split(num_arr, cpu_amount)

    #make smaller (pooled) datasets for each cpu core and drop unnecessary columns
    pooled_data = [dataset[dataset.moteid.isin(motenums)].drop(columns=exclude_columns) for motenums in pooled_motenums]


    input_data = []
    for i, motedata in enumerate(pooled_data):
        input_data.append((motedata, pooled_motenums[i], column, stepdev, scale, noise, num_points, point_selection))

    with Pool(processes=cpu_count()) as pool:
        for output in tqdm(pool.imap_unordered(rwi_parallel, input_data), total=len(input_data)):
            dataset.loc[output.index, "synthetic"] = output["synthetic"]

    #remove added columns
    for col in column_added:
        dataset.drop(col, axis=1, inplace=True)
    return dataset



if __name__ == "__main__":
    print("\nSTART!")
    currdir = os.path.dirname(__file__)
    os.chdir(currdir)
    dataset = pd.read_csv("data.csv")
    #convert date and time columns to datatime objects
    
    synthetic_data = rwi_all_days_parallel(dataset, 
                            "realworld",
                            stepdev=dataset.realworld.std(), 
                            scale=0.05, 
                            noise=0.1, 
                            num_points=12)
    print(synthetic_data)