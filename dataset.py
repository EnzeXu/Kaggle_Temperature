from torch.utils.data import Dataset, DataLoader
from const import *
from utils import fill_nan
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import pickle
import random
import time
import pandas as pd

class MyDataset(Dataset):
    def __init__(self, mode, index, data):
        super(MyDataset, self).__init__()
        assert mode in ["train", "valid", "test"], "Error in dataset mode: {}".format(mode)
        assert len(index) == len(data), "Error in shape mismatch: index({}) vs. data({})".format(len(index), len(data))
        self.mode = mode
        self.data = data
        self.index = index
        print("dataset: mode={}, index.length={}, data.shape={}".format(mode, len(self.index), self.data.shape))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index, :-1], self.data[index, -1:]



def generate_new_dataframe_expanded(df):
    df['startdate'] = pd.to_datetime(df['startdate'], format='%m/%d/%y')
    offsets = pd.to_timedelta(np.arange(14), unit='d')

    col_names = ['index', 'lat', 'lon', 'startdate'] + ['var{}+{}d'.format(a, b) for a in range(1, 243) for b in range(14)]
    data_dict = {col: [] for col in col_names}

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        lat = row['lat']
        lon = row['lon']
        startdate = row['startdate']

        for a in range(1, 243):
            var_col_name = 'var{}'.format(a)
            values = []

            for b, offset in enumerate(offsets):
                date = startdate + offset
                date_mask = df['startdate'] <= date
                loc_mask = (df['lat'] == lat) & (df['lon'] == lon)

                filtered_df = df.loc[date_mask & loc_mask, :]

                if not filtered_df.empty:
                    latest_row = filtered_df.tail(1)
                    value = latest_row[var_col_name].iloc[0]
                else:
                    value = values[-1] if len(values) > 0 else None

                values.append(value)

            col_name = 'var{}+{}d'.format(a, 0)
            data_dict[col_name].append(row[var_col_name])
            for b in range(1, 14):
                col_name = 'var{}+{}d'.format(a, b)
                data_dict[col_name].append(values[b])

        data_dict['index'].append(row['index'])
        data_dict['lat'].append(row['lat'])
        data_dict['lon'].append(row['lon'])
        data_dict['startdate'].append(row['startdate'])

    new_df = pd.DataFrame.from_dict(data_dict)

    return new_df

def one_time_generate_dataset():
    t0 = time.time()
    df_train_valid = pd.read_csv("data/train_raw.csv")
    df_test = pd.read_csv("data/test_raw.csv")
    print(len(df_train_valid))
    print(len(df_test))
    train_valid_index = list(df_train_valid[COLUMN_INDEX])
    test_index = list(df_test[COLUMN_INDEX])
    train_valid_y = np.asarray(list(df_train_valid[COLUMN_Y_NAME])).reshape(-1, 1)
    test_y = np.asarray(list(df_test[COLUMN_Y_NAME])).reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    df = pd.concat([df_train_valid, df_test])
    print(len(df))
    column_x_name_list_new = COLUMN_X_NAME_LIST  # ['lat', 'lon'] + ['var{}+{}d'.format(a, b) for a in range(1, 243) for b in range(14)]
    df_copy = df[column_x_name_list_new].copy()
    df_copy[COLUMN_CLIMATE_REGIONS] = [CLIMATE_REGIONS_DIC.get(item) for item in list(df_copy[COLUMN_CLIMATE_REGIONS])]
    for one_col in column_x_name_list_new:
        df_copy[one_col] = fill_nan(df_copy[one_col])
    x_numpy = scaler.fit_transform(df_copy)
    train_valid_x = x_numpy[:len(df_train_valid), :]
    test_x = x_numpy[len(df_train_valid):, :]
    valid_random_list = sorted(random.sample(range(len(df_train_valid)), len(df_test)))  # valid has the same size as test
    train_random_list = sorted([item for item in range(len(df_train_valid)) if item not in valid_random_list])
    print("train size: {}".format(len(train_random_list)))
    print("valid size: {}".format(len(valid_random_list)))
    print("test size: {}".format(len(df_test)))
    train_index = [item for i, item in enumerate(train_valid_index) if i in train_random_list]
    valid_index = [item for i, item in enumerate(train_valid_index) if i in valid_random_list]
    train_y = train_valid_y[train_random_list, :]
    valid_y = train_valid_y[valid_random_list, :]
    train_x = train_valid_x[train_random_list, :]
    valid_x = train_valid_x[valid_random_list, :]
    train_data = np.concatenate([train_x, train_y], -1)
    valid_data = np.concatenate([valid_x, valid_y], -1)
    test_data = np.concatenate([test_x, test_y], -1)
    train_dataset = MyDataset(mode="train", index=train_index, data=train_data)
    valid_dataset = MyDataset(mode="valid", index=valid_index, data=valid_data)
    test_dataset = MyDataset(mode="test", index=test_index, data=test_data)
    with open("processed/train.pkl", "wb") as f:
        pickle.dump(train_dataset, f)
    with open("processed/valid.pkl", "wb") as f:
        pickle.dump(valid_dataset, f)
    with open("processed/test.pkl", "wb") as f:
        pickle.dump(test_dataset, f)
    print("cost {} min".format((time.time() - t0) / 60.0))



if __name__ == "__main__":
    # print("step 1")
    # df_sample = pd.read_csv("data/train_raw.csv")[["index", "lat", "lon", "startdate"] + COLUMN_X_NAME_LIST]
    # df_sample.columns = ["index", "lat", "lon", "startdate"] + ["var{}".format(i) for i in range(1, 243)]
    # print(df_sample.columns)
    # df_output = generate_new_dataframe_expanded(df_sample)
    # print(df_sample.columns)
    # df_output.to_csv("data/train_raw_expand.csv", index=False)
    # print("step 2")
    # df_sample = pd.read_csv("data/test_raw.csv")[["index", "lat", "lon", "startdate"] + COLUMN_X_NAME_LIST]
    # df_sample.columns = ["index", "lat", "lon", "startdate"] + ["var{}".format(i) for i in range(1, 243)]
    # print(df_sample.columns)
    # df_output = generate_new_dataframe_expanded(df_sample)
    # print(df_sample.columns)
    # df_output.to_csv("data/test_raw_expand.csv", index=False)
    # print("step 3")
    # one_time_generate_dataset()
    # print(random.sample(range(5), 6))
    one_time_generate_dataset()
    # dataset = TemperatureDataset(mode="test")
    # with open("processed/test.pkl", "wb") as f:
    #     pickle.dump(dataset, f)
    # train_set, valid_set = torch.utils.data.random_split(dataset, [320000, 55734])
    # with open("processed/train.pkl", "wb") as f:
    #     pickle.dump(train_set, f)
    # with open("processed/valid.pkl", "wb") as f:
    #     pickle.dump(valid_set, f)
    # with open("processed/train.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))
    # with open("processed/valid.pkl", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))