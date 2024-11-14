import os
import numpy as np
import pandas as pd
import glob
import re
import torch
import random
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

warnings.filterwarnings('ignore')


class Dataset_ETT_hour_Multi_Input(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.window_len = window_len

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        # 取时间戳特征
        df_stamp = df_raw[['date']]  # [border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        # 区分训练、验证、测试数据，便于后续构造数据集
        if self.set_type == 0:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp[border1:border2]
        if self.set_type == 1:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp[border1:border2]
            self.data_x_add = data[0:border2s[1]]
            self.data_stamp_add = data_stamp[0:border2s[1]]
        if self.set_type == 2:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp
            self.data_x_add = data[0:border2s[2]]
            self.data_stamp_add = data_stamp[0:border2s[2]]

    def __getitem__(self, index):
        for i, tmp_window_len in enumerate(self.window_len):
            s_begin = index
            s_end = s_begin + self.seq_len

            cut_num = tmp_window_len // self.seq_len

            idx = np.array([random.choice(
                range((s_end - tmp_window_len) + i * cut_num, (s_end - tmp_window_len) + (i + 1) * cut_num)) for i in
                range(self.seq_len)])

            if self.set_type == 0:
                if (s_end - tmp_window_len) < 0:
                    idx = idx[np.where(np.array(idx) >= 0)]

            if self.set_type == 0:
                seq_x_tmp = self.data_x[idx]
                seq_x_mark_tmp = self.data_stamp[idx]
                if (s_end - tmp_window_len) < 0:
                        tmp_padding = np.array([random.uniform(-1, 1) for _ in
                                                range((self.seq_len - len(idx)) * (seq_x_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_tmp.shape[-1])
                        tmp_mark_padding = np.array([random.uniform(-1, 1) for _ in range(
                            (self.seq_len - len(idx)) * (seq_x_mark_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_mark_tmp.shape[-1])
                        tmp = self.data_x[idx]
                        mark_tmp = self.data_stamp[idx]
                        seq_x_tmp = np.concatenate([tmp_padding, tmp])
                        seq_x_mark_tmp = np.concatenate([tmp_mark_padding, mark_tmp])

            else:
                idx = idx + (len(self.data_x_add) - len(self.data_x))
                seq_x_tmp = self.data_x_add[idx]
                seq_x_mark_tmp = self.data_stamp_add[idx]

            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_y_tmp = self.data_y[r_begin:r_end]
            seq_y_mark_tmp = self.data_stamp[r_begin:r_end]

            # 数据合并
            seq_x_tmp = np.expand_dims(seq_x_tmp, axis=0)
            seq_x_mark_tmp = np.expand_dims(seq_x_mark_tmp, axis=0)
            if i == 0:
                seq_x = seq_x_tmp
                seq_x_mark = seq_x_mark_tmp
                seq_y = seq_y_tmp
                seq_y_mark = seq_y_mark_tmp
            else:
                seq_x = np.concatenate([seq_x, seq_x_tmp], axis=0)
                seq_x_mark = np.concatenate([seq_x_mark, seq_x_mark_tmp], axis=0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute_Multi_Input(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.window_len = window_len

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]#[border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.set_type == 0:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp[border1:border2]
        if self.set_type == 1:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp[border1:border2]
            self.data_x_add = data[0:border2s[1]]
            self.data_stamp_add = data_stamp[0:border2s[1]]
        if self.set_type == 2:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp
            self.data_x_add = data[0:border2s[2]]
            self.data_stamp_add = data_stamp[0:border2s[2]]

    def __getitem__(self, index):
        for i, tmp_window_len in enumerate(self.window_len):
            s_begin = index
            s_end = s_begin + self.seq_len

            cut_num = tmp_window_len // self.seq_len

            idx = np.array([random.choice(
                range((s_end - tmp_window_len) + i * cut_num, (s_end - tmp_window_len) + (i + 1) * cut_num)) for i in
                range(self.seq_len)])

            if self.set_type == 0:
                if (s_end - tmp_window_len) < 0:
                    idx = idx[np.where(np.array(idx) >= 0)]

            if self.set_type == 0:
                seq_x_tmp = self.data_x[idx]
                seq_x_mark_tmp = self.data_stamp[idx]
                if (s_end - tmp_window_len) < 0:
                        tmp_padding = np.array([random.uniform(-1, 1) for _ in
                                                range((self.seq_len - len(idx)) * (seq_x_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_tmp.shape[-1])
                        tmp_mark_padding = np.array([random.uniform(-1, 1) for _ in range(
                            (self.seq_len - len(idx)) * (seq_x_mark_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_mark_tmp.shape[-1])
                        tmp = self.data_x[idx]
                        mark_tmp = self.data_stamp[idx]
                        seq_x_tmp = np.concatenate([tmp_padding, tmp])
                        seq_x_mark_tmp = np.concatenate([tmp_mark_padding, mark_tmp])

            else:
                idx = idx + (len(self.data_x_add) - len(self.data_x))
                seq_x_tmp = self.data_x_add[idx]
                seq_x_mark_tmp = self.data_stamp_add[idx]

            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_y_tmp = self.data_y[r_begin:r_end]
            seq_y_mark_tmp = self.data_stamp[r_begin:r_end]

            # 数据合并
            seq_x_tmp = np.expand_dims(seq_x_tmp, axis=0)
            seq_x_mark_tmp = np.expand_dims(seq_x_mark_tmp, axis=0)
            if i == 0:
                seq_x = seq_x_tmp
                seq_x_mark = seq_x_mark_tmp
                seq_y = seq_y_tmp
                seq_y_mark = seq_y_mark_tmp
            else:
                seq_x = np.concatenate([seq_x, seq_x_tmp], axis=0)
                seq_x_mark = np.concatenate([seq_x_mark, seq_x_mark_tmp], axis=0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom_Multi_Input(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.window_len = window_len

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']]  # [border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        if self.set_type == 0:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp[border1:border2]
        if self.set_type == 1:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp[border1:border2]
            self.data_x_add = data[0:border2s[1]]
            self.data_stamp_add = data_stamp[0:border2s[1]]
        if self.set_type == 2:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_stamp = data_stamp
            self.data_x_add = data[0:border2s[2]]
            self.data_stamp_add = data_stamp[0:border2s[2]]

    def __getitem__(self, index):
        for i, tmp_window_len in enumerate(self.window_len):
            s_begin = index
            s_end = s_begin + self.seq_len

            cut_num = tmp_window_len // self.seq_len

            idx = np.array([random.choice(
                range((s_end - tmp_window_len) + i * cut_num, (s_end - tmp_window_len) + (i + 1) * cut_num)) for i in
                range(self.seq_len)])

            if self.set_type == 0:
                if (s_end - tmp_window_len) < 0:
                    idx = idx[np.where(np.array(idx) >= 0)]

            if self.set_type == 0:
                seq_x_tmp = self.data_x[idx]
                seq_x_mark_tmp = self.data_stamp[idx]
                if (s_end - tmp_window_len) < 0:
                        tmp_padding = np.array([random.uniform(-1, 1) for _ in
                                                range((self.seq_len - len(idx)) * (seq_x_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_tmp.shape[-1])
                        tmp_mark_padding = np.array([random.uniform(-1, 1) for _ in range(
                            (self.seq_len - len(idx)) * (seq_x_mark_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_mark_tmp.shape[-1])
                        tmp = self.data_x[idx]
                        mark_tmp = self.data_stamp[idx]
                        seq_x_tmp = np.concatenate([tmp_padding, tmp])
                        seq_x_mark_tmp = np.concatenate([tmp_mark_padding, mark_tmp])

            else:
                idx = idx + (len(self.data_x_add) - len(self.data_x))
                seq_x_tmp = self.data_x_add[idx]
                seq_x_mark_tmp = self.data_stamp_add[idx]

            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_y_tmp = self.data_y[r_begin:r_end]
            seq_y_mark_tmp = self.data_stamp[r_begin:r_end]

            # 数据合并
            seq_x_tmp = np.expand_dims(seq_x_tmp, axis=0)
            seq_x_mark_tmp = np.expand_dims(seq_x_mark_tmp, axis=0)
            if i == 0:
                seq_x = seq_x_tmp
                seq_x_mark = seq_x_mark_tmp
                seq_y = seq_y_tmp
                seq_y_mark = seq_y_mark_tmp
            else:
                seq_x = np.concatenate([seq_x, seq_x_tmp], axis=0)
                seq_x_mark = np.concatenate([seq_x_mark, seq_x_mark_tmp], axis=0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar_Multi_Input(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.window_len = window_len

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')  # 去除文本中的换行符
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        if self.set_type == 0:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
        if self.set_type == 1:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_x_add = data[0:border2s[1]]
        if self.set_type == 2:
            self.data_x = data[border1:border2]
            self.data_y = data[border1:border2]
            self.data_x_add = data[0:border2s[2]]

    def __getitem__(self, index):
        for i, tmp_window_len in enumerate(self.window_len):
            s_begin = index
            s_end = s_begin + self.seq_len

            cut_num = tmp_window_len // self.seq_len

            idx = np.array([random.choice(
                range((s_end - tmp_window_len) + i * cut_num, (s_end - tmp_window_len) + (i + 1) * cut_num)) for i in
                range(self.seq_len)])

            if self.set_type == 0:
                if (s_end - tmp_window_len) < 0:
                    idx = idx[np.where(np.array(idx) >= 0)]

            if self.set_type == 0:
                seq_x_tmp = self.data_x[idx]
                seq_x_mark_tmp = self.data_stamp[idx]
                if (s_end - tmp_window_len) < 0:
                        tmp_padding = np.array([random.uniform(-1, 1) for _ in
                                                range((self.seq_len - len(idx)) * (seq_x_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_tmp.shape[-1])
                        tmp_mark_padding = np.array([random.uniform(-1, 1) for _ in range(
                            (self.seq_len - len(idx)) * (seq_x_mark_tmp.shape[-1]))]).reshape(
                            self.seq_len - len(idx), seq_x_mark_tmp.shape[-1])
                        tmp = self.data_x[idx]
                        mark_tmp = self.data_stamp[idx]
                        seq_x_tmp = np.concatenate([tmp_padding, tmp])
                        seq_x_mark_tmp = np.concatenate([tmp_mark_padding, mark_tmp])

            else:
                idx = idx + (len(self.data_x_add) - len(self.data_x))
                seq_x_tmp = self.data_x_add[idx]
                seq_x_mark_tmp = self.data_stamp_add[idx]

            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

            seq_y_tmp = self.data_y[r_begin:r_end]
            seq_y_mark_tmp = self.data_stamp[r_begin:r_end]

            # 数据合并
            seq_x_tmp = np.expand_dims(seq_x_tmp, axis=0)
            seq_x_mark_tmp = np.expand_dims(seq_x_mark_tmp, axis=0)
            if i == 0:
                seq_x = seq_x_tmp
                seq_x_mark = seq_x_mark_tmp
                seq_y = seq_y_tmp
                seq_y_mark = seq_y_mark_tmp
            else:
                seq_x = np.concatenate([seq_x, seq_x_tmp], axis=0)
                seq_x_mark = np.concatenate([seq_x_mark, seq_x_mark_tmp], axis=0)

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(self, root_path, window_len, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(os.path.join(self.root_path, self.data_path), "r", encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n').split(',')  # 去除文本中的换行符
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_y.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
