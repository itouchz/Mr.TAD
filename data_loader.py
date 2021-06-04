# Dataset Loader
# Univariate Datasets: KDD Cup'21 UCR, Yahoo S5, Power-demand
# Multivariate Datasets: NASA, ECG, 2D Gesture, SMD

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm.notebook import tqdm

TIME_STEPS = 6

# Generated training sequences for use in the model.
def create_sequences(values, time_steps=TIME_STEPS, stride=1):
    output = []
    for i in range(0, len(values) - time_steps, stride):
        output.append(values[i : (i + time_steps)])
    return np.stack(output)


def load_kdd_cup_urc(seq_length=0, stride=1):
    # sequence length: 
    # source: https://compete.hexagon-ml.com/practice/competition/39/#data
    data_path = './datasets/kdd-cup-2021'
    datasets = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])
    
    x_train, x_test = [], []
    for data in tqdm(datasets):
        loc = int(data.split('_')[-1].split('.')[0])
        df = pd.read_csv(f'{data_path}/{data}', names=['values'], dtype={'values': float})
        train_df = df.iloc[:loc]
        test_df = df.iloc[loc:]
        
        scaler = MinMaxScaler()
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
                
        if seq_length > 0:
            x_train.append(create_sequences(train_df, time_steps=seq_length, stride=stride))
            x_test.append(create_sequences(test_df, time_steps=seq_length, stride=stride))
        else:
            x_train.append(train_df)
            x_test.append(test_df)
        
    return {'x_train': x_train, 'x_test': x_test, 'with_label': False}


def load_yahoo(dataset, seq_length=0, stride=1):
    # sequence length: 
    # source: https://webscope.sandbox.yahoo.com/catalog.php?datatype=s&did=70
    data_path = f'./datasets/yahoo/{dataset}Benchmark'
    datasets = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])

    x_train, x_test, y_test = [], [], []
    for data in tqdm(datasets):
        df = pd.read_csv(f'{data_path}/{data}')[['value']]
        label_df = pd.read_csv(f'{data_path}/{data}')[['is_anomaly']]

        test_idx = int(df.shape[0]*0.3) # train 70% test 30% (Ref. RAMED)
        train_df = df.iloc[:-test_idx]
        test_df = df.iloc[-test_idx:]
        labels = label_df.iloc[-test_idx:]

        scaler = MinMaxScaler()
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)

        if seq_length > 0:
            x_train.append(create_sequences(train_df, time_steps=seq_length, stride=stride))
            x_test.append(create_sequences(test_df, time_steps=seq_length, stride=stride))
            y_test.append(create_sequences(labels, time_steps=seq_length, stride=stride))
        else:
            x_train.append(train_df)
            x_test.append(test_df)
            y_test.append(labels)
    return {'x_train': x_train, 'x_test': x_test, 'y_test': y_test}

def load_yahoo_A1(seq_length=0, stride=1):
    return load_yahoo('A1', seq_length, stride)

def load_yahoo_A2(seq_length=0, stride=1):
    return load_yahoo('A2', seq_length, stride)

def load_yahoo_A3(seq_length=0, stride=1):
    return load_yahoo('A3', seq_length, stride)

def load_yahoo_A4(seq_length=0, stride=1):
    return load_yahoo('A4', seq_length, stride)

def load_power_demand(seq_length=0, stride=1):
    # sequence length: 80 (THOC)
    # stride: 1 (THOC)
    # source: https://www.cs.ucr.edu/~eamonn/discords/power_data.txt
    data_path = './datasets/power-demand'
    
    x_train, x_test = [], []
    df = pd.read_csv(f'{data_path}/power_data.txt', names=['values'], dtype={'values': float})

    test_idx = int(df.shape[0]*0.3) # train 70% test 30% (Ref. RAMED)
    train_df = df.iloc[:-test_idx]
    test_df = df.iloc[-test_idx:]

    scaler = MinMaxScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)
    
    
    if seq_length > 0:
        x_train.append(create_sequences(train_df, time_steps=seq_length, stride=stride))
        x_test.append(create_sequences(test_df, time_steps=seq_length, stride=stride))
    else:
        x_train.append(train_df)
        x_test.append(test_df)


    return {'x_train': x_train, 'x_test': x_test, 'with_label': False}

def load_nasa(seq_length=0, stride=1):
    # sequence length: 100 (THOC)
    # stride: 100 (THOC)
    # source: https://s3-us-west-2.amazonaws.com/telemanom/data.zip    
    data_path = './datasets/nasa'
    labels = pd.read_csv(f'{data_path}/labeled_anomalies.csv')

    x_train, x_test = [], []
    for dataset in tqdm(labels['spacecraft'].unique()):
        subdata_train, subdata_test = [], []
        for data in sorted(labels[labels['spacecraft'] == dataset]['chan_id'].values):
            train_df = np.load(f'{data_path}/train/{data}.npy')
            test_df = np.load(f'{data_path}/test/{data}.npy')

            subdata_train.append(train_df)
            subdata_test.append(test_df)

        subdata_train, subdata_test = np.concatenate(subdata_train), np.concatenate(subdata_test)
        scaler = MinMaxScaler()
        subdata_train = scaler.fit_transform(subdata_train)
        subdata_test = scaler.transform(subdata_test)
        
        if seq_length > 0:
            x_train.append(create_sequences(subdata_train, time_steps=seq_length, stride=stride))
            x_test.append(create_sequences(subdata_test, time_steps=seq_length, stride=stride))
        else:
            x_train.append(subdata_train)
            x_test.append(subdata_test)
        
    return {'x_train': x_train, 'x_test': x_test, 'with_label': False}

def load_ecg(seq_length=0, stride=1):
    # sequence length:
    # stride: 
    # source: https://www.cs.ucr.edu/~eamonn/discords/ECG_data.zip
    data_path = './datasets/ECG'
    datasets = sorted([f for f in os.listdir(data_path) if os.path.isfile(os.path.join(data_path, f))])

    x_train, x_test = [], []
    for data in tqdm(datasets):
        df = pd.read_csv(f'{data_path}/{data}', header=None, sep='\t').drop(columns=0)

        test_idx = int(df.shape[0]*0.3) # train 70% test 30% (Ref. RAMED)
        train_df = df.iloc[:-test_idx]
        test_df = df.iloc[-test_idx:]

        scaler = MinMaxScaler()
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
        
        if seq_length > 0:
            x_train.append(create_sequences(train_df, time_steps=seq_length, stride=stride))
            x_test.append(create_sequences(test_df, time_steps=seq_length, stride=stride))
        else:
            x_train.append(train_df)
            x_test.append(test_df)
            
    return {'x_train': x_train, 'x_test': x_test, 'with_label': False}

def load_gesture(seq_length=0, stride=1):
    # sequence length: 80 (THOC)
    # stride: 1 (THOC)
    # source: https://www.cs.ucr.edu/~eamonn/discords/ann_gun_CentroidA
    data_path = './datasets/2d-gesture'

    x_train, x_test = [], []
    df = pd.read_csv(f'{data_path}/data.txt', header=None, sep='\s+')

    test_idx = int(df.shape[0]*0.3) # train 70% test 30% (Ref. RAMED)
    train_df = df.iloc[:-test_idx]
    test_df = df.iloc[-test_idx:]

    scaler = MinMaxScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    if seq_length > 0:
        x_train.append(create_sequences(train_df, time_steps=seq_length, stride=stride))
        x_test.append(create_sequences(test_df, time_steps=seq_length, stride=stride))
    else:
        x_train.append(train_df)
        x_test.append(test_df)
        
    return {'x_train': x_train, 'x_test': x_test, 'with_label': False}

def load_smd(seq_length=0, stride=1):
    # source: https://github.com/NetManAIOps/OmniAnomaly/tree/master/ServerMachineDataset
    data_path = './datasets/smd-omni'
    datasets = sorted([f for f in os.listdir(f'{data_path}/train') if os.path.isfile(os.path.join(f'{data_path}/train', f))])
    
    x_train, x_test = [], []
    for data in tqdm(datasets):
        train_df = pd.read_csv(f'{data_path}/train/{data}', header=None, sep=',')
        test_df = pd.read_csv(f'{data_path}/test/{data}', header=None, sep=',')

        scaler = MinMaxScaler()
        train_df = scaler.fit_transform(train_df)
        test_df = scaler.transform(test_df)
        
        if seq_length > 0:
            x_train.append(create_sequences(train_df, time_steps=seq_length, stride=stride))
            x_test.append(create_sequences(test_df, time_steps=seq_length, stride=stride))
        else:
            x_train.append(train_df)
            x_test.append(test_df)

    return {'x_train': x_train, 'x_test': x_test, 'with_label': False}
