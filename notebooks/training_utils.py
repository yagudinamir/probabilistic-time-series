import pandas as pd
import numpy as np
from pandas import Timedelta
from tqdm import tqdm

from data_utils import load_features_and_targets
from sklearn.preprocessing import StandardScaler
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName

dataset_metadata = {
 'freq': '1H',
 'prediction_length': 1,
 'features_history': 5,
 'start': pd.Timestamp('2019-01-01 00:00:00', freq='H'),
 'num_train_days': 14,
 'num_test_days': 3
}

DATASET_PATH = '/home/ayagudin/dl/probabilistic-time-series/lobster-preprocessed-dataset/'



def get_scaled_datasets():
    features_per_day, targets_per_day = load_features_and_targets(DATASET_PATH)
    timestamps_to_cut = 200
    for day in features_per_day:
        features_per_day[day] = features_per_day[day][timestamps_to_cut:-timestamps_to_cut]
        targets_per_day[day] = targets_per_day[day][timestamps_to_cut:-timestamps_to_cut].reshape(-1, 1)
    
    full_features = np.concatenate(list(features_per_day.values()))
    full_targets = np.concatenate(list(targets_per_day.values()))
    print(full_features.shape, full_targets.shape)
    
    target_scaler = StandardScaler()
    feature_scaler = StandardScaler()
    
    feature_scaler.fit(full_features)
    target_scaler.fit(full_targets)
    
    for day in features_per_day:
        features_per_day[day] = feature_scaler.transform(features_per_day[day])
        targets_per_day[day] = target_scaler.transform(targets_per_day[day]).reshape(-1)
        
    days = sorted(list(features_per_day.keys()))
    train_days = days[:dataset_metadata['num_train_days']]
    test_days = days[dataset_metadata['num_train_days']:]
    print('len(train_days), len(test_days)', len(train_days), len(test_days))
        
    train_ds = ListDataset(
    [
        {
            FieldName.TARGET: targets_per_day[day][None],
            FieldName.START: dataset_metadata['start'],
            FieldName.FEAT_DYNAMIC_REAL: features_per_day[day].T
        }
        for day in train_days
    ],
    freq=dataset_metadata['freq'],
    one_dim_target=False
    )
    
    test_ds = ListDataset(
        [
            {
                FieldName.TARGET: targets_per_day[day][None],
                FieldName.START: dataset_metadata['start'],
                FieldName.FEAT_DYNAMIC_REAL: features_per_day[day].T
            }
            for day in test_days
        ],
        freq=dataset_metadata['freq'],
        one_dim_target=False
    )    
        
    return train_ds, test_ds  


def get_rolled_dataset(dataset, context_length, prediction_length, max_size=None):
    keys_to_split = [
        FieldName.FEAT_DYNAMIC_REAL,
        FieldName.TARGET
    ]
    window_size = context_length + prediction_length
    
    rolled_dataset = []
    
    for dataset_entry in tqdm(iter(dataset)):
        for i in range(dataset_entry[FieldName.TARGET].shape[-1] - window_size):
            if len(rolled_dataset) >= max_size:
                break
            new_entry = {}
            for key in dataset_entry.keys():
                if key not in keys_to_split:
                    new_entry[key] = dataset_entry[key]
                else:
                    new_entry[key] = dataset_entry[key][..., i: i + window_size]
                    
            new_entry[FieldName.START] = dataset_entry[FieldName.START] + Timedelta(dataset_metadata['freq']) * i
            
            rolled_dataset.append(new_entry)
    
    return ListDataset(rolled_dataset, freq=dataset_metadata['freq'], one_dim_target=False)
            