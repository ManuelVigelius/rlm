import pandas as pd
from tsfeatures import tsfeatures, stl_features, entropy, lumpiness, stability, statistics, series_length

def get_features(ts_df, start_date=None, end_date=None):
    # Create a copy of the dataframe to slice
    df_slice = ts_df.copy()
    
    # Apply date filtering if provided
    if start_date is not None:
        start_dt = pd.to_datetime(start_date)
        df_slice = df_slice[df_slice['ds'] >= start_dt]
    
    if end_date is not None:
        end_dt = pd.to_datetime(end_date)
        df_slice = df_slice[df_slice['ds'] <= end_dt]
    
    # Print the actual start and end dates from the sliced dataframe

    actual_start = df_slice['ds'].min().strftime('%Y-%m-%d %H:%M:%S')
    actual_end = df_slice['ds'].max().strftime('%Y-%m-%d %H:%M:%S')

    # Extract features from the (potentially sliced) time series
    features_df = tsfeatures(df_slice, freq=48, features=[stl_features, entropy, lumpiness, stability, statistics, series_length])  # freq=48 for 30-minute intervals (48 per day)
    features_df['start_date'] = actual_start
    features_df['end_date'] = actual_end

    key_features = ['trend', 'seasonal_strength', 'linearity', 'curvature', 
                    'entropy', 'lumpiness', 'stability', 
                    'mean', 'variance', 'min', 'max', 'series_length', 'start_date', 'end_date']
    available_keys = [k for k in key_features if k in features_df.columns]
    return {
        key: features_df[key].values.item() for key in available_keys
    }
