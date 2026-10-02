"""Препроцессинг транзакций (логика из соревнования teta-ml-1-2025).

Все статистики по train-данным (маппинг категорий, mean-encoding, импутер)
считаются ОДИН раз при первом обращении и кэшируются, поэтому обработка
одного сообщения из Kafka не требует повторных groupby по всему train.
"""
import logging
import os

import numpy as np
import pandas as pd
from geopy.distance import great_circle
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)
RANDOM_STATE = 42

TARGET_COL = 'target'
CATEGORICAL_COLS = ['gender', 'merch', 'cat_id', 'one_city', 'us_state', 'jobs']
CONTINUOUS_COLS = ['amount', 'population_city']
DROP_COLS = ['name_1', 'name_2', 'street', 'post_code']
N_CATS = 50
TRAIN_PATH = os.getenv('TRAIN_DATA_PATH', './train_data/train.csv')

_ARTIFACTS = {}


def add_time_features(df):
    logger.debug('Adding time features...')
    df['transaction_time'] = pd.to_datetime(df['transaction_time'])
    dt = df['transaction_time'].dt
    df['hour'] = dt.hour
    df['year'] = dt.year
    df['month'] = dt.month
    df['day_of_month'] = dt.day
    df['day_of_week'] = dt.dayofweek
    df.drop(columns='transaction_time', inplace=True)
    return df


def cat_encode(mapping, input_df, col):
    logger.debug('Encoding category: %s', col)
    return input_df.merge(mapping, how='left', on=col).drop(columns=col)


def add_distance_features(df):
    logger.debug('Calculating distances...')
    df['distance'] = df.apply(
        lambda x: great_circle(
            (x['lat'], x['lon']),
            (x['merchant_lat'], x['merchant_lon'])
        ).km,
        axis=1
    )
    return df.drop(columns=['lat', 'lon', 'merchant_lat', 'merchant_lon'])


def load_train_data():
    """Читает train.csv и строит признаки (вызывается при старте контейнера)."""
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(
            f'Не найден {TRAIN_PATH}. Скачайте train.csv с Kaggle '
            '(teta-ml-1-2025) и положите в fraud_detector/train_data/ '
            '(см. README).'
        )

    logger.info('Loading training data...')
    train = pd.read_csv(TRAIN_PATH).drop(columns=DROP_COLS)
    logger.info('Raw train data imported. Shape: %s', train.shape)

    train = add_time_features(train)

    for col in CATEGORICAL_COLS:
        new_col = col + '_cat'
        temp_df = train\
            .groupby(col, dropna=False)[[TARGET_COL]]\
            .count()\
            .sort_values(TARGET_COL, ascending=False)\
            .reset_index()\
            .set_axis([col, 'count'], axis=1)\
            .reset_index()
        temp_df['index'] = temp_df.apply(
            lambda x: np.nan if pd.isna(x[col]) else x['index'], axis=1)
        temp_df[new_col] = [
            'cat_NAN' if pd.isna(x)
            else 'cat_' + str(x) if x < N_CATS
            else f'cat_{N_CATS}+'
            for x in temp_df['index']
        ]
        train = train.merge(temp_df[[col, new_col]], how='left', on=col)

    train = add_distance_features(train)
    logger.info('Train data processed. Shape: %s', train.shape)
    return train


def _get_artifacts(train):
    """Считает и кэширует статистики по train."""
    key = id(train)
    if key in _ARTIFACTS:
        return _ARTIFACTS[key]

    cat_cols = [c + '_cat' for c in CATEGORICAL_COLS]
    cat_cols += ['hour', 'year', 'month', 'day_of_month', 'day_of_week']

    cat_mappings = {
        col: train[[col, col + '_cat']].drop_duplicates()
        for col in CATEGORICAL_COLS
    }
    means = {
        col: train.groupby(col)[[TARGET_COL]].mean()
                  .reset_index().rename(columns={TARGET_COL: f'{col}_mean_enc'})
        for col in cat_cols
    }
    cont_cols = CONTINUOUS_COLS + ['distance']
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer = imputer.fit(train[cont_cols])

    _ARTIFACTS[key] = {
        'cat_cols': cat_cols,
        'cat_mappings': cat_mappings,
        'means': means,
        'cont_cols': cont_cols,
        'imputer': imputer,
    }
    logger.info('Preprocessing artifacts cached')
    return _ARTIFACTS[key]


def run_preproc(train, input_df):
    """Главная функция препроцессинга одного/нескольких объектов."""
    art = _get_artifacts(train)
    input_df = input_df.drop(columns=DROP_COLS)

    for col in CATEGORICAL_COLS:
        input_df = cat_encode(art['cat_mappings'][col], input_df, col)
    logger.info('Categorical merging completed. Output shape: %s', input_df.shape)

    input_df = add_time_features(input_df)
    logger.info('Added time features. Output shape: %s', input_df.shape)

    for col in art['cat_cols']:
        input_df[col] = input_df[col].fillna('cat_NAN')
        input_df = input_df.merge(art['means'][col], how='left', on=col)
    logger.info('Categorical mean encoding completed. Output shape: %s', input_df.shape)

    input_df = add_distance_features(input_df)

    cont_cols = art['cont_cols']
    output_df = pd.concat([
        input_df.drop(columns=cont_cols),
        pd.DataFrame(art['imputer'].transform(input_df[cont_cols]), columns=cont_cols)
    ], axis=1)

    for col in cont_cols:
        output_df[col + '_log'] = np.log(output_df[col] + 1)
        output_df.drop(columns=col, inplace=True)

    logger.info('Continuous features preprocessing completed. Output shape: %s', output_df.shape)
    return output_df
