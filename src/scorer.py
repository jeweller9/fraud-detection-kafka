import logging
import os

import pandas as pd
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

MODEL_PATH = os.getenv('MODEL_PATH', './models/my_catboost.cbm')

logger.info('Importing pretrained model...')
model = CatBoostClassifier()
model.load_model(MODEL_PATH)

# Оптимальный порог, подобранный в соревновании
model_th = float(os.getenv('FRAUD_THRESHOLD', '0.98'))
logger.info('Pretrained model imported successfully...')

CATEGORICAL_FEATURES = [
    'hour', 'year', 'month', 'day_of_month', 'day_of_week',
    'gender_cat', 'merch_cat', 'cat_id_cat', 'one_city_cat',
    'us_state_cat', 'jobs_cat',
]


def make_pred(dt, source_info="kafka"):
    """Возвращает DataFrame со столбцами score и fraud_flag."""
    # Берём ровно те признаки и в том порядке, которые ждёт модель
    dt = dt[model.feature_names_].copy()

    # CatBoost ждёт категориальные признаки как строки
    for col in CATEGORICAL_FEATURES:
        dt[col] = dt[col].astype(str)

    proba = model.predict_proba(dt)[:, 1]
    submission = pd.DataFrame({
        'score': proba,
        'fraud_flag': (proba > model_th).astype(int),
    })
    logger.info('Prediction complete for data from %s', source_info)
    return submission
