import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def prep_ver_01(data, TRAIN=False):
    """데이터 전처리 함수 : 2024.05.20. ver_01

    Args:
        data : Dataframe
        TRAIN : 기본 2 < scale_pv < 4, 학습용 추가 전처리 반영 (default=False)

    Returns:
        data : Dataframe
    """
    # (1) 2 < scale_pv < 4
    data = data[(data['scale_pv'] > 2) & (data['scale_pv'] < 4)]
    
    if TRAIN:
    # (2) 100 < k_rpm_pv
        data = data[data['k_rpm_pv'] > 100]

        # (3) n_temp_sv != 0
        data = data[data['n_temp_sv'] != 0]

    # (4) Eliminate Columns : E_scr_sv, c_temp_sv, n_temp_sv, s_temp_sv, k_rpm_sv, time
    data = data.drop(['E_scr_sv', 'c_temp_sv', 'n_temp_sv', 's_temp_sv', "k_rpm_sv", 'time'], axis=1)
    
    return data

