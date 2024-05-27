import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

def init_prep(data, TRAIN=False):
    """데이터 전처리 함수
    ver_01 2024.05.20. 
    ver_02 2024.05.26.
        : E_scr_pv != 7 제거 추가

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
        # - 이거 의미 없는데?
        data = data[data['n_temp_sv'] != 0]
        
        # (4) E_scr_pv != 7
        data = data[data['E_scr_pv'] != 7]

    # (4) Eliminate Columns : E_scr_sv, c_temp_sv, n_temp_sv, s_temp_sv, k_rpm_sv, time
    data = data.drop(['E_scr_sv', 'c_temp_sv', 'n_temp_sv', 's_temp_sv', "k_rpm_sv", 'time'], axis=1)
    
    return data

def SM_prep(data):
    """승민님 전처리 함수
    ver_01 : 2024.05.26.

    Returns:
        mydata2 : 10월전 데이터 (학습용)
        after_data : 10월 데이터 (테스트용)
    """
    # 2. Preprocess the data
    data.drop('Unnamed: 12', axis=1, inplace=True)
    data['time'] = pd.to_datetime(data['time']).dt.tz_localize(None)

    # 2.1 Split the data
    oct_1 = pd.Timestamp('2023-10-01')
    before_data = data[data['time'] < oct_1]
    after_data = data[data['time'] >= oct_1]
    mydata = before_data.drop('time', axis=1)

    # 무의미 sv 제거
    mydata.drop(['s_temp_sv', 'c_temp_sv'], axis=1, inplace=True)

    # E_scr_sv가 8인 데이터만 남기기
    mydata = mydata[mydata['E_scr_sv']==8]

    # E_scr_sv, n_temp_sv 제거
    mydata.drop(['E_scr_sv', 'n_temp_sv'], axis=1, inplace=True)

    # scale_pv < 4
    mydata2 = mydata[mydata['scale_pv'] < 4]

    # c_temp_pv >= 68
    # - 65.1 일때 scale_pv는 모두 0
    mydata2 = mydata2[mydata2['c_temp_pv'] >= 68]

    # k_rpm_sv, E_scr_sv 제거 
    mydata2.drop('k_rpm_sv', axis=1, inplace=True)
    mydata2.drop('E_scr_pv', axis=1, inplace=True)

    # filtering 분석 : k_rpm_pv < 50 확인
    # - 결과 : 2.5이상이 22개, 해당 구간 삭제
    mydata2 = mydata2[mydata2['k_rpm_pv'] >= 50]

    # scale_pv : 0 초과 2.5 미만 삭제
    # - .unique()보단 hist나 box로 보는 게 이상치 판단에 도움될 듯!
    # - 결과 : 0 초과 2.5 미만인 데이터 삭제
    mydata2 = mydata2[(mydata2['scale_pv'] <= 0) | (mydata2['scale_pv'] >= 2.5)]

    # 이상치로 판단해 k_rpm_pv < 162.5 제거
    mydata2 = mydata2[mydata2['k_rpm_pv'] >= 162.5]

    return mydata2, after_data
