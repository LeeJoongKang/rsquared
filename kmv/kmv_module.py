"""
KMV 모델을 이용한 부도확률(PD) 계산 모듈
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from pykalman import KalmanFilter
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


class KMVModel:
    """KMV 모델을 이용한 PD 계산 클래스"""
    
    def __init__(self, data_path=None):
        """
        Parameters
        ----------
        data_path : str
            parquet 파일 경로
        """
        self.data_path = data_path
        self.df = None
        self.result_df = None
        
    def load_data(self, data_path=None):
        """데이터 로드 및 전처리"""
        if data_path:
            self.data_path = data_path
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터 로드 시작: {self.data_path}")
        
        # 데이터 로드
        self.df = pd.read_parquet(self.data_path)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 데이터 로드 완료: {self.df.shape}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 컬럼: {list(self.df.columns)}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 종목 수: {self.df['종목코드'].nunique()}")
        
        # 날짜 변환 및 정렬
        self.df['날짜'] = pd.to_datetime(self.df['날짜'])
        self.df = self.df.sort_values(['종목코드', '날짜']).reset_index(drop=True)
        
        # 기준금리 전처리
        if '기준금리(%)' in self.df.columns:
            self.df['risk_free_rate'] = self.df['KOFR'] / 100
        else:
            self.df['risk_free_rate'] = 0.025  # 기본값 2.5%
            
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 날짜 범위: {self.df['날짜'].min()} ~ {self.df['날짜'].max()}")
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 전처리 완료!")
        
    @staticmethod
    def estimate_V_sigmaV(E, sigma_E, DP, r, T=1.0, max_iter=100, tol=1e-6):
        """KMV 모델 - 자산가치와 자산변동성 추정"""
        V, sigma_V = E + DP, sigma_E
        
        for _ in range(max_iter):
            d1 = (np.log(V / DP) + (r + 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
            d2 = d1 - sigma_V * np.sqrt(T)
            
            E_est = V * norm.cdf(d1) - DP * np.exp(-r * T) * norm.cdf(d2)
            sigma_E_est = (V / E) * norm.cdf(d1) * sigma_V
            
            V_new = V * (E / E_est)
            sigma_V_new = sigma_V * (sigma_E / sigma_E_est)
            
            if abs(V_new - V) < tol and abs(sigma_V_new - sigma_V) < tol:
                break
            
            V, sigma_V = V_new, sigma_V_new
        
        return V, sigma_V
    
    @staticmethod
    def kmv_dd_pd(E, sigma_E, DP, r, T=1.0):
        """KMV 모델 - PD 계산"""
        V_est, sigma_V_est = KMVModel.estimate_V_sigmaV(E, sigma_E, DP, r, T)
        DD = (np.log(V_est / DP) + (r - 0.5 * sigma_V_est**2) * T) / (sigma_V_est * np.sqrt(T))
        PD = norm.cdf(-DD)
        return V_est, sigma_V_est, DD, PD
    
    @staticmethod
    def kalman_smooth(series):
        """칼만 필터 평활화"""
        clean_series = series.dropna()
        if len(clean_series) < 2:
            return series
        
        kf = KalmanFilter(
            transition_matrices=[1],
            observation_matrices=[1],
            initial_state_mean=clean_series.iloc[0],
            initial_state_covariance=1,
            observation_covariance=0.01,
            transition_covariance=0.001
        )
        
        state_means, _ = kf.filter(clean_series.values)
        result = pd.Series(index=series.index, dtype=float)
        result.loc[clean_series.index] = state_means.flatten()
        
        return result
    
    def calculate_pd_for_stock(self, stock_data, window=60):
        """단일 종목 PD 계산"""
        results = []
        stock_code = stock_data['종목코드'].iloc[0]
        
        # 로그수익률 계산
        if 'log_return' not in stock_data.columns or stock_data['log_return'].isna().all():
            stock_data['log_return'] = np.log(stock_data['close_price'] / stock_data['close_price'].shift(1))
        
        # 변동성 계산
        stock_data['rolling_sigmaE'] = stock_data['log_return'].rolling(window).std() * np.sqrt(252)
        
        # equity_vol 우선 사용, 없으면 rolling 사용
        if 'equity_vol' in stock_data.columns:
            stock_data['sigma_E'] = stock_data['equity_vol'].fillna(stock_data['rolling_sigmaE'])
        else:
            stock_data['sigma_E'] = stock_data['rolling_sigmaE']
        
        # 각 날짜별 PD 계산
        for _, row in stock_data.iterrows():
            E = row['equity_value']
            sigma_E = row['sigma_E']
            DP = row['EDF']
            r = row['risk_free_rate']
            
            if pd.isna(E) or pd.isna(sigma_E) or pd.isna(DP) or E <= 0 or sigma_E <= 0 or DP <= 0:
                results.append({
                    '날짜': row['날짜'],
                    '종목코드': stock_code,
                    'PD_raw': np.nan,
                    'V_est': np.nan,
                    'sigma_V_est': np.nan,
                    'DD': np.nan
                })
                continue
            
            try:
                V_est, sigma_V_est, DD, PD = self.kmv_dd_pd(E, sigma_E, DP, r, T=1.0)
                results.append({
                    '날짜': row['날짜'],
                    '종목코드': stock_code,
                    'PD_raw': PD,
                    'V_est': V_est,
                    'sigma_V_est': sigma_V_est,
                    'DD': DD
                })
            except:
                results.append({
                    '날짜': row['날짜'],
                    '종목코드': stock_code,
                    'PD_raw': np.nan,
                    'V_est': np.nan,
                    'sigma_V_est': np.nan,
                    'DD': np.nan
                })
        
        return results
    
    def calculate_all_pd(self):
        """전체 종목 PD 계산"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PD 계산 시작...")
        all_results = []
        unique_stocks = self.df['종목코드'].unique()
        total_stocks = len(unique_stocks)
        
        for i, stock_code in enumerate(unique_stocks):
            if i % 50 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] 진행률: {i+1}/{total_stocks} ({100*(i+1)/total_stocks:.1f}%)")
            
            stock_data = self.df[self.df['종목코드'] == stock_code].copy()
            stock_results = self.calculate_pd_for_stock(stock_data)
            all_results.extend(stock_results)
        
        self.result_df = pd.DataFrame(all_results)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] PD 계산 완료! 총 {len(self.result_df)}개 데이터")
        
    def apply_smoothing(self):
        """PD 보정 (EWMA, 칼만필터)"""
        print(f"[{datetime.now().strftime('%H:%M:%S')}] EWMA 보정 적용...")
        self.result_df['PD_EWMA'] = self.result_df.groupby('종목코드')['PD_raw'].transform(
            lambda x: x.ewm(alpha=0.05).mean()
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 칼만 필터 보정 적용...")
        self.result_df['PD_Kalman'] = self.result_df.groupby('종목코드')['PD_raw'].transform(
            self.kalman_smooth
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 변동성 계산...")
        self.result_df['Vol_Raw'] = self.result_df.groupby('종목코드')['PD_raw'].transform(
            lambda x: x.rolling(30).std()
        )
        self.result_df['Vol_EWMA'] = self.result_df.groupby('종목코드')['PD_EWMA'].transform(
            lambda x: x.rolling(30).std()
        )
        self.result_df['Vol_Kalman'] = self.result_df.groupby('종목코드')['PD_Kalman'].transform(
            lambda x: x.rolling(30).std()
        )
        
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 보정 완료!")
        
    def print_summary(self):
        """결과 요약 출력"""
        print("\n" + "="*60)
        print("PD 계산 결과 요약")
        print("="*60)
        print(f"총 데이터 수: {len(self.result_df):,}")
        print(f"종목 수: {self.result_df['종목코드'].nunique()}")
        print(f"날짜 범위: {self.result_df['날짜'].min()} ~ {self.result_df['날짜'].max()}")
        
        print("\nPD 기초 통계:")
        pd_stats = self.result_df[['PD_raw', 'PD_EWMA', 'PD_Kalman']].describe()
        print(pd_stats)
        
        print("\nPD_raw 분포:")
        pd_counts = self.result_df['PD_raw'].value_counts().head(5)
        print(pd_counts)
        
    def save_results(self, output_path='results/kmv_pd_results.parquet', save_csv=True):
        """결과 저장 (results 폴더에 저장)"""
        import os
        
        # results 폴더 확인 및 생성
        if os.path.dirname(output_path):
            results_dir = os.path.dirname(output_path)
        else:
            results_dir = 'results'
            output_path = os.path.join(results_dir, output_path)
            
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        
        # parquet 저장
        self.result_df.to_parquet(output_path, index=False)
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 결과 저장 완료: {output_path}")
        
        # CSV 저장
        if save_csv:
            csv_path = output_path.replace('.parquet', '.csv')
            self.result_df.to_csv(csv_path, index=False)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] CSV 저장 완료: {csv_path}")
            
    def run(self, data_path=None, output_path='kmv_pd_results.parquet'):
        """전체 프로세스 실행"""
        print("\n" + "="*60)
        print("KMV PD 계산 시작")
        print("="*60)
        
        # 1. 데이터 로드
        self.load_data(data_path)
        
        # 2. PD 계산
        self.calculate_all_pd()
        
        # 3. 보정 적용
        self.apply_smoothing()
        
        # 4. 결과 요약
        self.print_summary()
        
        # 5. 결과 저장
        self.save_results(output_path)
        
        print("\n" + "="*60)
        print("모든 작업 완료!")
        print("="*60)
        
        return self.result_df