import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from typing import Dict, List, Tuple, Optional, Union
import logging
import matplotlib.font_manager as fm
import platform
import urllib.request
import zipfile
from pathlib import Path
from dotenv import load_dotenv
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def download_nanum_font():
    """나눔고딕 폰트 다운로드 및 설치"""
    font_dir = os.path.join(os.path.dirname(__file__), 'fonts')
    font_path = os.path.join(font_dir, 'NanumGothic.ttf')
    
    # 폰트 디렉토리가 없으면 생성
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
    
    # 폰트 파일이 없으면 다운로드
    if not os.path.exists(font_path):
        # 여러 다운로드 URL 시도
        urls = [
            "https://raw.githubusercontent.com/moonspam/NanumGothic/master/NanumGothic.ttf",
            "https://cdn.jsdelivr.net/gh/moonspam/NanumGothic@latest/NanumGothic.ttf",
            "https://github.com/moonspam/NanumGothic/raw/master/NanumGothic.ttf"
        ]
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        success = False
        for url in urls:
            try:
                logging.info(f"나눔고딕 폰트 다운로드 시도 중... (URL: {url})")
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req) as response:
                    with open(font_path, 'wb') as out_file:
                        out_file.write(response.read())
                logging.info(f"폰트 다운로드 완료: {font_path}")
                success = True
                break
            except Exception as e:
                logging.warning(f"해당 URL에서 다운로드 실패: {str(e)}")
                continue
        
        if not success:
            logging.error("모든 다운로드 시도 실패. 기본 폰트를 사용합니다.")
            return None
    
    return font_path

def setup_visualization():
    """시각화 설정"""
    # Windows 시스템에서는 기본 설치된 맑은 고딕 사용
    if platform.system() == 'Windows':
        font_name = 'Malgun Gothic'
        plt.rcParams['font.family'] = font_name
        plt.rcParams['axes.unicode_minus'] = False
        
        # matplotlib 기본 설정
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': [10, 6],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        })
        
        # seaborn 설정
        sns.set_style("whitegrid", {
            'font.family': font_name,
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.color': '.8',
            'grid.linestyle': '-',
            'grid.alpha': 0.3
        })
        sns.set_context("notebook", font_scale=1.2)
        
        logging.info("맑은 고딕 폰트로 시각화 설정 완료")
        return font_name
    
    # Windows가 아닌 경우 나눔고딕 폰트 설치 시도
    font_path = download_nanum_font()
    
    if font_path and os.path.exists(font_path):
        # 폰트 매니저 초기화
        fm._load_fontmanager(try_read_cache=False)
        
        # 폰트 직접 등록
        fm.fontManager.addfont(font_path)
        
        # matplotlib 설정
        plt.rcParams['font.family'] = 'NanumGothic'
        plt.rcParams['axes.unicode_minus'] = False
        
        # matplotlib 기본 설정
        plt.style.use('default')
        plt.rcParams.update({
            'figure.figsize': [10, 6],
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.facecolor': 'white',
            'figure.facecolor': 'white',
            'axes.spines.top': False,
            'axes.spines.right': False,
            'axes.labelcolor': '#333333',
            'xtick.color': '#333333',
            'ytick.color': '#333333',
            'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
        })
        
        # seaborn 설정
        sns.set_style("whitegrid", {
            'font.family': 'NanumGothic',
            'axes.facecolor': 'white',
            'axes.grid': True,
            'grid.color': '.8',
            'grid.linestyle': '-',
            'grid.alpha': 0.3
        })
        sns.set_context("notebook", font_scale=1.2)
        
        logging.info("나눔고딕 폰트로 시각화 설정 완료")
        return 'NanumGothic'
    else:
        # 폰트 설치 실패 시 기본 폰트 사용
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['axes.unicode_minus'] = False
        logging.warning("폰트 설정 실패. 기본 폰트를 사용합니다.")
        return 'sans-serif'

# 시각화 설정 적용
FONT_FAMILY = setup_visualization()

class CSVAnalyzer:
    """CSV 파일 기반 데이터 분석 시스템의 핵심 클래스"""
    
    def __init__(self, csv_path: str = "data/apart_real_estate_20250404.csv"):
        """
        CSV Analyzer 초기화
        
        Args:
            csv_path: CSV 파일 경로 (기본값: data/apart_real_estate_20250404.csv)
        """
        self.csv_path = csv_path
        self.df = None
        self.numeric_cols = []
        self.categorical_cols = []
        self.date_cols = []
        self.summary_stats = {}
        self.agent = None
        
        # 데이터 로드 시도
        self._load_data()
        
        # LangChain Agent 초기화
        self._initialize_agent()
        
    def _load_data(self) -> None:
        """CSV 파일을 로드하고 데이터프레임 생성"""
        try:
            if not os.path.exists(self.csv_path):
                logging.error(f"파일을 찾을 수 없습니다: {self.csv_path}")
                return
            
            # 여러 인코딩 시도
            encodings = ['utf-8', 'cp949', 'euc-kr', 'latin1']
            for encoding in encodings:
                try:
                    self.df = pd.read_csv(self.csv_path, encoding=encoding)
                    logging.info(f"데이터 로드 완료. 크기: {self.df.shape}, 인코딩: {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
                    return
            
            if self.df is None:
                logging.error("지원하는 모든 인코딩으로 시도했지만 파일을 읽을 수 없습니다.")
                return
                
            # 데이터 타입 분류
            self._classify_columns()
            
        except Exception as e:
            logging.error(f"데이터 로드 중 오류 발생: {str(e)}")
            
    def _classify_columns(self) -> None:
        """데이터프레임의 컬럼을 타입별로 분류"""
        if self.df is None:
            return
            
        for col in self.df.columns:
            # 수치형 컬럼 확인
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.numeric_cols.append(col)
            # 날짜형 컬럼 확인 (문자열 형태의 날짜도 탐지)
            elif self._is_date_column(col):
                self.date_cols.append(col)
                # 날짜형으로 변환
                try:
                    self.df[col] = pd.to_datetime(self.df[col])
                except:
                    # 변환 실패 시 원래대로 두고 범주형으로 분류
                    self.date_cols.remove(col)
                    self.categorical_cols.append(col)
            else:
                self.categorical_cols.append(col)
                
        logging.info(f"컬럼 분류 완료. 수치형: {len(self.numeric_cols)}, 범주형: {len(self.categorical_cols)}, 날짜형: {len(self.date_cols)}")
    
    def _is_date_column(self, column: str) -> bool:
        """컬럼이 날짜형인지 확인"""
        if self.df is None:
            return False
            
        # 샘플 20개만 확인
        sample_size = min(20, len(self.df))
        sample_vals = self.df[column].dropna().head(sample_size)
        
        if len(sample_vals) == 0:
            return False
            
        try:
            # infer_datetime_format 대신 format='mixed' 사용
            pd.to_datetime(sample_vals, format='mixed')
            return True
        except:
            return False
    
    def get_basic_info(self) -> Dict:
        """데이터프레임 기본 정보 반환"""
        if self.df is None:
            return {}
            
        info = {
            "shape": self.df.shape,
            "columns": list(self.df.columns),
            "numeric_columns": self.numeric_cols,
            "categorical_columns": self.categorical_cols,
            "date_columns": self.date_cols,
            "missing_values": self.df.isna().sum().to_dict(),
            "dtypes": self.df.dtypes.astype(str).to_dict()
        }
        return info
    
    def get_summary_statistics(self) -> Dict:
        """데이터 요약 통계 생성"""
        if self.df is None:
            return {}
            
        self.summary_stats = {
            "numeric": {},
            "categorical": {},
            "date": {}
        }
        
        # 수치형 컬럼 통계
        if self.numeric_cols:
            self.summary_stats["numeric"] = self.df[self.numeric_cols].describe().T.to_dict()
            
        # 범주형 컬럼 통계
        for col in self.categorical_cols:
            value_counts = self.df[col].value_counts().head(10).to_dict()  # 상위 10개만
            unique_count = self.df[col].nunique()
            self.summary_stats["categorical"][col] = {
                "unique_count": unique_count,
                "top_values": value_counts,
                "missing_values": self.df[col].isna().sum()
            }
            
        # 날짜형 컬럼 통계
        for col in self.date_cols:
            try:
                self.summary_stats["date"][col] = {
                    "min": self.df[col].min(),
                    "max": self.df[col].max(),
                    "range_days": (self.df[col].max() - self.df[col].min()).days
                }
            except:
                pass
                
        return self.summary_stats
    
    def generate_visualizations(self) -> Dict:
        """다양한 시각화 그래프 생성"""
        if self.df is None:
            return {}
            
        visualizations = {}
        
        # 수치형 컬럼 히스토그램
        for col in self.numeric_cols[:5]:  # 처음 5개만
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(self.df[col].dropna(), ax=ax, kde=True)
            ax.set_title(f'{col} 분포')
            ax.set_xlabel(col)
            ax.set_ylabel('빈도')
            visualizations[f"{col}_hist"] = fig
            
        # 범주형 컬럼 막대 그래프
        for col in self.categorical_cols[:5]:  # 처음 5개만
            if self.df[col].nunique() <= 20:  # 고유값이 20개 이하인 경우에만
                fig, ax = plt.subplots(figsize=(10, 6))
                top_categories = self.df[col].value_counts().head(10)
                sns.barplot(x=top_categories.index, y=top_categories.values, ax=ax)
                ax.set_title(f'{col} 빈도')
                ax.set_xlabel(col)
                ax.set_ylabel('개수')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                visualizations[f"{col}_bar"] = fig
                
        # 상관관계 히트맵 (수치형 컬럼이 2개 이상인 경우)
        if len(self.numeric_cols) >= 2:
            fig, ax = plt.subplots(figsize=(12, 10))
            correlation_matrix = self.df[self.numeric_cols].corr()
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
            ax.set_title('상관관계 히트맵')
            plt.tight_layout()
            visualizations["correlation_heatmap"] = fig
            
        # 시계열 그래프 (날짜 컬럼이 있고 수치형 컬럼이 있는 경우)
        if self.date_cols and self.numeric_cols:
            date_col = self.date_cols[0]  # 첫 번째 날짜 컬럼 사용
            numeric_col = self.numeric_cols[0]  # 첫 번째 수치형 컬럼 사용
            
            try:
                # 시계열 데이터 준비
                time_series_df = self.df[[date_col, numeric_col]].dropna().sort_values(by=date_col)
                
                if len(time_series_df) > 0:
                    fig, ax = plt.subplots(figsize=(12, 6))
                    ax.plot(time_series_df[date_col], time_series_df[numeric_col])
                    ax.set_title(f'{numeric_col} 시계열 추세')
                    ax.set_xlabel(date_col)
                    ax.set_ylabel(numeric_col)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    visualizations["time_series"] = fig
            except:
                pass
                
        return visualizations
    
    def perform_clustering(self, n_clusters: int = 3) -> Dict:
        """K-means 클러스터링 수행 (수치형 데이터에 대해서만)"""
        if self.df is None or len(self.numeric_cols) < 2:
            return {}
            
        try:
            # 수치형 데이터만 선택
            numeric_data = self.df[self.numeric_cols].copy()
            
            # 결측치 제거
            numeric_data = numeric_data.dropna()
            
            if len(numeric_data) < 10:  # 데이터가 너무 적으면 클러스터링 수행 안함
                return {}
                
            # 표준화
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(numeric_data)
            
            # K-means 클러스터링
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(scaled_data)
            
            # 원본 데이터에 클러스터 할당
            numeric_data['cluster'] = clusters
            
            # 시각화 (처음 두 개의 변수만 사용)
            if len(self.numeric_cols) >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                for cluster in range(n_clusters):
                    cluster_data = numeric_data[numeric_data['cluster'] == cluster]
                    ax.scatter(
                        cluster_data[self.numeric_cols[0]], 
                        cluster_data[self.numeric_cols[1]], 
                        label=f'Cluster {cluster}'
                    )
                ax.set_title('K-means 클러스터링 결과')
                ax.set_xlabel(self.numeric_cols[0])
                ax.set_ylabel(self.numeric_cols[1])
                ax.legend()
                
                return {
                    "plot": fig,
                    "cluster_stats": numeric_data.groupby('cluster').mean().to_dict()
                }
                
        except Exception as e:
            logging.error(f"클러스터링 중 오류 발생: {str(e)}")
            
        return {}
    
    def search_data(self, query: str) -> pd.DataFrame:
        """간단한 쿼리 기반 데이터 검색"""
        if self.df is None:
            return pd.DataFrame()
            
        query = query.lower()
        results = self.df.copy()
        found_match = False
        
        try:
            # 컬럼명 정규화 (앞뒤 공백 제거)
            price_col = next((col for col in self.df.columns if '거래금액' in col), None)
            area_col = next((col for col in self.df.columns if '전용면적' in col), None)
            district_col = next((col for col in self.df.columns if '시군구' in col), None)
            apt_col = next((col for col in self.df.columns if '단지명' in col), None)
            
            # 지역 검색 (시군구, 도로명주소 등)
            if any(keyword in query for keyword in ['구', '동', '시']):
                for col in [district_col, '도로명']:
                    if col and col in self.df.columns:
                        for keyword in query.split():
                            if any(k in keyword for k in ['구', '동', '시']):
                                mask = results[col].astype(str).str.contains(keyword, case=False, na=False)
                                if mask.any():
                                    results = results[mask]
                                    found_match = True
                                    logging.info(f"지역 검색 결과: {len(results)}개 ({keyword})")
            
            # 아파트 이름 검색
            if '아파트' in query or '단지' in query:
                search_terms = [term for term in query.split() if term not in ['아파트', '단지', '보여줘', '검색', '찾아줘']]
                if search_terms and apt_col and apt_col in self.df.columns:
                    for term in search_terms:
                        if len(term) >= 2:
                            mask = results[apt_col].astype(str).str.contains(term, case=False, na=False)
                            if mask.any():
                                results = results[mask]
                                found_match = True
                                logging.info(f"아파트 검색 결과: {len(results)}개 ({term})")
            
            # 거래금액 검색
            if price_col and price_col in self.df.columns and any(keyword in query for keyword in ['가격', '거래금액', '억']):
                import re
                amounts = re.findall(r'\d+(?:\.\d+)?', query)
                if amounts:
                    value = float(amounts[0])
                    if '억' in query:
                        value = value * 10000
                    
                    # 쉼표 제거하고 숫자로 변환
                    prices = pd.to_numeric(results[price_col].astype(str).str.replace(',', '').str.replace('-', 'NaN'), errors='coerce')
                    
                    if "초과" in query or "넘는" in query:
                        results = results[prices > value]
                    elif "이상" in query:
                        results = results[prices >= value]
                    elif "미만" in query:
                        results = results[prices < value]
                    elif "이하" in query:
                        results = results[prices <= value]
                    else:  # 정확한 가격 검색
                        results = results[prices == value]
                    
                    if len(results) > 0:
                        found_match = True
                        logging.info(f"가격 검색 결과: {len(results)}개 ({value}만원)")
            
            # 면적 검색
            if area_col and area_col in self.df.columns and any(keyword in query for keyword in ['면적', '평', '㎡']):
                import re
                areas = re.findall(r'\d+(?:\.\d+)?', query)
                if areas:
                    value = float(areas[0])
                    if '평' in query:
                        value = value * 3.3058
                    
                    areas = pd.to_numeric(results[area_col].astype(str).str.replace('-', 'NaN'), errors='coerce')
                    
                    if "초과" in query:
                        results = results[areas > value]
                    elif "이상" in query:
                        results = results[areas >= value]
                    elif "미만" in query:
                        results = results[areas < value]
                    elif "이하" in query:
                        results = results[areas <= value]
                    else:  # 정확한 면적 검색
                        results = results[areas == value]
                    
                    if len(results) > 0:
                        found_match = True
                        logging.info(f"면적 검색 결과: {len(results)}개 ({value}㎡)")
            
            # 층수 검색
            if '층' in self.df.columns and ('층' in query):
                import re
                floors = re.findall(r'\d+(?:\.\d+)?', query)
                if floors:
                    value = float(floors[0])
                    floors = pd.to_numeric(results['층'].astype(str).str.replace('-', 'NaN'), errors='coerce')
                    
                    if "이상" in query or "높은" in query:
                        results = results[floors >= value]
                    elif "이하" in query or "낮은" in query:
                        results = results[floors <= value]
                    else:  # 정확한 층수 검색
                        results = results[floors == value]
                    
                    if len(results) > 0:
                        found_match = True
                        logging.info(f"층수 검색 결과: {len(results)}개 ({value}층)")
            
            # 건축년도 검색
            if '건축년도' in self.df.columns and any(keyword in query for keyword in ['년', '연도', '준공']):
                import re
                years = re.findall(r'\d{4}', query)
                if years:
                    year = int(years[0])
                    years = pd.to_numeric(results['건축년도'].astype(str).str.replace('-', 'NaN'), errors='coerce')
                    
                    if "이후" in query or "이상" in query or "넘는" in query:
                        results = results[years >= year]
                    elif "이전" in query or "이하" in query:
                        results = results[years <= year]
                    else:  # 정확한 년도 검색
                        results = results[years == year]
                    
                    if len(results) > 0:
                        found_match = True
                        logging.info(f"건축년도 검색 결과: {len(results)}개 ({year}년)")
            
            # 정렬 조건 처리
            if price_col and any(keyword in query for keyword in ['높은', '비싼', '내림차순']):
                prices = pd.to_numeric(results[price_col].astype(str).str.replace(',', '').str.replace('-', 'NaN'), errors='coerce')
                results = results.assign(price_num=prices).sort_values('price_num', ascending=False).drop('price_num', axis=1)
                found_match = True
            elif price_col and any(keyword in query for keyword in ['낮은', '싼', '오름차순']):
                prices = pd.to_numeric(results[price_col].astype(str).str.replace(',', '').str.replace('-', 'NaN'), errors='coerce')
                results = results.assign(price_num=prices).sort_values('price_num', ascending=True).drop('price_num', axis=1)
                found_match = True
            
            # 검색 조건에 맞는 결과가 없으면 빈 데이터프레임 반환
            if not found_match or len(results) == 0:
                logging.warning("검색 결과가 없습니다.")
                return pd.DataFrame()
            
            # 중복 제거
            results = results.drop_duplicates()
            
            logging.info(f"최종 검색 결과: {len(results)}개의 행이 일치")
            return results.head(100)  # 최대 100개 행만 반환
            
        except Exception as e:
            logging.error(f"검색 중 오류 발생: {str(e)}")
            return pd.DataFrame()

    def _initialize_agent(self):
        """LangChain Agent 초기화"""
        try:
            # Streamlit secrets에서 먼저 확인
            try:
                import streamlit as st
                api_key = st.secrets["OPENAI_API_KEY"]
            except:
                # .env 파일에서 확인
                api_key = os.getenv("OPENAI_API_KEY")
            
            if not api_key:
                logging.warning("OPENAI_API_KEY가 설정되지 않았습니다.")
                self.agent = None
                return
                
            # API 키가 유효한지 확인
            if not api_key.startswith("sk-"):
                logging.warning("유효하지 않은 OpenAI API 키 형식입니다.")
                self.agent = None
                return
                
            # OpenAI 모델 설정
            llm = ChatOpenAI(
                temperature=0,
                model="gpt-3.5-turbo",
                api_key=api_key
            )
            
            # 데이터프레임이 비어있지 않은지 확인
            if self.df is None or self.df.empty:
                logging.warning("데이터프레임이 비어있어 Agent를 초기화할 수 없습니다.")
                self.agent = None
                return
                
            # Agent 초기화 시도
            self.agent = create_pandas_dataframe_agent(
                llm,
                self.df,
                agent_type="zero-shot-react-description",
                verbose=True,
                max_iterations=5,
                max_execution_time=30,
                allow_dangerous_code=True
            )
            
            logging.info("LangChain Agent 초기화 완료")
            
        except Exception as e:
            logging.error(f"LangChain Agent 초기화 실패: {str(e)}")
            self.agent = None

    def query_data(self, query: str) -> Dict:
        """자연어 쿼리를 사용하여 데이터 분석"""
        if self.agent is None:
            return {
                "error": "LangChain Agent가 초기화되지 않았습니다. OpenAI API 키를 확인해주세요.",
                "success": False
            }
        
        try:
            # 컬럼명 정보 추가
            column_info = f"""
            현재 데이터프레임의 컬럼 정보:
            {', '.join(self.df.columns)}
            
            주요 컬럼 설명:
            - 아파트: 아파트 이름
            - 거래금액: 실거래가(만원)
            - 전용면적: 제곱미터
            - 층: 해당 아파트의 층수
            - 건축년도: 아파트 건축년도
            - 도로명주소: 도로명 주소
            """
            
            # 특별한 쿼리 패턴 처리
            if "가장 비싼" in query and "아파트" in query:
                translated_query = f"""
                {column_info}
                
                다음 작업을 수행해주세요:
                1. 거래금액을 기준으로 가장 비싼 아파트를 찾아주세요.
                2. 찾은 결과에서 다음 정보를 보여주세요:
                   - 아파트 이름
                   - 거래금액 (만원 단위로 표시)
                   - 전용면적
                   - 층수
                   - 도로명주소
                3. 결과를 다음과 같은 형식으로 한글로 작성해주세요:
                   "가장 비싼 아파트는 [아파트명]이며, 거래금액은 [금액]만원입니다.
                    전용면적은 [면적]㎡이고, [층]층에 위치해 있습니다.
                    주소는 [도로명주소]입니다."
                """
            elif "가장 저렴한" in query and "아파트" in query:
                translated_query = f"""
                {column_info}
                
                다음 작업을 수행해주세요:
                1. 거래금액을 기준으로 가장 저렴한 아파트를 찾아주세요.
                2. 찾은 결과에서 다음 정보를 보여주세요:
                   - 아파트 이름
                   - 거래금액 (만원 단위로 표시)
                   - 전용면적
                   - 층수
                   - 도로명주소
                3. 결과를 다음과 같은 형식으로 한글로 작성해주세요:
                   "가장 저렴한 아파트는 [아파트명]이며, 거래금액은 [금액]만원입니다.
                    전용면적은 [면적]㎡이고, [층]층에 위치해 있습니다.
                    주소는 [도로명주소]입니다."
                """
            elif "평균 거래가" in query:
                translated_query = f"""
                {column_info}
                
                다음 작업을 수행해주세요:
                1. 전체 아파트의 평균 거래금액을 계산해주세요.
                2. 결과를 다음과 같은 형식으로 한글로 작성해주세요:
                   "전체 아파트의 평균 거래금액은 [금액]만원입니다."
                """
            else:
                translated_query = f"{column_info}\n\n{query}\n\n결과는 한글로 작성해주세요."
            
            logging.info(f"원본 쿼리: {query}")
            logging.info(f"번역된 쿼리: {translated_query}")
            
            # Agent를 통한 쿼리 실행
            response = self.agent.invoke({"input": translated_query})
            result = response.get("output", "") if isinstance(response, dict) else str(response)
            
            logging.info(f"쿼리 결과: {result}")
            
            return {
                "result": result,
                "success": True
            }
        except Exception as e:
            error_msg = str(e)
            logging.error(f"쿼리 실행 중 오류 발생: {error_msg}")
            
            # 사용자 친화적인 에러 메시지 생성
            if "API key" in error_msg:
                error_msg = "OpenAI API 키가 유효하지 않거나 만료되었습니다."
            elif "rate limit" in error_msg.lower():
                error_msg = "OpenAI API 호출 한도를 초과했습니다. 잠시 후 다시 시도해주세요."
            elif "python repl" in error_msg.lower():
                error_msg = "Python REPL 도구 사용 권한이 필요합니다. 관리자에게 문의해주세요."
            
            return {
                "error": f"쿼리 실행 중 오류가 발생했습니다: {error_msg}",
                "success": False
            }
    
    def _translate_query(self, query: str) -> str:
        """간단한 한글 쿼리를 영어로 변환"""
        # 부동산 데이터 관련 번역 매핑 추가
        translations = {
            "평균": "average",
            "나이": "age",
            "성별": "gender",
            "분포": "distribution",
            "보여줘": "show",
            "분석": "analyze",
            "통계": "statistics",
            "최대": "maximum",
            "최소": "minimum",
            "개수": "count",
            "합계": "sum",
            "그래프": "plot",
            "거래가": "transaction price",
            "가장 비싼": "most expensive",
            "아파트": "apartment",
            "가격": "price",
            "면적": "area",
            "동": "district",
            "지역": "region"
        }
        
        # 번역 수행
        translated_query = query
        for kor, eng in translations.items():
            translated_query = translated_query.replace(kor, eng)
        
        # 특별한 쿼리 패턴 처리
        if "가장 비싼" in query:
            translated_query = "What is the apartment with the highest transaction price? Show the details."
            
        logging.info(f"원본 쿼리: {query}")
        logging.info(f"번역된 쿼리: {translated_query}")
        
        return translated_query

    def generate_search_examples(self):
        """데이터셋의 특성을 기반으로 검색 예시를 생성합니다."""
        examples = []
        
        try:
            # 지역 관련 예시
            if '시군구' in self.df.columns:
                district = self.df['시군구'].mode().iloc[0]
                examples.append(f"{district} 아파트 보여줘")
            
            # 가격 관련 예시
            if '거래금액' in self.df.columns:
                median_price = self.df['거래금액'].median()
                examples.append(f"거래금액이 {int(median_price):,} 이상인 아파트")
                examples.append("거래금액 높은 순으로 정렬")
            
            # 면적 관련 예시
            if '전용면적' in self.df.columns:
                examples.append("전용면적이 85 이상인 아파트")
            
            # 층수 관련 예시
            if '층' in self.df.columns:
                examples.append("20층 이상 아파트만 보기")
            
            # 아파트 이름 관련 예시
            if '아파트' in self.df.columns:
                apt_sample = self.df['아파트'].mode().iloc[0]
                examples.append(f"{apt_sample} 아파트 검색")
            
            # 건축년도 관련 예시
            if '건축년도' in self.df.columns:
                examples.append("2020년 이후 지어진 아파트")
            
            # 복합 조건 예시
            examples.append("전용면적 85 이상이고 20층 이상인 아파트")
            examples.append("강남구 아파트 중 거래금액 높은 순")
            
        except Exception as e:
            logging.error(f"검색 예시 생성 중 오류 발생: {str(e)}")
        
        # 예시가 없는 경우 기본 예시 반환
        return examples if examples else [
            "지역구 이름으로 검색 (예: 강남구 아파트)",
            "가격으로 검색 (예: 10억 이상 아파트)",
            "면적으로 검색 (예: 85㎡ 이상)",
            "층수로 검색 (예: 20층 이상)",
            "아파트 이름으로 검색 (예: 래미안)",
            "건축년도로 검색 (예: 2020년 이후)"
        ] 