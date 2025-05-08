import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
from analysis_core import CSVAnalyzer, FONT_FAMILY
import tempfile
import seaborn as sns
import platform
import matplotlib.font_manager as fm

# 페이지 설정
st.set_page_config(
    page_title="CSV 데이터 교체형 분석 시스템",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS로 폰트 설정
st.markdown(f"""
<style>
    * {{
        font-family: {FONT_FAMILY}, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif !important;
    }}
    
    .stMarkdown, .stText, .stTitle, .stHeader, .stSelectbox, .stMultiselect {{
        font-family: {FONT_FAMILY}, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen-Sans, Ubuntu, Cantarell, 'Helvetica Neue', sans-serif !important;
    }}
    
    .main-header {{
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E90FF;
        margin-bottom: 1rem;
    }}
    .sub-header {{
        font-size: 1.8rem;
        font-weight: bold;
        color: #4682B4;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }}
    .card {{
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
    }}
    .info-text {{
        font-size: 1.1rem;
        color: #666;
    }}
</style>
""", unsafe_allow_html=True)

# 제목
st.title("📊 CSV 데이터 분석 시스템")

# 사이드바 제목
st.sidebar.title("📊 CSV 데이터 분석 시스템")

# 메인 함수
def main():
    # 세션 상태 초기화
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
        
    if 'openai_api_key' not in st.session_state:
        st.session_state.openai_api_key = os.getenv("OPENAI_API_KEY", "")

    # 현재 로드된 파일명 표시
    if st.session_state.data_loaded and st.session_state.analyzer is not None:
        file_name = os.path.basename(st.session_state.analyzer.csv_path)
        st.markdown(f'<div class="main-header">CSV 데이터 교체형 분석 시스템 - {file_name}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-header">CSV 데이터 교체형 분석 시스템</div>', unsafe_allow_html=True)
    
    # 사이드바: CSV 파일 업로드 또는 기본 파일 사용
    with st.sidebar:
        st.markdown("## 1. 데이터 로드")
        
        # 초기화 버튼
        if st.button("🔄 초기화", type="primary"):
            # 세션 상태 초기화
            for key in st.session_state.keys():
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        
        option = st.radio(
            "데이터 소스 선택:",
            ["기본 CSV 파일 사용", "CSV 파일 업로드"]
        )
        
        if option == "기본 CSV 파일 사용":
            csv_path = "data/apart_real_estate_20250404.csv"
            
            if not os.path.exists(csv_path):
                st.error("기본 CSV 파일(data/apart_real_estate_20250404.csv)을 찾을 수 없습니다. CSV 파일을 업로드하세요.")
                csv_path = None
            else:
                st.success(f"기본 CSV 파일을 사용합니다: {csv_path}")
                
                # 데이터 로드 버튼
                if st.button("데이터 분석 시작", key="load_default"):
                    with st.spinner("데이터 분석 중..."):
                        st.session_state.analyzer = CSVAnalyzer(csv_path)
                        st.session_state.data_loaded = True
                        
        else:  # CSV 파일 업로드
            uploaded_file = st.file_uploader("CSV 파일 선택", type=["csv"])
            
            if uploaded_file is not None:
                # 임시 파일로 저장
                with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                st.success("CSV 파일이 업로드되었습니다!")
                
                # 데이터 로드 버튼
                if st.button("데이터 분석 시작", key="load_uploaded"):
                    with st.spinner("데이터 분석 중..."):
                        st.session_state.analyzer = CSVAnalyzer(tmp_path)
                        st.session_state.data_loaded = True
        
        # 분석 옵션 (데이터가 로드된 경우에만)
        if st.session_state.data_loaded:
            st.markdown("## 2. 분석 옵션")
            analysis_sections = st.multiselect(
                "표시할 분석 섹션 선택:",
                ["기본 정보", "요약 통계", "데이터 시각화", "클러스터링", "데이터 검색"],
                default=["기본 정보", "요약 통계", "데이터 시각화"],
                key="analysis_options_sidebar"
            )
            
            # 테마 선택
            st.markdown("## 3. 시각화 설정")
            viz_theme = st.selectbox(
                "시각화 테마:",
                ["기본", "밝은계열", "어두운계열", "미니멀"]
            )
            
            # 테마 적용
            if viz_theme == "어두운계열":
                plt.style.use('default')
                plt.rcParams.update({
                    'axes.facecolor': '#2b2b2b',
                    'figure.facecolor': '#2b2b2b',
                    'text.color': 'white',
                    'axes.labelcolor': 'white',
                    'xtick.color': 'white',
                    'ytick.color': 'white',
                    'grid.alpha': 0.2,
                    'axes.prop_cycle': plt.cycler(color=['#00ff00', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                })
            elif viz_theme == "밝은계열":
                plt.style.use('default')
                plt.rcParams.update({
                    'axes.facecolor': '#f0f2f6',
                    'figure.facecolor': 'white',
                    'axes.grid': True,
                    'grid.alpha': 0.2,
                    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                })
            elif viz_theme == "미니멀":
                plt.style.use('default')
                plt.rcParams.update({
                    'axes.facecolor': 'white',
                    'figure.facecolor': 'white',
                    'axes.grid': False,
                    'axes.spines.top': False,
                    'axes.spines.right': False,
                    'axes.prop_cycle': plt.cycler(color=['#4A90E2', '#50E3C2', '#F5A623', '#D0021B', '#9013FE'])
                })
            else:  # 기본 테마
                plt.style.use('default')
                plt.rcParams.update({
                    'axes.facecolor': 'white',
                    'figure.facecolor': 'white',
                    'axes.grid': True,
                    'grid.alpha': 0.3,
                    'axes.prop_cycle': plt.cycler(color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'])
                })
    
    # 메인 컨텐츠: 데이터가 로드된 경우에만 표시
    if st.session_state.data_loaded and st.session_state.analyzer is not None:
        analyzer = st.session_state.analyzer
        
        # 선택된 분석 섹션
        sections = st.sidebar.multiselect(
            "표시할 분석 섹션 선택:",
            ["기본 정보", "요약 통계", "데이터 시각화", "클러스터링", "데이터 검색"],
            default=["기본 정보", "요약 통계", "데이터 시각화", "데이터 검색"]
        )

        # 데이터 검색 섹션을 먼저 표시
        if "데이터 검색" in sections:
            st.markdown('<div class="sub-header">🔎 데이터 검색 및 분석</div>', unsafe_allow_html=True)
            
            # 탭 생성
            search_tab, query_tab = st.tabs(["기본 검색", "자연어 분석"])
            
            # 기본 검색 탭
            with search_tab:
                # 데이터 기반 검색 예시 생성
                search_examples = analyzer.generate_search_examples()
                example_text = "\n".join([f"- {example}" for example in search_examples])
                
                st.markdown(f"""
                ### 🔍 기본 검색
                
                기본 검색어로 데이터를 검색해보세요. 예:
                {example_text}
                """)
                
                # 검색 입력 필드와 버튼을 나란히 배치
                col1, col2 = st.columns([3, 1])
                with col1:
                    basic_query = st.text_input(
                        "검색어 입력:",
                        key="basic_search",
                        placeholder="예: 나이가 30 이상인 데이터"
                    )
                with col2:
                    search_clicked = st.button("검색하기", type="primary")
                
                # 검색 실행
                if basic_query:  # 검색어가 있을 때
                    if search_clicked:  # 검색 버튼을 클릭했을 때
                        st.session_state.search_results = analyzer.search_data(basic_query)
                        st.session_state.search_performed = True
                    
                    # 검색 결과 표시 (세션 상태에 저장된 결과 사용)
                    if 'search_performed' in st.session_state and st.session_state.search_performed:
                        results = st.session_state.search_results
                        if not results.empty:
                            st.success(f"✨ 검색 결과: {len(results)}개의 행이 일치합니다.")
                            st.dataframe(results, use_container_width=True)
                        else:
                            st.info("💡 검색 결과가 없습니다.")
            
            # 자연어 질의 탭
            with query_tab:
                st.markdown("""
                ### 💡 자연어로 데이터 분석하기
                
                CSV 데이터에 대해 자연어로 질문해보세요. 예시:
                - "성별에 따른 생존율이 어떻게 되나요?"
                - "평균 나이는 얼마인가요?"
                - "가장 많은 승객이 탑승한 항구는 어디인가요?"
                - "1등석 승객의 평균 요금은 얼마인가요?"
                - "나이대별 생존율을 그래프로 보여주세요"
                """)
                
                # OpenAI API 키 입력 (세션 상태로 관리)
                if not st.session_state.openai_api_key:
                    api_key = st.text_input(
                        "OpenAI API 키를 입력하세요:",
                        type="password",
                        help="분석을 위해 OpenAI API 키가 필요합니다. API 키는 안전하게 저장되며 세션에서만 사용됩니다."
                    )
                    if api_key:
                        st.session_state.openai_api_key = api_key
                        os.environ["OPENAI_API_KEY"] = api_key
                        # Agent 재초기화
                        analyzer._initialize_agent()
                        st.success("✅ API 키가 설정되었습니다!")
                
                # 구분선 추가
                st.markdown("---")
                
                # 질의 입력 필드
                col1, col2 = st.columns([3, 1])
                with col1:
                    query = st.text_input(
                        "질문을 입력하세요:",
                        key="nl_query",
                        placeholder="예: 성별에 따른 생존율이 어떻게 되나요?"
                    )
                with col2:
                    analyze_button = st.button("분석하기", type="primary")
                
                if query and analyze_button:
                    if not st.session_state.openai_api_key:
                        st.error("❗ OpenAI API 키를 먼저 입력해주세요.")
                    else:
                        with st.spinner("🔄 데이터 분석 중..."):
                            try:
                                result = analyzer.query_data(query)
                                if result.get("success", False):
                                    st.markdown("#### ✨ 분석 결과")
                                    st.write(result["result"])
                                else:
                                    st.error(result.get("error", "분석 중 오류가 발생했습니다."))
                            except Exception as e:
                                st.error(f"❌ 오류가 발생했습니다: {str(e)}")

        # 나머지 섹션들...
        if "기본 정보" in sections:
            st.markdown('<div class="sub-header">📋 기본 정보</div>', unsafe_allow_html=True)
            
            info = analyzer.get_basic_info()
            
            if info:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 데이터셋 개요")
                    st.markdown(f"- **행 개수:** {info['shape'][0]:,}")
                    st.markdown(f"- **열 개수:** {info['shape'][1]}")
                    st.markdown(f"- **수치형 컬럼:** {len(info['numeric_columns'])}")
                    st.markdown(f"- **범주형 컬럼:** {len(info['categorical_columns'])}")
                    st.markdown(f"- **날짜형 컬럼:** {len(info['date_columns'])}")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 결측치 정보")
                    
                    missing_df = pd.DataFrame({
                        '컬럼명': list(info['missing_values'].keys()),
                        '결측치 수': list(info['missing_values'].values())
                    })
                    missing_df = missing_df.sort_values('결측치 수', ascending=False).reset_index(drop=True)
                    missing_df = missing_df[missing_df['결측치 수'] > 0]
                    
                    if len(missing_df) > 0:
                        st.dataframe(missing_df, use_container_width=True)
                    else:
                        st.markdown("결측치가 없습니다! 👍")
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # 데이터 미리보기
                st.markdown("#### 데이터 미리보기")
                if analyzer.df is not None:
                    st.dataframe(analyzer.df.head(10), use_container_width=True)
        
        if "요약 통계" in sections:
            st.markdown('<div class="sub-header">📊 요약 통계</div>', unsafe_allow_html=True)
            
            stats = analyzer.get_summary_statistics()
            
            if stats:
                # 탭 생성
                tab1, tab2, tab3 = st.tabs(["수치형 변수", "범주형 변수", "날짜형 변수"])
                
                # 수치형 변수 통계
                with tab1:
                    if stats['numeric']:
                        # 데이터프레임으로 변환
                        numeric_stats = pd.DataFrame(stats['numeric'])
                        
                        st.markdown("##### 수치형 변수 통계 요약")
                        st.dataframe(numeric_stats.T, use_container_width=True)
                    else:
                        st.info("수치형 변수가 없습니다.")
                
                # 범주형 변수 통계
                with tab2:
                    if stats['categorical']:
                        st.markdown("##### 범주형 변수 통계 요약")
                        
                        for col, col_stats in stats['categorical'].items():
                            with st.expander(f"{col} (고유값: {col_stats['unique_count']})"):
                                if col_stats['top_values']:
                                    top_values_df = pd.DataFrame({
                                        '값': list(col_stats['top_values'].keys()),
                                        '빈도': list(col_stats['top_values'].values())
                                    })
                                    st.dataframe(top_values_df, use_container_width=True)
                                else:
                                    st.info("데이터가 없습니다.")
                    else:
                        st.info("범주형 변수가 없습니다.")
                
                # 날짜형 변수 통계
                with tab3:
                    if stats['date']:
                        st.markdown("##### 날짜형 변수 통계 요약")
                        
                        for col, col_stats in stats['date'].items():
                            st.markdown(f"**{col}**")
                            st.markdown(f"- 시작일: {col_stats['min']}")
                            st.markdown(f"- 종료일: {col_stats['max']}")
                            st.markdown(f"- 기간(일): {col_stats['range_days']}")
                    else:
                        st.info("날짜형 변수가 없습니다.")
        
        if "데이터 시각화" in sections:
            st.markdown('<div class="sub-header">📈 데이터 시각화</div>', unsafe_allow_html=True)
            
            visualizations = analyzer.generate_visualizations()
            
            if visualizations:
                # 시각화 탭 생성
                vis_tabs = []
                
                # 히스토그램 탭
                hist_plots = [k for k in visualizations.keys() if k.endswith('_hist')]
                if hist_plots:
                    vis_tabs.append("히스토그램")
                
                # 막대 그래프 탭
                bar_plots = [k for k in visualizations.keys() if k.endswith('_bar')]
                if bar_plots:
                    vis_tabs.append("막대 그래프")
                
                # 상관관계 히트맵 탭
                if "correlation_heatmap" in visualizations:
                    vis_tabs.append("상관관계 히트맵")
                
                # 시계열 그래프 탭
                if "time_series" in visualizations:
                    vis_tabs.append("시계열 그래프")
                
                # 탭 생성
                if vis_tabs:
                    tabs = st.tabs(vis_tabs)
                    
                    tab_idx = 0
                    
                    # 히스토그램 탭 내용
                    if "히스토그램" in vis_tabs:
                        with tabs[tab_idx]:
                            st.markdown("##### 수치형 변수 분포")
                            
                            # 2개의 열로 표시
                            cols = st.columns(2)
                            col_idx = 0
                            
                            for key in hist_plots:
                                with cols[col_idx % 2]:
                                    st.pyplot(visualizations[key])
                                col_idx += 1
                        
                        tab_idx += 1
                    
                    # 막대 그래프 탭 내용
                    if "막대 그래프" in vis_tabs:
                        with tabs[tab_idx]:
                            st.markdown("##### 범주형 변수 빈도")
                            
                            # 2개의 열로 표시
                            cols = st.columns(2)
                            col_idx = 0
                            
                            for key in bar_plots:
                                with cols[col_idx % 2]:
                                    st.pyplot(visualizations[key])
                                col_idx += 1
                        
                        tab_idx += 1
                    
                    # 상관관계 히트맵 탭 내용
                    if "상관관계 히트맵" in vis_tabs:
                        with tabs[tab_idx]:
                            st.markdown("##### 수치형 변수 간 상관관계")
                            st.pyplot(visualizations["correlation_heatmap"])
                        
                        tab_idx += 1
                    
                    # 시계열 그래프 탭 내용
                    if "시계열 그래프" in vis_tabs:
                        with tabs[tab_idx]:
                            st.markdown("##### 시계열 추세")
                            st.pyplot(visualizations["time_series"])
                else:
                    st.info("시각화 가능한 데이터가 없습니다.")
            else:
                st.info("시각화 가능한 데이터가 없습니다.")
        
        if "클러스터링" in sections:
            st.markdown('<div class="sub-header">🔍 데이터 클러스터링</div>', unsafe_allow_html=True)
            
            if len(analyzer.numeric_cols) >= 2:
                st.markdown("K-means 클러스터링으로 데이터 그룹 분석")
                
                # 클러스터 수 선택
                n_clusters = st.slider("클러스터 수 선택:", min_value=2, max_value=10, value=3)
                
                if st.button("클러스터링 실행"):
                    with st.spinner("클러스터링 분석 중..."):
                        clustering_results = analyzer.perform_clustering(n_clusters=n_clusters)
                        
                        if clustering_results and "plot" in clustering_results:
                            col1, col2 = st.columns([2, 1])
                            
                            with col1:
                                st.pyplot(clustering_results["plot"])
                            
                            with col2:
                                st.markdown("#### 클러스터별 특성")
                                
                                if "cluster_stats" in clustering_results:
                                    # 클러스터별 통계를 데이터프레임으로 변환
                                    cluster_stats_df = pd.DataFrame(clustering_results["cluster_stats"])
                                    st.dataframe(cluster_stats_df)
                        else:
                            st.warning("클러스터링을 수행할 수 없습니다. 충분한 수치형 데이터가 없습니다.")
            else:
                st.warning("클러스터링을 수행하려면 최소 2개 이상의 수치형 변수가 필요합니다.")
    
    # 데이터가 로드되지 않은 경우
    else:
        st.markdown('<div class="card info-text">', unsafe_allow_html=True)
        st.markdown("""
        👈 사이드바에서 CSV 파일을 선택하고 '데이터 분석 시작' 버튼을 클릭하세요.
        
        이 애플리케이션은 CSV 파일을 자동으로 분석하여 다음과 같은 정보를 제공합니다:
        - 기본 정보: 행/열 개수, 데이터 타입, 결측치 등
        - 요약 통계: 수치형/범주형/날짜형 변수의 통계 요약
        - 데이터 시각화: 히스토그램, 막대 그래프, 상관관계 히트맵 등
        - 클러스터링: K-means 알고리즘을 통한 데이터 그룹화
        - 데이터 검색: 자연어 쿼리를 통한 데이터 검색
        """)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # 가이드와 샘플 데이터
        st.markdown('<div class="sub-header">🚀 시작하기</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### 사용 방법")
            st.markdown("""
            1. data/new.csv 파일을 교체하거나, 새 CSV 파일을 업로드하세요.
            2. '데이터 분석 시작' 버튼을 클릭하세요.
            3. 자동으로 생성된 시각화와 분석 결과를 확인하세요.
            4. 필요한 경우 사이드바에서 추가 분석 옵션을 선택하세요.
            """)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("#### CSV 파일 요구사항")
            st.markdown("""
            - 첫 번째 행은 반드시 헤더(컬럼명)여야 합니다.
            - 결측치는 빈 셀이나 'NA', 'NULL' 등으로 표시될 수 있습니다.
            - 날짜 형식은 자동으로 인식되지만, 표준 형식이 좋습니다(YYYY-MM-DD).
            - 큰 파일의 경우 처리 시간이 더 오래 걸릴 수 있습니다.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main() 