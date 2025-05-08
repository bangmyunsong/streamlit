# CSV 데이터 교체형 분석 시스템

## 개요

CSV 데이터 교체형 분석 시스템은 사용자가 `data/new.csv` 파일만 교체하면 다양한 데이터를 자동 분석할 수 있는 Python 및 Streamlit 기반의 데이터 분석 플랫폼입니다.

## 주요 기능

- CSV 파일 자동 로드 및 데이터 탐색
- 데이터 요약 및 시각화
- 기본형 에이전트형 질의 응답 (LangChain 기반, 선택사항)
- 사용자 정의 질문을 통한 인터랙티브 분석

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/yourusername/csv-analysis-system.git
cd csv-analysis-system
```

2. 가상환경 생성 및 활성화
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

3. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

## 사용 방법

1. `data/new.csv` 파일을 교체하거나 웹 인터페이스를 통해 CSV 파일 업로드

2. Streamlit 앱 실행
```bash
streamlit run app.py
```

3. 웹 브라우저에서 분석 시스템 열기 (기본: http://localhost:8501)

4. 다양한 분석 옵션 및 결과 확인

## 파일 및 디렉터리 구조

```
/project-root/
├── data/
│   └── new.csv                    # 🔁 교체 가능한 CSV 파일
│
├── app.py                         # Streamlit 실행 파일
├── analysis_core.py               # 📦 데이터 로드 및 시각화/분석 핵심 로직
├── requirements.txt               # 의존 라이브러리 목록
├── README.md                      # 문서
└── .env                           # 환경 변수 (OpenAI API Key 등)
```

## 기술 스택

- Python 3.9+
- Pandas / Matplotlib / Seaborn
- Streamlit
- scikit-learn
- LangChain (선택)
- OpenAI API (선택)

## 프로젝트 확장 아이디어

- CSV 형식이 아닌 Excel, Google Sheets 지원
- 추가 ML 기반 분석 기능 통합
- PDF 자동 보고서 생성
- 다국어 지원

## 라이센스

MIT License 