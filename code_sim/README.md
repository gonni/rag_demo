# RAG 최적화 시뮬레이션

이 프로젝트는 "PNS의 purchaseState 값은 무엇이 있나요?" 쿼리에 대해 특정 내용이 반드시 검색되도록 하는 최적의 RAG 설정을 찾기 위한 시뮬레이션입니다.

## 목표

다음 내용이 포함된 Document가 반드시 검색되도록 하는 것:
```
| purcahseState      | String        | COMPLETED : 결제완료 / CANCELED : 취소
```

## 파일 구조

```
code_sim/
├── config.py          # 시뮬레이션 설정
├── text_splitters.py  # 다양한 텍스트 분할 방법
├── simulator.py       # 시뮬레이션 실행기
├── main.py           # 메인 실행 파일
├── requirements.txt  # 필요한 패키지
├── README.md         # 이 파일
├── results/          # 결과 파일들
└── models/           # 벡터 DB 모델들
```

## 사용법

### 1. 환경 설정

```bash
# 필요한 패키지 설치
pip install -r requirements.txt

# Ollama 모델 설치 (필요한 경우)
ollama pull exaone3.5:latest
ollama pull deepseek-coder:6.7b
ollama pull nomic-embed-text
```

### 2. 시뮬레이션 실행

```bash
# 빠른 테스트 (일부 설정만)
python main.py quick

# 전체 테스트 (모든 조합)
python main.py full

# 기존 결과 분석
python main.py analyze <results_file.json>
```

## 테스트 조합

### Text Splitters
- **RecursiveCharacterTextSplitter**: 다양한 chunk_size (500~3000)
- **MarkdownHeaderTextSplitter**: 헤더 기반 분할
- **Hybrid**: 헤더 + 재분할 조합

### Retrievers
- **Similarity Search**: k=5~20
- **MMR**: 다양한 k, fetch_k, lambda_mult 조합

### Embedding Models
- exaone3.5:latest
- deepseek-coder:6.7b  
- nomic-embed-text

## 평가 기준

1. **성공률**: 목표 내용이 검색된 쿼리 비율
2. **평균 위치**: 목표 내용이 검색 결과에서 나타나는 평균 순위
3. **검색 시간**: 검색 수행 시간

## 결과 분석

시뮬레이션 결과는 다음 정보를 포함합니다:

- 최고 성능 조합 (성공률/위치 기준)
- 임베딩 모델별 성능 비교
- 스플리터별 성능 비교
- 리트리버별 성능 비교

## 예상 결과

최적의 조합은 다음과 같을 것으로 예상됩니다:
- **큰 chunk_size** (2000~3000): PNS 섹션 전체를 하나의 청크로 유지
- **적절한 overlap** (300~500): 컨텍스트 유지
- **Similarity search**: 정확한 매칭
- **충분한 k값** (10~15): 다양한 결과 확보

## 주의사항

- Ollama 모델이 설치되어 있어야 합니다
- 시뮬레이션은 시간이 오래 걸릴 수 있습니다 (전체 테스트 시 1-2시간)
- 메모리 사용량이 클 수 있습니다