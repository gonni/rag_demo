# -*- coding: utf-8 -*-
"""
RAG 최적화 시뮬레이션을 위한 설정 파일
"""

# 테스트할 Text Splitter 설정
SPLITTER_CONFIGS = [
    # RecursiveCharacterTextSplitter 설정들
    {
        "name": "recursive_500_100",
        "type": "recursive",
        "chunk_size": 500,
        "chunk_overlap": 100,
    },
    {
        "name": "recursive_1000_200", 
        "type": "recursive",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    },
    {
        "name": "recursive_1500_300",
        "type": "recursive", 
        "chunk_size": 1500,
        "chunk_overlap": 300,
    },
    {
        "name": "recursive_2000_400",
        "type": "recursive",
        "chunk_size": 2000, 
        "chunk_overlap": 400,
    },
    # 특별히 PNS 섹션을 위한 큰 청크
    {
        "name": "recursive_3000_500",
        "type": "recursive",
        "chunk_size": 3000,
        "chunk_overlap": 500,
    },
    # MarkdownHeaderTextSplitter 설정들
    {
        "name": "markdown_headers",
        "type": "markdown_headers",
        "headers_to_split_on": [
            ("#", "title"),
            ("##", "section"), 
            ("###", "subsection"),
            ("####", "subsubsection"),
        ]
    },
    # 하이브리드 접근: Markdown + Recursive
    {
        "name": "hybrid_markdown_recursive_1000",
        "type": "hybrid",
        "chunk_size": 1000,
        "chunk_overlap": 200,
    }
]

# 테스트할 Retriever 설정
RETRIEVER_CONFIGS = [
    # Similarity Search 설정들
    {
        "name": "similarity_k5",
        "search_type": "similarity",
        "k": 5,
    },
    {
        "name": "similarity_k10", 
        "search_type": "similarity",
        "k": 10,
    },
    {
        "name": "similarity_k15",
        "search_type": "similarity", 
        "k": 15,
    },
    {
        "name": "similarity_k20",
        "search_type": "similarity",
        "k": 20,
    },
    # MMR 설정들
    {
        "name": "mmr_k5_fetch15_lambda0.5",
        "search_type": "mmr",
        "k": 5,
        "fetch_k": 15,
        "lambda_mult": 0.5,
    },
    {
        "name": "mmr_k10_fetch30_lambda0.5",
        "search_type": "mmr", 
        "k": 10,
        "fetch_k": 30,
        "lambda_mult": 0.5,
    },
    {
        "name": "mmr_k10_fetch30_lambda0.7",
        "search_type": "mmr",
        "k": 10, 
        "fetch_k": 30,
        "lambda_mult": 0.7,
    },
    {
        "name": "mmr_k15_fetch50_lambda0.6",
        "search_type": "mmr",
        "k": 15,
        "fetch_k": 50, 
        "lambda_mult": 0.6,
    },
    {
        "name": "mmr_k20_fetch60_lambda0.5",
        "search_type": "mmr",
        "k": 20,
        "fetch_k": 60,
        "lambda_mult": 0.5,
    }
]

# 테스트할 임베딩 모델들
EMBEDDING_MODELS = [
    "exaone3.5:latest",
    "deepseek-coder:6.7b",
    "nomic-embed-text",
]

# 테스트 쿼리들
TEST_QUERIES = [
    "PNS의 purchaseState 값은 무엇이 있나요?",
    "purchaseState의 가능한 값들을 알려주세요",
    "PNS 메시지에서 purchaseState는 어떤 값을 가지나요?",
    "결제 상태 값 종류가 뭐가 있나요?",
    "COMPLETED와 CANCELED는 무엇을 의미하나요?",
]

# 목표 검색 내용 (반드시 검색되어야 하는 내용)
TARGET_CONTENT_KEYWORDS = [
    "purcahseState",  # 원문의 오타 포함
    "COMPLETED",
    "CANCELED", 
    "결제완료",
    "취소",
    "String",
]

# 목표 문서가 포함해야 하는 핵심 내용
TARGET_CONTENT_PATTERN = r"purcahseState.*String.*COMPLETED.*결제완료.*CANCELED.*취소"

# 파일 경로
SOURCE_FILE_PATH = "../data/dev_center_guide_allmd_touched.md"
RESULTS_DIR = "results"
MODELS_DIR = "models"