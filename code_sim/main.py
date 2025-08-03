# -*- coding: utf-8 -*-
"""
RAG 최적화 시뮬레이션 메인 실행 파일
"""

import sys
import os

# 현재 디렉토리를 Python path에 추가
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(current_dir)
sys.path.append(parent_dir)

from simulator import RAGSimulator
from config import SOURCE_FILE_PATH, TEST_QUERIES


def quick_test():
    """빠른 테스트를 위한 함수 (일부 설정만 테스트)"""
    print("🔥 빠른 테스트 모드")
    
    # 테스트할 설정 축소
    quick_splitter_configs = [
        {
            "name": "recursive_1000_200",
            "type": "recursive", 
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        {
            "name": "recursive_2000_400",
            "type": "recursive",
            "chunk_size": 2000,
            "chunk_overlap": 400,
        },
        {
            "name": "markdown_headers",
            "type": "markdown_headers",
            "headers_to_split_on": [
                ("#", "title"),
                ("##", "section"),
                ("###", "subsection"),
            ]
        }
    ]
    
    quick_retriever_configs = [
        {
            "name": "similarity_k10",
            "search_type": "similarity",
            "k": 10,
        },
        {
            "name": "mmr_k10_fetch30_lambda0.7",
            "search_type": "mmr",
            "k": 10,
            "fetch_k": 30,
            "lambda_mult": 0.7,
        }
    ]
    
    quick_embedding_models = ["exaone3.5:latest"]
    
    # 임시로 설정 덮어쓰기
    import config
    config.SPLITTER_CONFIGS = quick_splitter_configs
    config.RETRIEVER_CONFIGS = quick_retriever_configs
    config.EMBEDDING_MODELS = quick_embedding_models
    
    simulator = RAGSimulator()
    results = simulator.run_all_experiments(SOURCE_FILE_PATH)
    
    # 결과 저장 및 분석
    results_file = simulator.save_results(results, "quick_test_results.json")
    analysis = simulator.analyze_results(results)
    simulator.print_analysis(analysis)
    
    return results, analysis


def full_test():
    """전체 테스트 실행"""
    print("🚀 전체 테스트 모드")
    
    simulator = RAGSimulator()
    results = simulator.run_all_experiments(SOURCE_FILE_PATH)
    
    # 결과 저장 및 분석
    results_file = simulator.save_results(results)
    analysis = simulator.analyze_results(results)
    simulator.print_analysis(analysis)
    
    return results, analysis


def analyze_existing_results(results_file: str):
    """기존 결과 파일 분석"""
    import json
    
    print(f"📊 기존 결과 분석: {results_file}")
    
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    simulator = RAGSimulator()
    analysis = simulator.analyze_results(results)
    simulator.print_analysis(analysis)
    
    return analysis


def main():
    """메인 함수"""
    print("="*80)
    print("🔍 RAG 최적화 시뮬레이션")
    print("목표: 'PNS의 purchaseState 값은 무엇이 있나요?' 검색 최적화")
    print("="*80)
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == "quick":
            results, analysis = quick_test()
        elif command == "full":
            results, analysis = full_test()
        elif command == "analyze" and len(sys.argv) > 2:
            results_file = sys.argv[2]
            analysis = analyze_existing_results(results_file)
        else:
            print("사용법:")
            print("  python main.py quick     # 빠른 테스트")
            print("  python main.py full      # 전체 테스트")
            print("  python main.py analyze <results_file>  # 기존 결과 분석")
            return
    else:
        # 기본값: 빠른 테스트
        results, analysis = quick_test()
    
    # 최적 설정 추천
    if "best_by_success_rate" in analysis and analysis["best_by_success_rate"]:
        best = analysis["best_by_success_rate"][0]
        print("\n" + "🎯 최적 설정 추천" + "="*60)
        print(f"실험명: {best['experiment_name']}")
        print(f"성공률: {best['summary']['success_rate']:.2%}")
        print(f"평균 목표 위치: {best['summary']['avg_target_position']:.1f}")
        print("\n설정 세부사항:")
        print(f"  스플리터: {best['splitter_config']}")
        print(f"  리트리버: {best['retriever_config']}")
        print(f"  임베딩 모델: {best['embedding_model']}")
        print("="*80)


if __name__ == "__main__":
    main()