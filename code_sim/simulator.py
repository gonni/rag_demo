# -*- coding: utf-8 -*-
"""
RAG 최적화 시뮬레이션 실행기
"""

import os
import json
import time
import re
from typing import List, Dict, Any
from datetime import datetime

from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document

from config import *
from text_splitters import *


class RAGSimulator:
    def __init__(self):
        self.results = []
        self.current_embedding_model = None
        self.ensure_directories()
    
    def ensure_directories(self):
        """필요한 디렉토리들을 생성합니다."""
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(MODELS_DIR, exist_ok=True)
    
    def create_vector_db(self, docs: List[Document], embedding_model: str, config_name: str) -> FAISS:
        """문서들로부터 벡터 DB를 생성합니다."""
        embeddings = OllamaEmbeddings(model=embedding_model)
        
        print(f"Creating vector DB with {len(docs)} documents using {embedding_model}...")
        db = FAISS.from_documents(docs, embeddings)
        
        # 벡터 DB 저장
        db_path = os.path.join(MODELS_DIR, f"{config_name}_{embedding_model.replace(':', '_')}")
        db.save_local(db_path)
        print(f"Vector DB saved to {db_path}")
        
        return db
    
    def create_retriever(self, db: FAISS, retriever_config: dict):
        """설정에 따라 retriever를 생성합니다."""
        search_type = retriever_config["search_type"]
        
        if search_type == "similarity":
            return db.as_retriever(
                search_type="similarity",
                search_kwargs={"k": retriever_config["k"]}
            )
        elif search_type == "mmr":
            return db.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": retriever_config["k"],
                    "fetch_k": retriever_config["fetch_k"],
                    "lambda_mult": retriever_config["lambda_mult"]
                }
            )
        else:
            raise ValueError(f"Unknown search type: {search_type}")
    
    def evaluate_retrieval(self, retriever, query: str, target_content_docs: List[int]) -> dict:
        """검색 성능을 평가합니다."""
        start_time = time.time()
        
        try:
            retrieved_docs = retriever.invoke(query)
            retrieval_time = time.time() - start_time
            
            # 검색된 문서들 분석
            retrieved_indices = []
            target_found = False
            target_position = -1
            
            for i, doc in enumerate(retrieved_docs):
                doc_index = doc.metadata.get("chunk_idx", -1)
                retrieved_indices.append(doc_index)
                
                # 목표 내용이 포함되어 있는지 확인
                if doc.metadata.get("has_target_content", False):
                    target_found = True
                    if target_position == -1:
                        target_position = i + 1  # 1-based position
            
            # 목표 문서들이 검색되었는지 확인 (backup check)
            target_docs_retrieved = len(set(retrieved_indices) & set(target_content_docs))
            
            return {
                "success": True,
                "query": query,
                "total_retrieved": len(retrieved_docs),
                "target_found": target_found,
                "target_position": target_position,
                "target_docs_retrieved": target_docs_retrieved,
                "retrieval_time": retrieval_time,
                "retrieved_indices": retrieved_indices[:5],  # 상위 5개만 저장
                "error": None
            }
        
        except Exception as e:
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "retrieval_time": time.time() - start_time
            }
    
    def run_single_experiment(self, 
                            splitter_config: dict, 
                            retriever_config: dict, 
                            embedding_model: str,
                            source_text: str) -> dict:
        """단일 실험을 실행합니다."""
        
        experiment_name = f"{splitter_config['name']}_{retriever_config['name']}_{embedding_model.replace(':', '_')}"
        print(f"\n🧪 Running experiment: {experiment_name}")
        
        # 1. 문서 분할
        print("1. Splitting documents...")
        splitter_func = get_splitter_function(splitter_config)
        docs = splitter_func(source_text)
        
        # 2. 목표 내용 분석
        analysis = analyze_target_content_coverage(docs)
        target_content_docs = [doc["index"] for doc in analysis["docs_with_target_content"]]
        
        print(f"   - Total documents: {len(docs)}")
        print(f"   - Documents with target content: {len(target_content_docs)}")
        print(f"   - Documents with purchaseState: {len(analysis['docs_with_purchase_state'])}")
        
        if len(target_content_docs) == 0:
            print("   ⚠️ No documents contain target content!")
            return {
                "experiment_name": experiment_name,
                "splitter_config": splitter_config,
                "retriever_config": retriever_config,
                "embedding_model": embedding_model,
                "success": False,
                "error": "No target content found in split documents",
                "analysis": analysis
            }
        
        # 3. 벡터 DB 생성
        print("2. Creating vector database...")
        try:
            db = self.create_vector_db(docs, embedding_model, experiment_name)
        except Exception as e:
            print(f"   ❌ Failed to create vector DB: {e}")
            return {
                "experiment_name": experiment_name,
                "splitter_config": splitter_config,
                "retriever_config": retriever_config,
                "embedding_model": embedding_model,
                "success": False,
                "error": f"Vector DB creation failed: {str(e)}",
                "analysis": analysis
            }
        
        # 4. Retriever 생성
        print("3. Creating retriever...")
        retriever = self.create_retriever(db, retriever_config)
        
        # 5. 쿼리 테스트
        print("4. Testing queries...")
        query_results = []
        for query in TEST_QUERIES:
            result = self.evaluate_retrieval(retriever, query, target_content_docs)
            query_results.append(result)
            
            if result["success"] and result["target_found"]:
                print(f"   ✅ '{query}' -> Target found at position {result['target_position']}")
            elif result["success"]:
                print(f"   ❌ '{query}' -> Target not found")
            else:
                print(f"   💥 '{query}' -> Error: {result['error']}")
        
        # 6. 결과 정리
        successful_queries = [r for r in query_results if r["success"] and r["target_found"]]
        avg_target_position = sum(r["target_position"] for r in successful_queries) / len(successful_queries) if successful_queries else float('inf')
        success_rate = len(successful_queries) / len(TEST_QUERIES)
        
        result = {
            "experiment_name": experiment_name,
            "splitter_config": splitter_config,
            "retriever_config": retriever_config,
            "embedding_model": embedding_model,
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "query_results": query_results,
            "summary": {
                "total_queries": len(TEST_QUERIES),
                "successful_queries": len(successful_queries),
                "success_rate": success_rate,
                "avg_target_position": avg_target_position,
                "total_documents": len(docs),
                "target_content_documents": len(target_content_docs)
            }
        }
        
        print(f"   📊 Success rate: {success_rate:.2%} ({len(successful_queries)}/{len(TEST_QUERIES)})")
        print(f"   🎯 Avg target position: {avg_target_position:.1f}")
        
        return result
    
    def run_all_experiments(self, source_file_path: str) -> List[dict]:
        """모든 실험을 실행합니다."""
        print(f"🚀 Starting RAG optimization simulation")
        print(f"📁 Source file: {source_file_path}")
        
        # 소스 텍스트 로드
        source_text = load_markdown_file(source_file_path)
        print(f"📄 Source text length: {len(source_text)} characters")
        
        all_results = []
        total_experiments = len(SPLITTER_CONFIGS) * len(RETRIEVER_CONFIGS) * len(EMBEDDING_MODELS)
        current_experiment = 0
        
        for embedding_model in EMBEDDING_MODELS:
            print(f"\n🤖 Testing embedding model: {embedding_model}")
            
            for splitter_config in SPLITTER_CONFIGS:
                for retriever_config in RETRIEVER_CONFIGS:
                    current_experiment += 1
                    print(f"\n[{current_experiment}/{total_experiments}]", end=" ")
                    
                    try:
                        result = self.run_single_experiment(
                            splitter_config, 
                            retriever_config, 
                            embedding_model, 
                            source_text
                        )
                        all_results.append(result)
                        
                    except Exception as e:
                        print(f"💥 Experiment failed: {e}")
                        error_result = {
                            "experiment_name": f"{splitter_config['name']}_{retriever_config['name']}_{embedding_model.replace(':', '_')}",
                            "splitter_config": splitter_config,
                            "retriever_config": retriever_config,
                            "embedding_model": embedding_model,
                            "success": False,
                            "error": str(e),
                            "timestamp": datetime.now().isoformat()
                        }
                        all_results.append(error_result)
        
        return all_results
    
    def save_results(self, results: List[dict], filename: str = None):
        """결과를 파일로 저장합니다."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_simulation_results_{timestamp}.json"
        
        filepath = os.path.join(RESULTS_DIR, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Results saved to: {filepath}")
        return filepath
    
    def analyze_results(self, results: List[dict]) -> dict:
        """결과를 분석하고 최적의 설정을 찾습니다."""
        successful_results = [r for r in results if r.get("success", False) and "summary" in r]
        
        if not successful_results:
            return {"error": "No successful experiments found"}
        
        # 성공률 기준으로 정렬
        sorted_by_success = sorted(
            successful_results, 
            key=lambda x: (x["summary"]["success_rate"], -x["summary"]["avg_target_position"]),
            reverse=True
        )
        
        # 평균 위치 기준으로 정렬 (성공한 것들만)
        successful_with_target = [r for r in successful_results if r["summary"]["success_rate"] > 0]
        sorted_by_position = sorted(
            successful_with_target,
            key=lambda x: x["summary"]["avg_target_position"]
        )
        
        analysis = {
            "total_experiments": len(results),
            "successful_experiments": len(successful_results),
            "experiments_with_target_found": len(successful_with_target),
            "best_by_success_rate": sorted_by_success[:3],
            "best_by_target_position": sorted_by_position[:3],
            "embedding_model_performance": {},
            "splitter_performance": {},
            "retriever_performance": {}
        }
        
        # 임베딩 모델별 성능
        for model in EMBEDDING_MODELS:
            model_results = [r for r in successful_results if r["embedding_model"] == model]
            if model_results:
                avg_success_rate = sum(r["summary"]["success_rate"] for r in model_results) / len(model_results)
                analysis["embedding_model_performance"][model] = {
                    "experiments": len(model_results),
                    "avg_success_rate": avg_success_rate
                }
        
        # 스플리터별 성능 
        for config in SPLITTER_CONFIGS:
            splitter_results = [r for r in successful_results if r["splitter_config"]["name"] == config["name"]]
            if splitter_results:
                avg_success_rate = sum(r["summary"]["success_rate"] for r in splitter_results) / len(splitter_results)
                analysis["splitter_performance"][config["name"]] = {
                    "experiments": len(splitter_results),
                    "avg_success_rate": avg_success_rate
                }
        
        # 리트리버별 성능
        for config in RETRIEVER_CONFIGS:
            retriever_results = [r for r in successful_results if r["retriever_config"]["name"] == config["name"]]
            if retriever_results:
                avg_success_rate = sum(r["summary"]["success_rate"] for r in retriever_results) / len(retriever_results)
                analysis["retriever_performance"][config["name"]] = {
                    "experiments": len(retriever_results),
                    "avg_success_rate": avg_success_rate
                }
        
        return analysis
    
    def print_analysis(self, analysis: dict):
        """분석 결과를 출력합니다."""
        print("\n" + "="*80)
        print("📊 RAG 최적화 시뮬레이션 결과 분석")
        print("="*80)
        
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        print(f"총 실험 수: {analysis['total_experiments']}")
        print(f"성공한 실험 수: {analysis['successful_experiments']}")
        print(f"목표 내용을 찾은 실험 수: {analysis['experiments_with_target_found']}")
        
        print(f"\n🏆 최고 성능 조합 (성공률 기준):")
        for i, result in enumerate(analysis["best_by_success_rate"], 1):
            print(f"{i}. {result['experiment_name']}")
            print(f"   성공률: {result['summary']['success_rate']:.2%}")
            print(f"   평균 순위: {result['summary']['avg_target_position']:.1f}")
            print(f"   스플리터: {result['splitter_config']['name']}")
            print(f"   리트리버: {result['retriever_config']['name']}")
            print(f"   임베딩: {result['embedding_model']}")
            print()
        
        if analysis["best_by_target_position"]:
            print(f"🎯 최고 성능 조합 (목표 위치 기준):")
            for i, result in enumerate(analysis["best_by_target_position"], 1):
                print(f"{i}. {result['experiment_name']}")
                print(f"   평균 순위: {result['summary']['avg_target_position']:.1f}")
                print(f"   성공률: {result['summary']['success_rate']:.2%}")
                print(f"   스플리터: {result['splitter_config']['name']}")
                print(f"   리트리버: {result['retriever_config']['name']}")
                print(f"   임베딩: {result['embedding_model']}")
                print()
        
        print(f"📈 임베딩 모델별 성능:")
        for model, perf in analysis["embedding_model_performance"].items():
            print(f"   {model}: {perf['avg_success_rate']:.2%} (실험 {perf['experiments']}개)")
        
        print(f"\n🔧 스플리터별 성능:")
        for splitter, perf in analysis["splitter_performance"].items():
            print(f"   {splitter}: {perf['avg_success_rate']:.2%} (실험 {perf['experiments']}개)")
        
        print(f"\n🔍 리트리버별 성능:")
        for retriever, perf in analysis["retriever_performance"].items():
            print(f"   {retriever}: {perf['avg_success_rate']:.2%} (실험 {perf['experiments']}개)")


def main():
    """메인 실행 함수"""
    simulator = RAGSimulator()
    
    # 모든 실험 실행
    results = simulator.run_all_experiments(SOURCE_FILE_PATH)
    
    # 결과 저장
    results_file = simulator.save_results(results)
    
    # 결과 분석
    analysis = simulator.analyze_results(results)
    simulator.print_analysis(analysis)
    
    # 분석 결과도 저장
    analysis_file = results_file.replace('.json', '_analysis.json')
    with open(analysis_file, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 분석 결과 저장: {analysis_file}")
    
    return results, analysis


if __name__ == "__main__":
    main()