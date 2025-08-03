# -*- coding: utf-8 -*-
"""
다양한 텍스트 분할 방법 구현
"""

import re
from typing import List
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_text_splitters import MarkdownHeaderTextSplitter


def load_markdown_file(file_path: str) -> str:
    """마크다운 파일을 로드합니다."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def recursive_text_split(md_text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """RecursiveCharacterTextSplitter를 사용하여 문서를 분할합니다."""
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=[
            "\n\n",  # 단락 구분
            "\n",    # 줄 구분  
            " ",     # 단어 구분
            ""       # 문자 구분
        ]
    )
    
    chunks = splitter.split_text(md_text)
    
    docs = []
    for i, chunk in enumerate(chunks):
        # 간단한 제목 추출 (첫 번째 줄이 #으로 시작하는 경우)
        lines = chunk.split('\n')
        title = ""
        for line in lines:
            if line.startswith('#'):
                title = line.strip('#').strip()
                break
        
        # PNS 및 purchaseState 관련 청크 특별 처리
        is_pns_related = any(keyword in chunk.upper() for keyword in ['PNS', 'PAYMENT NOTIFICATION'])
        has_purchase_state = 'purchaseState' in chunk or 'purcahseState' in chunk
        has_target_content = all(keyword in chunk for keyword in ['purcahseState', 'COMPLETED', 'CANCELED'])
        
        doc = Document(
            page_content=chunk,
            metadata={
                "type": "documentation",
                "source": "dev_center_guide_allmd_touched.md",
                "chunk_idx": i,
                "chunk_size": len(chunk),
                "title": title if title else f"Chunk {i+1}",
                "is_pns": is_pns_related,
                "has_purchase_state": has_purchase_state,
                "has_target_content": has_target_content,
                "splitter_type": "recursive",
                "splitter_config": f"size_{chunk_size}_overlap_{chunk_overlap}"
            }
        )
        docs.append(doc)
    
    return docs


def markdown_header_split(md_text: str, headers_to_split_on: List[tuple]) -> List[Document]:
    """MarkdownHeaderTextSplitter를 사용하여 문서를 분할합니다."""
    
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = splitter.split_text(md_text)
    
    result_docs = []
    for i, doc in enumerate(docs):
        metadata = doc.metadata.copy()
        
        # 계층적 제목 생성
        full_title = ""
        if "title" in metadata:
            full_title = metadata["title"]
        if "section" in metadata:
            full_title += f" > {metadata['section']}" if full_title else metadata["section"]
        if "subsection" in metadata:
            full_title += f" > {metadata['subsection']}"
        if "subsubsection" in metadata:
            full_title += f" > {metadata['subsubsection']}"
        
        # 내용 앞에 제목 추가
        if full_title:
            content = f"Title: {full_title}\n{doc.page_content}"
        else:
            content = doc.page_content
        
        # 특별 처리
        is_pns_related = any(keyword in content.upper() for keyword in ['PNS', 'PAYMENT NOTIFICATION'])
        has_purchase_state = 'purchaseState' in content or 'purcahseState' in content
        has_target_content = all(keyword in content for keyword in ['purcahseState', 'COMPLETED', 'CANCELED'])
        
        result_doc = Document(
            page_content=content,
            metadata={
                **metadata,
                "type": "documentation", 
                "source": "dev_center_guide_allmd_touched.md",
                "chunk_idx": i,
                "chunk_size": len(content),
                "full_title": full_title,
                "is_pns": is_pns_related,
                "has_purchase_state": has_purchase_state,
                "has_target_content": has_target_content,
                "splitter_type": "markdown_headers"
            }
        )
        result_docs.append(result_doc)
    
    return result_docs


def hybrid_split(md_text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """하이브리드 접근: 먼저 마크다운 헤더로 분할 후 큰 섹션만 다시 분할"""
    
    # 1단계: 마크다운 헤더로 분할
    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "title"),
            ("##", "section"), 
            ("###", "subsection"),
        ]
    )
    header_docs = header_splitter.split_text(md_text)
    
    # 2단계: 큰 청크들만 다시 분할
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    
    final_docs = []
    chunk_idx = 0
    
    for header_doc in header_docs:
        if len(header_doc.page_content) > chunk_size * 1.5:
            # 큰 섹션은 다시 분할
            sub_chunks = recursive_splitter.split_text(header_doc.page_content)
            for sub_chunk in sub_chunks:
                # 계층적 제목 유지
                metadata = header_doc.metadata.copy()
                full_title = ""
                if "title" in metadata:
                    full_title = metadata["title"]
                if "section" in metadata:
                    full_title += f" > {metadata['section']}" if full_title else metadata["section"]
                if "subsection" in metadata:
                    full_title += f" > {metadata['subsection']}"
                
                if full_title:
                    content = f"Title: {full_title}\n{sub_chunk}"
                else:
                    content = sub_chunk
                
                # 특별 처리
                is_pns_related = any(keyword in content.upper() for keyword in ['PNS', 'PAYMENT NOTIFICATION'])
                has_purchase_state = 'purchaseState' in content or 'purcahseState' in content
                has_target_content = all(keyword in content for keyword in ['purcahseState', 'COMPLETED', 'CANCELED'])
                
                doc = Document(
                    page_content=content,
                    metadata={
                        **metadata,
                        "type": "documentation",
                        "source": "dev_center_guide_allmd_touched.md", 
                        "chunk_idx": chunk_idx,
                        "chunk_size": len(content),
                        "full_title": full_title,
                        "is_pns": is_pns_related,
                        "has_purchase_state": has_purchase_state,
                        "has_target_content": has_target_content,
                        "splitter_type": "hybrid",
                        "was_resplit": True
                    }
                )
                final_docs.append(doc)
                chunk_idx += 1
        else:
            # 작은 섹션은 그대로 유지
            metadata = header_doc.metadata.copy()
            full_title = ""
            if "title" in metadata:
                full_title = metadata["title"]
            if "section" in metadata:
                full_title += f" > {metadata['section']}" if full_title else metadata["section"]
            if "subsection" in metadata:
                full_title += f" > {metadata['subsection']}"
            
            if full_title:
                content = f"Title: {full_title}\n{header_doc.page_content}"
            else:
                content = header_doc.page_content
            
            # 특별 처리
            is_pns_related = any(keyword in content.upper() for keyword in ['PNS', 'PAYMENT NOTIFICATION'])
            has_purchase_state = 'purchaseState' in content or 'purcahseState' in content
            has_target_content = all(keyword in content for keyword in ['purcahseState', 'COMPLETED', 'CANCELED'])
            
            doc = Document(
                page_content=content,
                metadata={
                    **metadata,
                    "type": "documentation",
                    "source": "dev_center_guide_allmd_touched.md",
                    "chunk_idx": chunk_idx, 
                    "chunk_size": len(content),
                    "full_title": full_title,
                    "is_pns": is_pns_related,
                    "has_purchase_state": has_purchase_state,
                    "has_target_content": has_target_content,
                    "splitter_type": "hybrid",
                    "was_resplit": False
                }
            )
            final_docs.append(doc)
            chunk_idx += 1
    
    return final_docs


def get_splitter_function(splitter_config: dict):
    """설정에 따라 적절한 분할 함수를 반환합니다."""
    splitter_type = splitter_config["type"]
    
    if splitter_type == "recursive":
        return lambda md_text: recursive_text_split(
            md_text, 
            splitter_config["chunk_size"], 
            splitter_config["chunk_overlap"]
        )
    elif splitter_type == "markdown_headers":
        return lambda md_text: markdown_header_split(
            md_text, 
            splitter_config["headers_to_split_on"]
        )
    elif splitter_type == "hybrid":
        return lambda md_text: hybrid_split(
            md_text, 
            splitter_config["chunk_size"], 
            splitter_config["chunk_overlap"]
        )
    else:
        raise ValueError(f"Unknown splitter type: {splitter_type}")


def analyze_target_content_coverage(docs: List[Document]) -> dict:
    """목표 내용이 어떻게 분할되었는지 분석합니다."""
    
    analysis = {
        "total_docs": len(docs),
        "docs_with_target_content": [],
        "docs_with_purchase_state": [],
        "docs_with_pns": [],
        "target_content_fragments": []
    }
    
    for i, doc in enumerate(docs):
        content = doc.page_content
        metadata = doc.metadata
        
        if metadata.get("has_target_content", False):
            analysis["docs_with_target_content"].append({
                "index": i,
                "title": metadata.get("title", ""),
                "chunk_size": len(content),
                "content_preview": content[:200] + "..." if len(content) > 200 else content
            })
        
        if metadata.get("has_purchase_state", False):
            analysis["docs_with_purchase_state"].append({
                "index": i, 
                "title": metadata.get("title", ""),
                "chunk_size": len(content)
            })
        
        if metadata.get("is_pns", False):
            analysis["docs_with_pns"].append({
                "index": i,
                "title": metadata.get("title", ""),
                "chunk_size": len(content)
            })
        
        # 목표 내용의 일부라도 포함하는 경우
        target_keywords = ['purcahseState', 'COMPLETED', 'CANCELED']
        found_keywords = [kw for kw in target_keywords if kw in content]
        if found_keywords:
            analysis["target_content_fragments"].append({
                "index": i,
                "found_keywords": found_keywords,
                "title": metadata.get("title", ""),
                "content_preview": content[:150] + "..." if len(content) > 150 else content
            })
    
    return analysis