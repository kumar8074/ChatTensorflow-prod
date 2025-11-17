# ===================================================================================
# Project: ChatTensorFlow
# File: src/services/chunking/content_chunker.py
# Description: Loads TensorFlow documentation from scraped JSON and creates RAG-optimized chunks
# Author: LALAN KUMAR
# Created: [09-11-2025]
# Updated: [09-11-2025]
# LAST MODIFIED BY: LALAN KUMAR [https://github.com/kumar8074]
# Version: 1.1.0
# ===================================================================================

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import traceback

# Dynamically add the project root directory to sys.path
current_file_path = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(current_file_path, "../../../.."))
if project_root not in sys.path:
    sys.path.append(project_root)
    
from src.logger import logging
from src.config import CHUNKER_OUTPUT_DIR, CHUNKER_CHUNK_SIZE, CHUNKER_CHUNK_OVERLAP

class TensorFlowContentChunker:
    """
    Processes TensorFlow documentation from scraped JSON and creates
    intelligent, context-rich chunks optimized for RAG retrieval
    """
    
    def __init__(self, 
                 input_file="temp/docs_rag.json",
                 output_dir=CHUNKER_OUTPUT_DIR,
                 chunk_size=CHUNKER_CHUNK_SIZE,
                 chunk_overlap=CHUNKER_CHUNK_OVERLAP):
        self.input_file = input_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.processed_docs = []
        self.failed_docs = []
        self.all_chunks = []
        
        # Create output directory
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_scraped_data(self) -> List[Dict]:
        """Load the scraped TensorFlow documentation JSON"""
        if not os.path.exists(self.input_file):
            logging.error(f"Input file not found: {self.input_file}")
            logging.info("Please run tensorflow_scrapper.py first to generate the data")
            return []
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"Loaded {len(data)} documents from {self.input_file}")
        return data
    
    def classify_tf_page_type(self, url: str, title: str) -> str:
        """Classify TensorFlow documentation page type"""
        url_lower = url.lower()
        title_lower = title.lower()
        
        # API Reference pages
        if '/api_docs/python/tf/' in url_lower:
            if '/tf/keras/' in url_lower:
                return 'keras_api'
            elif '/tf/nn/' in url_lower:
                return 'nn_api'
            elif '/tf/data/' in url_lower:
                return 'data_api'
            elif '/tf/train/' in url_lower:
                return 'training_api'
            else:
                return 'core_api'
        
        # Guide pages
        elif '/guide/' in url_lower:
            if 'keras' in url_lower or 'keras' in title_lower:
                return 'keras_guide'
            elif 'data' in url_lower:
                return 'data_guide'
            elif 'estimator' in url_lower:
                return 'estimator_guide'
            else:
                return 'general_guide'
        
        # Tutorial pages
        elif '/tutorials/' in url_lower:
            return 'tutorial'
        
        # Overview/index pages
        elif 'overview' in title_lower or 'index' in url_lower:
            return 'overview'
        
        else:
            return 'documentation'
    
    def extract_breadcrumbs_from_url(self, url: str) -> List[str]:
        """Extract breadcrumb navigation from URL path"""
        breadcrumbs = []
        
        # Parse URL path
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split('/') if p]
        
        # Build meaningful breadcrumbs
        for i, part in enumerate(path_parts):
            # Clean up part
            clean_part = part.replace('_', ' ').replace('-', ' ').title()
            
            # Skip version numbers
            if re.match(r'v\d+', part):
                continue
            
            # Map common paths
            if part == 'api_docs':
                breadcrumbs.append('API Documentation')
            elif part == 'python':
                breadcrumbs.append('Python')
            elif part == 'tf':
                breadcrumbs.append('TensorFlow')
            elif part in ['guide', 'tutorials']:
                breadcrumbs.append(clean_part)
            else:
                breadcrumbs.append(clean_part)
        
        return breadcrumbs
    
    def extract_code_blocks(self, content: str) -> List[Dict]:
        """Extract code blocks from markdown content"""
        code_blocks = []
        
        # Pattern for markdown code blocks
        pattern = r'```(\w+)?\n(.*?)```'
        matches = re.finditer(pattern, content, re.DOTALL)
        
        for idx, match in enumerate(matches):
            language = match.group(1) or 'python'
            code = match.group(2).strip()
            
            if not code:
                continue
            
            # Get context (text before the code block)
            start_pos = match.start()
            context_text = content[max(0, start_pos - 300):start_pos]
            
            # Find the last heading or sentence
            context = self._extract_code_context(context_text)
            
            code_blocks.append({
                'index': idx,
                'code': code,
                'language': language,
                'context': context,
                'lines': len(code.split('\n')),
                'start_pos': match.start(),
                'end_pos': match.end()
            })
        
        return code_blocks
    
    def _extract_code_context(self, text: str) -> str:
        """Extract meaningful context for a code block"""
        # Look for the last heading
        heading_match = re.search(r'#+\s+(.+?)(?:\n|$)', text)
        if heading_match:
            return heading_match.group(1).strip()
        
        # Look for the last complete sentence
        sentences = re.split(r'[.!?]\s+', text)
        if sentences:
            last_sentence = sentences[-1].strip()
            if len(last_sentence) > 20:
                return last_sentence[:100] + '...' if len(last_sentence) > 100 else last_sentence
        
        return ""
    
    def extract_headings(self, content: str) -> List[Dict]:
        """Extract all headings from markdown content"""
        headings = []
        
        # Pattern for markdown headings
        pattern = r'^(#{1,6})\s+(.+?)$'
        
        for match in re.finditer(pattern, content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            position = match.start()
            
            headings.append({
                'level': level,
                'text': text,
                'position': position
            })
        
        return headings
    
    def extract_api_signature(self, content: str, title: str) -> str:
        """Extract API signature for API reference pages"""
        # Look for function/class signatures in the content
        # Common patterns in TensorFlow docs
        patterns = [
            r'```python\n((?:class|def)\s+\w+.*?)\n```',
            r'^((?:class|def)\s+\w+.*?)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content[:1000], re.MULTILINE)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def split_content_into_sections(self, content: str, headings: List[Dict]) -> List[Dict]:
        """Split content into sections based on headings"""
        sections = []
        
        if not headings:
            # No headings, treat entire content as one section
            return [{'heading': '', 'level': 0, 'content': content, 'start': 0, 'end': len(content)}]
        
        for i, heading in enumerate(headings):
            start = heading['position']
            end = headings[i + 1]['position'] if i + 1 < len(headings) else len(content)
            
            section_content = content[start:end].strip()
            
            # Remove the heading line from content
            section_content = re.sub(r'^#+\s+.+?\n', '', section_content, count=1)
            
            sections.append({
                'heading': heading['text'],
                'level': heading['level'],
                'content': section_content,
                'start': start,
                'end': end
            })
        
        return sections
    
    def chunk_section(self, section: Dict, url: str, title: str, breadcrumbs: List[str], 
                     page_type: str, code_blocks: List[Dict]) -> List[Dict]:
        """Create chunks from a section with intelligent splitting"""
        chunks = []
        content = section['content']
        heading = section['heading']
        
        # Find code blocks in this section
        section_code_blocks = [
            cb for cb in code_blocks 
            if section['start'] <= cb['start_pos'] < section['end']
        ]
        
        # Split content by paragraphs
        paragraphs = content.split('\n\n')
        
        current_chunk_paras = []
        current_chunk_codes = []
        word_count = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if this paragraph contains a code block
            para_codes = [cb for cb in section_code_blocks if cb['code'] in para]
            
            para_words = len(para.split())
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if word_count > 0 and (word_count + para_words > self.chunk_size):
                chunk_text = '\n\n'.join(current_chunk_paras)
                chunks.append(self._create_chunk(
                    text=chunk_text,
                    heading=heading,
                    url=url,
                    title=title,
                    breadcrumbs=breadcrumbs,
                    page_type=page_type,
                    code_blocks=current_chunk_codes
                ))
                
                # Start new chunk with overlap
                if current_chunk_paras:
                    overlap_text = ' '.join(current_chunk_paras).split()[-self.chunk_overlap:]
                    current_chunk_paras = [' '.join(overlap_text)] if overlap_text else []
                    word_count = len(overlap_text)
                else:
                    current_chunk_paras = []
                    word_count = 0
                current_chunk_codes = []
            
            current_chunk_paras.append(para)
            word_count += para_words
            
            if para_codes:
                current_chunk_codes.extend(para_codes)
        
        # Save final chunk
        if current_chunk_paras:
            chunk_text = '\n\n'.join(current_chunk_paras)
            chunks.append(self._create_chunk(
                text=chunk_text,
                heading=heading,
                url=url,
                title=title,
                breadcrumbs=breadcrumbs,
                page_type=page_type,
                code_blocks=current_chunk_codes
            ))
        
        return chunks
    
    def _create_chunk(self, text: str, heading: str, url: str, title: str,
                     breadcrumbs: List[str], page_type: str, code_blocks: List[Dict]) -> Dict:
        """Create a complete chunk with metadata"""
        # Format code blocks
        formatted_code = ""
        if code_blocks:
            for cb in code_blocks:
                formatted_code += f"\n\n```{cb['language']}\n{cb['code']}\n```"
        
        full_text = text + formatted_code if formatted_code else text
        
        # Create enriched text with context
        enriched_text = self._enrich_context(
            text=full_text,
            heading=heading,
            title=title,
            url=url,
            breadcrumbs=breadcrumbs,
            page_type=page_type,
            has_code=len(code_blocks) > 0
        )
        
        return {
            'heading': heading,
            'text': text,
            'full_text': full_text,
            'enriched_text': enriched_text,
            'source_url': url,
            'page_title': title,
            'breadcrumbs': breadcrumbs,
            'page_type': page_type,
            'code_blocks': code_blocks,
            'has_code': len(code_blocks) > 0,
            'word_count': len(text.split()),
            'total_code_lines': sum(cb['lines'] for cb in code_blocks),
            'chunk_id': f"{url}#{heading.replace(' ', '_')}" if heading else url
        }
    
    def _enrich_context(self, text: str, heading: str, title: str, url: str,
                       breadcrumbs: List[str], page_type: str, has_code: bool) -> str:
        """Add contextual information to chunk for better RAG retrieval"""
        context_parts = []
        
        context_parts.append("Documentation: TensorFlow")
        
        if title:
            context_parts.append(f"Page: {title}")
        
        if breadcrumbs:
            context_parts.append(f"Location: {' > '.join(breadcrumbs)}")
        
        if heading:
            context_parts.append(f"Section: {heading}")
        
        context_parts.append(f"Type: {page_type.replace('_', ' ').title()}")
        
        if has_code:
            context_parts.append("Contains: Code Examples")
        
        context_header = '\n'.join(context_parts)
        enriched = f"{context_header}\n\n{text}"
        
        return enriched
    
    def process_document(self, doc: Dict):
        """Process a single document from the scraped data"""
        try:
            url = doc.get('url', '')
            title = doc.get('title', '')
            content = doc.get('content', '')
            
            if not content or len(content.strip()) < 100:
                self.failed_docs.append({
                    'url': url,
                    'reason': 'Content too short or empty'
                })
                return
            
            # Classify page type
            page_type = self.classify_tf_page_type(url, title)
            
            # Extract breadcrumbs
            breadcrumbs = self.extract_breadcrumbs_from_url(url)
            
            # Extract code blocks
            code_blocks = self.extract_code_blocks(content)
            
            # Extract headings
            headings = self.extract_headings(content)
            
            # Split into sections
            sections = self.split_content_into_sections(content, headings)
            
            # Create header chunk
            header_chunk = self._create_header_chunk(
                title=title,
                url=url,
                content=content,
                breadcrumbs=breadcrumbs,
                page_type=page_type
            )
            
            doc_chunks = []
            if header_chunk:
                doc_chunks.append(header_chunk)
            
            # Process each section
            for section in sections:
                section_chunks = self.chunk_section(
                    section=section,
                    url=url,
                    title=title,
                    breadcrumbs=breadcrumbs,
                    page_type=page_type,
                    code_blocks=code_blocks
                )
                doc_chunks.extend(section_chunks)
            
            # Store processed document info
            self.processed_docs.append({
                'url': url,
                'title': title,
                'page_type': page_type,
                'total_chunks': len(doc_chunks),
                'total_words': sum(c['word_count'] for c in doc_chunks),
                'total_code_blocks': len(code_blocks),
                'chunks': doc_chunks
            })
            
            # Add to all chunks
            self.all_chunks.extend(doc_chunks)
            
        except Exception as e:
            logging.error(f"Error processing document {doc.get('url', 'unknown')}: {e}")
            traceback.print_exc()
            self.failed_docs.append({
                'url': doc.get('url', 'unknown'),
                'reason': str(e)
            })
    
    def _create_header_chunk(self, title: str, url: str, content: str,
                            breadcrumbs: List[str], page_type: str) -> Dict:
        """Create header chunk with page overview"""
        if not title:
            return None
        
        # Extract first paragraph as description
        description = ""
        paragraphs = content.split('\n\n')
        for para in paragraphs[:3]:
            para = para.strip()
            # Skip headings and code blocks
            if para and not para.startswith('#') and not para.startswith('```'):
                description = para
                break
        
        # For API pages, try to extract signature
        signature = ""
        if 'api' in page_type:
            signature = self.extract_api_signature(content, title)
        
        text_parts = [title]
        if signature:
            text_parts.append(f"\nSignature: {signature}")
        if description:
            text_parts.append(f"\n{description[:300]}")
        
        text = '\n'.join(text_parts)
        
        enriched_text = self._enrich_context(
            text=text,
            heading="Overview",
            title=title,
            url=url,
            breadcrumbs=breadcrumbs,
            page_type=page_type,
            has_code=bool(signature)
        )
        
        return {
            'heading': 'Overview',
            'text': text,
            'full_text': text,
            'enriched_text': enriched_text,
            'source_url': url,
            'page_title': title,
            'breadcrumbs': breadcrumbs,
            'page_type': page_type,
            'code_blocks': [],
            'has_code': bool(signature),
            'word_count': len(text.split()),
            'total_code_lines': 0,
            'chunk_id': f"{url}#overview"
        }
    
    def process_all_documents(self):
        """Process all documents from the scraped data"""
        docs = self.load_scraped_data()
        
        if not docs:
            logging.error("No documents to process!")
            return
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting TensorFlow documentation chunking...")
        logging.info(f"Total documents to process: {len(docs)}")
        logging.info(f"Chunk size: {self.chunk_size} words")
        logging.info(f"Chunk overlap: {self.chunk_overlap} words")
        logging.info(f"{'='*60}\n")
        
        for idx, doc in enumerate(docs, 1):
            if idx % 10 == 0:
                logging.info(f"Progress: {idx}/{len(docs)} documents processed")
            self.process_document(doc)
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Chunking completed!")
        logging.info(f"Total documents processed: {len(self.processed_docs)}")
        logging.info(f"Total documents failed: {len(self.failed_docs)}")
        logging.info(f"Total chunks created: {len(self.all_chunks)}")
        logging.info(f"Total code blocks: {sum(d['total_code_blocks'] for d in self.processed_docs)}")
        logging.info(f"{'='*60}\n")
    
    def save_data(self):
        """Save chunked data in RAG-optimized formats"""
        
        # 1. Save as JSONL for easy streaming and vector DB ingestion
        chunks_file = os.path.join(self.output_dir, "chunks_for_rag.jsonl")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            for chunk in self.all_chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        logging.info(f"Saved chunks for RAG (JSONL): {chunks_file}")
        
        # 2. Save all chunks as JSON
        all_chunks_file = os.path.join(self.output_dir, "all_chunks.json")
        with open(all_chunks_file, 'w', encoding='utf-8') as f:
            json.dump(self.all_chunks, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved all chunks (JSON): {all_chunks_file}")
        
        # 3. Save document-level data with chunks
        docs_file = os.path.join(self.output_dir, "docs_with_chunks.json")
        with open(docs_file, 'w', encoding='utf-8') as f:
            json.dump(self.processed_docs, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved document-level data: {docs_file}")
        
        # 4. Save statistics
        stats = self._compute_statistics()
        stats_file = os.path.join(self.output_dir, "chunking_statistics.json")
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved statistics: {stats_file}")
        
        self._print_statistics(stats)
    
    def _compute_statistics(self) -> Dict:
        """Compute comprehensive statistics"""
        stats = {
            'total_documents': len(self.processed_docs),
            'total_chunks': len(self.all_chunks),
            'total_words': sum(d['total_words'] for d in self.processed_docs),
            'total_code_blocks': sum(d['total_code_blocks'] for d in self.processed_docs),
            'chunks_with_code': sum(1 for c in self.all_chunks if c['has_code']),
            'avg_chunks_per_doc': len(self.all_chunks) / len(self.processed_docs) if self.processed_docs else 0,
            'avg_words_per_chunk': sum(c['word_count'] for c in self.all_chunks) / len(self.all_chunks) if self.all_chunks else 0,
            'failed_documents': len(self.failed_docs),
            'failed_docs_list': self.failed_docs,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'docs_by_type': {},
            'chunks_by_type': {},
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        # Count by page type
        for doc in self.processed_docs:
            page_type = doc['page_type']
            stats['docs_by_type'][page_type] = stats['docs_by_type'].get(page_type, 0) + 1
        
        for chunk in self.all_chunks:
            page_type = chunk['page_type']
            stats['chunks_by_type'][page_type] = stats['chunks_by_type'].get(page_type, 0) + 1
        
        return stats
    
    def _print_statistics(self, stats: Dict):
        """Print statistics summary"""
        logging.info(f"\nChunking Statistics Summary:")
        logging.info(f"   - Total documents: {stats['total_documents']}")
        logging.info(f"   - Total chunks: {stats['total_chunks']}")
        logging.info(f"   - Total words: {stats['total_words']:,}")
        logging.info(f"   - Total code blocks: {stats['total_code_blocks']}")
        logging.info(f"   - Chunks with code: {stats['chunks_with_code']}")
        logging.info(f"   - Avg chunks/document: {stats['avg_chunks_per_doc']:.1f}")
        logging.info(f"   - Avg words/chunk: {stats['avg_words_per_chunk']:.1f}")
        logging.info(f"   - Failed documents: {stats['failed_documents']}")
        logging.info(f"   - Chunk size: {stats['chunk_size']} words")
        logging.info(f"   - Chunk overlap: {stats['chunk_overlap']} words")
        
        logging.info(f"\nDocuments by Type:")
        for page_type, count in sorted(stats['docs_by_type'].items(), key=lambda x: x[1], reverse=True):
            logging.info(f"   - {page_type}: {count}")
        
        logging.info(f"\nChunks by Type:")
        for page_type, count in sorted(stats['chunks_by_type'].items(), key=lambda x: x[1], reverse=True):
            logging.info(f"   - {page_type}: {count}")


def main():
    """Main entry point"""
    import time
    start_time = time.time()
    
    chunker = TensorFlowContentChunker(
        input_file="temp/docs_rag.json",
        output_dir="temp/chunked_data",
        chunk_size=1000,
        chunk_overlap=200
    )
    
    logging.info("Starting TensorFlow documentation chunking...")
    chunker.process_all_documents()
    
    chunker.save_data()
    
    duration = time.time() - start_time
    
    logging.info(f"\nChunking completed successfully!")
    logging.info(f"Finished in {duration:.2f} seconds")
    logging.info("\nNext steps:")
    logging.info("1. Load 'tensorflow_chunks_for_rag.jsonl' into your vector database")
    logging.info("2. Generate embeddings for the 'enriched_text' field")
    logging.info("3. Use 'full_text' field for displaying content with code examples")
    logging.info("4. Store 'source_url', 'page_title', 'heading', and 'code_blocks' for citations")
    logging.info("5. Use 'page_type' for filtering specific types of documentation")


if __name__ == "__main__":
    main()