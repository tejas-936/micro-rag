import os
import re
import glob
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict

class VectorStore:
    def __init__(self, data_dir: str = "../data", embed_dim: int = 384, model_name: str = 'all-MiniLM-L6-v2'):
        self.data_dir = data_dir
        self.model_name = model_name
        self.cache_dir = os.path.join(os.path.dirname(__file__), ".cache")
        self.vec_path = os.path.join(self.cache_dir, "embeddings.npy")
        self.txt_path = os.path.join(self.cache_dir, "chunks.json")
        
        self.chunks: List[str] = []
        self.embeddings: np.ndarray = np.empty((0, embed_dim))
        self.model = None

    def _get_model(self):
        if self.model is None:
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def load_or_build(self):
        if os.path.exists(self.vec_path) and os.path.exists(self.txt_path):
            print("Loading cached embeddings and chunks...")
            self.embeddings = np.load(self.vec_path)
            with open(self.txt_path, 'r', encoding='utf-8') as f:
                self.chunks = json.load(f)
        else:
            print("No cache found. Building embeddings from source files...")
            self.build_index()

    def build_index(self):
        self.chunks = self._read_and_chunk_documents()
        if not self.chunks:
            print("Warning: No documents found to index. Clearing cache...")
            self.embeddings = np.empty((0, self.embeddings.shape[1] if self.embeddings.ndim > 1 else 384))
            if os.path.exists(self.vec_path):
                try: os.remove(self.vec_path)
                except: pass
            if os.path.exists(self.txt_path):
                try: os.remove(self.txt_path)
                except: pass
            return

        model = self._get_model()
        print(f"Computing embeddings for {len(self.chunks)} chunks...")
        
        embeds = model.encode(self.chunks, convert_to_numpy=True)
        self.embeddings = embeds

        os.makedirs(self.cache_dir, exist_ok=True)
        np.save(self.vec_path, self.embeddings)
        with open(self.txt_path, 'w', encoding='utf-8') as f:
            json.dump(self.chunks, f)
        print("Embeddings cached successfully.")

    def _read_and_chunk_documents(self, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        all_chunks = []
        search_paths = [
            os.path.join(self.data_dir, "**/*.txt"),
            os.path.join(self.data_dir, "**/*.md")
        ]
        
        files = []
        for path in search_paths:
            files.extend(glob.glob(path, recursive=True))

        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            basename = os.path.basename(file_path)
            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.md':
                chunks = self._chunk_markdown(text, basename, chunk_size)
            else:
                chunks = self._chunk_sliding_window(text, basename, chunk_size, overlap)
            
            all_chunks.extend(chunks)
                
        return all_chunks

    def _chunk_markdown(self, text: str, source: str, max_chunk_size: int = 500) -> List[str]:
        heading_pattern = r'(?=^#{1,4}\s+.+$)'
        sections = re.split(heading_pattern, text, flags=re.MULTILINE)
        
        sections = [s.strip() for s in sections if s.strip()]
        
        chunks = []
        for section in sections:
            if len(section) <= max_chunk_size:
                chunks.append(f"[Source: {source}]\n{section}")
            else:
                sub_chunks = self._chunk_sliding_window(section, source, max_chunk_size, overlap=100)
                chunks.extend(sub_chunks)
        
        merged = []
        buffer = ""
        for chunk in chunks:
            content = chunk.split('\n', 1)[1] if '\n' in chunk else chunk
            if len(content) < 80 and buffer == "":
                buffer = content
            elif buffer:
                merged_content = buffer + "\n\n" + content
                if len(merged_content) <= max_chunk_size:
                    merged.append(f"[Source: {source}]\n{merged_content}")
                    buffer = ""
                else:
                    merged.append(f"[Source: {source}]\n{buffer}")
                    merged.append(chunk)
                    buffer = ""
            else:
                merged.append(chunk)
        
        if buffer:
            merged.append(f"[Source: {source}]\n{buffer}")
        
        return merged if merged else chunks

    def _chunk_sliding_window(self, text: str, source: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk = text[start:end]
            chunks.append(f"[Source: {source}]\n{chunk.strip()}")
            
            if end == len(text):
                break
            start += (chunk_size - overlap)
        
        return chunks

    def search(self, query: str, top_k: int = 5) -> List[Dict[str, float]]:
        if len(self.chunks) == 0:
            return []

        model = self._get_model()
        query_vec = model.encode([query], convert_to_numpy=True)[0]
        
        norm_query = np.linalg.norm(query_vec)
        norm_embeds = np.linalg.norm(self.embeddings, axis=1)
        
        if norm_query == 0 or np.any(norm_embeds == 0):
            similarities = np.zeros(len(self.embeddings))
        else:
            similarities = np.dot(self.embeddings, query_vec) / (norm_embeds * norm_query)

        actual_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(similarities)[-actual_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(similarities[idx])
            })
        return results
