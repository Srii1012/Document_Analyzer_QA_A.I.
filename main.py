import os 
import shutil
from pathlib import Path
from typing import Dict, List
import uuid

import pdfplumber
import pytesseract
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel

import json
import random

from groq import Groq
from dotenv import load_dotenv

import warnings
import logging

warnings.filterwarnings("ignore",message="Cropbox missing from /Page")

logging.getLogger("pdfplumber").setLevel(logging.ERROR)

load_dotenv()

# LLM initialize to the system 
llm_customer = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Pydantic models
class SearchRequest(BaseModel):
    query: str
    doc_ids: List[str]
    mode: str = "all_documents"

class SampleGenerationRequest(BaseModel):
    num_docs: int = 25
    topics: List[str] = ["Technology", "Healthcare", "Finance"]

def generate_answer(my_ques: str, context: List[str]) -> str:
    """Use Basic LLM to generate answers from search results"""
    
    prompt = f"""
    Help to answer the question: {my_ques}
    
    Based on user's document excerpts:
    
    {chr(10).join([f"[Doc{i+1}]: {ctx}" for i, ctx in enumerate(context)])}

    Provide a comprehensive answer that:
    - Directly answers the question
    - Synthesizes information from multiple sources
    - Uses [Doc1], [Doc2] etc. to cite sources
    - Keeps the answer concise but informative
    """
    
    try:
        response = llm_customer.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as wrong:
        return f"Error generating answer: {str(wrong)}"
    
def identify_themes(contexts: List[Dict], my_ques: str) -> List[Dict]:
    """Identify common themes across documents"""
    try:
        # Combine all contexts
        combined_text = " ".join([ctx.get("text", "") for ctx in contexts])
        
        theme_prompt = f"""
        Analyze these document excerpts and identify 2-3 major themes related to the query: "{my_ques}"
        
        Document content:
        {combined_text[:3000]}  # Limit to avoid token limits
        
        Return themes in this format:
        Theme 1: [Title]
        Description: [Brief description]
        
        Theme 2: [Title] 
        Description: [Brief description]
        
        Focus on the most significant patterns or topics that appear across multiple documents.
        """
        response = llm_customer.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": theme_prompt}],
            temperature=0.4
        )

        # Breaks down response into Clean structured format
        themes_words = response.choices[0].message.content
        themes = []

        lined_one = themes_words.split('\n')
        current_theme = {}

        for line in lined_one:
            if line.startswith('Theme '):
                if current_theme:
                    themes.append(current_theme)
                current_theme = {"title": line.split(':', 1)[1].strip() if ':' in line else line}
            elif line.startswith('Description:'):
                current_theme["description"] = line.split(':', 1)[1].strip()
                current_theme["supporting_docs"] = [ctx.get("filename", "Unknown") for ctx in contexts[:3]]

        # Don't forget to add the last theme
        if current_theme:
            themes.append(current_theme)

        return themes[:3]
    except Exception as Wrong:
        return [{"title": "Analysis Error", "description": f"Could not analyze themes: {Wrong}", "supporting_docs": []}]

def generate_sample_documents(num_docs: int, topics: List[str]) -> List[Dict]:
    """Generate sample files for testing"""
    documents = []

    # example title templates
    templates = {
        "Technology": [
            "Artificial Intelligence in Modern Computing",
            "The Future of Cloud Computing",
            "Cybersecurity Best Practices",
            "Machine Learning Applications",
            "Blockchain Technology Overview"
        ],
        "Healthcare": [
            "Advances in Medical Technology",
            "Patient Care Quality Improvement",
            "Telemedicine Implementation",
            "Healthcare Data Security",
            "Preventive Medicine Strategies"
        ],
        "Finance": [
            "Investment Portfolio Management",
            "Risk Assessment Methodologies",
            "Digital Banking Evolution",
            "Cryptocurrency Market Analysis",
            "Financial Regulatory Compliance"
        ],
        "Education": [
            "Online Learning Effectiveness",
            "Student Assessment Methods",
            "Educational Technology Integration",
            "Curriculum Development Strategies",
            "Teacher Training Programs"
        ],
        "Environment": [
            "Climate Change Mitigation",
            "Renewable Energy Solutions",
            "Sustainable Development Goals",
            "Environmental Impact Assessment",
            "Green Technology Innovation"
        ]
    }

    for i in range(num_docs):
        topic = random.choice(topics)
        title = random.choice(templates.get(topic, ["General Document"]))

        # Generate content using LLM
        try:
            content_prompt = f"""
            Write a brief 200-300 word document about: {title}
            
            Make it informative and professional. Include:
            - Key concepts and definitions
            - Current trends or developments
            - Practical applications or implications
            - Future outlook or recommendations
            
            Topic category: {topic}
            """
            
            response = llm_customer.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": content_prompt}],
                temperature=0.7
            )
            
            content = response.choices[0].message.content
            
        except Exception as e:
            # Fallback content if LLM fails
            content = f"""
            {title}

            This document covers important aspects of {topic.lower()}. It discusses current trends, 
            challenges, and opportunities in the field. Key considerations include implementation 
            strategies, best practices, and future developments.
            
            The document provides insights into practical applications and recommendations for 
            professionals working in {topic.lower()}. It serves as a reference for understanding 
            the current state and future direction of this important area.
            
            Generated document #{i+1} for testing purposes.
            """
        
        # Create document entry
        doc_id = str(uuid.uuid4())
        doc_info = {
            "filename": f"{title.replace(' ', '_')}_{i+1}.txt",
            "doc_id": doc_id,
            "status": "success",
            "text_length": len(content),
            "content": content,
            "topic": topic
        }
        
        documents.append(doc_info)
        
        # Store in vector database
        try:
            collection_name = f"documents_{doc_id}"
            
            # Create collection
            qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            
            # Store content with proper metadata structure
            store_to_vector_db(
                text=content,
                metadata={
                    "doc_id": doc_id,
                    "filename": doc_info["filename"],
                    "topic": topic,
                    "page": 1
                },
                collection_name=collection_name
            )
            
        except Exception as e:
            print(f"Error storing sample document {i+1}: {e}")
            # Continue with other documents even if one fails
    
    return documents


app = FastAPI(
    title="Document Researcher",
    description="Upload documents & query them. For personal search use.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "data/uploads" 
os.makedirs(UPLOAD_DIR, exist_ok=True)

embedder = SentenceTransformer("all-MiniLM-L6-v2")
qdrant = QdrantClient(path="./qdrant_db")

def extract_pdf_text(file_path: str) -> tuple[str, List[Dict]]:
    """Extract text from PDF with page information"""
    try:
        text_parts = []
        page_info = []
        
        with pdfplumber.open(file_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
                    page_info.append({
                        "page": page_num,
                        "text": page_text,
                        "char_start": len("".join(text_parts[:-1])),
                        "char_end": len("".join(text_parts))
                    })
        
        return "\n".join(text_parts).strip(), page_info
    except Exception as e:
        raise RuntimeError(f"PDF extraction failed: {str(e)}")

def extract_image_text(file_path: str) -> tuple[str, List[Dict]]:
    """Extract text from images using Tesseract OCR"""
    try:
        pic = Image.open(file_path)
        
        if pic.mode != 'RGB':
            pic = pic.convert('RGB')
        
        ocr_config = r'--oem 3 --psm 6'
        text = pytesseract.image_to_string(pic, config=ocr_config).strip()

        if not text:
            ocr_config = r'--oem 3 --psm 3'
            text = pytesseract.image_to_string(pic, config=ocr_config).strip()
            
        if not text:
            raise ValueError("OCR returned empty text - check image quality")
        
        # For images, we treat the whole image as one "page"
        page_info = [{
            "page": 1,
            "text": text,
            "char_start": 0,
            "char_end": len(text)
        }]
        
        return text, page_info
    except Exception as e:
        raise RuntimeError(f"OCR Failed: {str(e)}")

def store_to_vector_db(text: str, metadata: Dict, collection_name: str):
    """Store text chunks with metadata"""
    chunk_size = 500
    overlap = 50
    chunks = []
    
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        if chunk.strip():
            chunks.append({
                "text": chunk.strip(),
                "char_start": i,
                "char_end": min(i + chunk_size, len(text))
            })
    
    if not chunks:
        raise ValueError("No valid text chunks to store")

    for chunk_info in chunks:
        embedding = embedder.encode(chunk_info["text"])
        point_id = str(uuid.uuid4())

        qdrant.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=point_id,
                    vector=embedding.tolist(),
                    payload={
                        'text': chunk_info["text"],
                        'char_start': chunk_info["char_start"],
                        'char_end': chunk_info["char_end"],
                        **metadata,
                        "chunk_id": point_id
                    }
                )
            ]
        )

@app.post("/upload")
async def upload_file(file: UploadFile):
    """Handle document upload and processing"""
    try:
        doc_id = str(uuid.uuid4())
        collection_name = f"documents_{doc_id}"

        # Create collection
        qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE),
        )

        # Save file
        file_path = Path(UPLOAD_DIR) / file.filename
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text with page information
        if file.filename.lower().endswith(".pdf"):
            text, page_info = extract_pdf_text(str(file_path))
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):  
            text, page_info = extract_image_text(str(file_path))
        else:
            raise ValueError("Unsupported file type")
        
        if not text.strip():
            raise ValueError("No text extracted")

        # Store in vector database
        store_to_vector_db(
            text=text,
            metadata={
                "doc_id": doc_id,
                "filename": file.filename,
                "filepath": str(file_path),
                "page": 1
            },
            collection_name=collection_name
        )

        return JSONResponse(
            content={
                "filename": file.filename, 
                "status": "processed",
                "doc_id": doc_id,
                "text_length": len(text),
                "pages": len(page_info)
            },
            status_code=200
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
@app.post("/search_multiple")
async def search_multiple_documents(request: SearchRequest):
    """Search across multiple documents with enhanced response format"""
    try:
        all_results = []
        individual_answers = []
        
        # Search each document collection
        for doc_id in request.doc_ids:
            collection_name = f"documents_{doc_id}"
            
            try:
                results = qdrant.search(
                    collection_name=collection_name,
                    query_vector=embedder.encode(request.query).tolist(),
                    limit=3
                )
                
                doc_results = []
                filename = "Unknown"
                
                # Collect results for this document
                for hit in results:
                    if hit.score > 0.3:  # Filter by relevance
                        filename = hit.payload.get("filename", "Unknown")
                        doc_results.append({
                            "text": hit.payload["text"],
                            "score": hit.score,
                            "filename": filename,
                            "doc_id": doc_id,
                            "page": hit.payload.get("page", 1),
                            "metadata": hit.payload
                        })
                        all_results.append(doc_results[-1])
                
                # Generate individual answer for this document
                if doc_results:
                    doc_contexts = [r["text"] for r in doc_results[:2]]
                    individual_answer = generate_answer(request.query, doc_contexts)
                    
                    individual_answers.append({
                        "doc_id": doc_id,
                        "filename": filename,
                        "extracted_answer": individual_answer,
                        "citation": f"Based on {len(doc_results)} relevant sections from {filename}",
                        "confidence": max(r["score"] for r in doc_results)
                    })
                        
            except Exception as e:
                print(f"Error searching document {doc_id}: {e}")
                continue
        
        if not all_results:
            return {
                "query": request.query,
                "individual_answers": [],
                "synthesized_answer": "No relevant information found across the uploaded documents.",
                "themes": [],
                "confidence_score": 0.0,
                "total_documents_searched": len(request.doc_ids),
                "documents_with_answers": 0
            }
        
        # Sort by relevance
        all_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Generate synthesized answer from all results
        top_contexts = [result["text"] for result in all_results[:10]]
        synthesized_answer = generate_answer(request.query, top_contexts)
        
        # Identify themes
        themes = identify_themes(all_results[:15], request.query)
        
        # Calculate confidence score
        avg_score = sum(r["score"] for r in all_results[:5]) / min(5, len(all_results))
        confidence_score = min(avg_score * 1.2, 1.0)  # Boost and cap at 1.0
        
        return {
            "query": request.query,
            "individual_answers": individual_answers,
            "synthesized_answer": synthesized_answer,
            "themes": themes,
            "confidence_score": confidence_score,
            "total_documents_searched": len(request.doc_ids),
            "documents_with_answers": len(individual_answers)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/documents")
async def list_documents():
    """List all uploaded documents"""
    try:
        # Get all collections
        collections = qdrant.get_collections()
        documents = []
        
        for collection in collections.collections:
            if collection.name.startswith("documents_"):
                doc_id = collection.name.replace("documents_", "")
                
                # Get sample data to extract metadata
                try:
                    sample_points = qdrant.scroll(
                        collection_name=collection.name,
                        limit=1
                    )[0]
                    
                    if sample_points:
                        point = sample_points[0]
                        documents.append({
                            "doc_id": doc_id,
                            "filename": point.payload.get("filename", "Unknown"),
                            "collection_name": collection.name,
                            "vector_count": collection.vectors_count or 0,
                            "points_count": collection.points_count or 0
                        })
                except Exception as e:
                    print(f"Error getting document info for {doc_id}: {e}")
                    continue
        
        return {"documents": documents, "total": len(documents)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")

@app.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a specific document and its collection"""
    try:
        collection_name = f"documents_{doc_id}"
        
        # Check if collection exists
        collections = qdrant.get_collections()
        collection_exists = any(c.name == collection_name for c in collections.collections)
        
        if not collection_exists:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete the collection
        qdrant.delete_collection(collection_name=collection_name)
        
        return {"message": f"Document {doc_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/generate_samples")
async def generate_sample_docs(request: SampleGenerationRequest):
    """Generate sample documents for testing"""
    try:
        documents = generate_sample_documents(request.num_docs, request.topics)
        
        return {
            "message": f"Generated {len(documents)} sample documents",
            "documents": documents,
            "topics_used": request.topics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate samples: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "embedder_model": "all-MiniLM-L6-v2",
        "vector_db": "qdrant",
        "llm_model": "llama3-70b-8192"
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Multi-Document Research Assistant",
        "version": "1.0.0",
        "description": "Upload multiple documents and query them with AI-powered analysis",
        "endpoints": {
            "POST /upload": "Upload a document (PDF or image)",
            "POST /search_multiple": "Search across multiple documents",
            "GET /documents": "List all uploaded documents",
            "DELETE /documents/{doc_id}": "Delete a specific document",
            "POST /generate_samples": "Generate sample documents for testing",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True
    )