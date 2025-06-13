import streamlit as st
import requests
from PIL import Image
import pytesseract
import io
import os 
import zipfile
from pathlib import Path 
import pandas as pd

# Backend Connection with the API key 
BACKEND_URL = "http://127.0.0.1:8000"

# Streamlit Setup of the page 
st.set_page_config(
    page_title="Mr.Document Chat A.I.",
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'id_doc' not in st.session_state:
    st.session_state.id_doc = []
if 'loading_progression' not in st.session_state:
    st.session_state.loading_progression = {}
if 'loading_progression' not in st.session_state:
    st.session_state.loading_progression = {}

# Page Title Rendering 
st.title("Ask Your File - Ask across all files!")

# Tabs for upload and Camera Capture of the document 
tab1, tab2, tab3, tab4 = st.tabs(["📁 Upload Files", "📸 Camera Capture", "Mock files", "File Organizer"])


# Add this with your other session state initializations
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = set()

# Tab 1: multiple files Upload function
with tab1:
    st.header("📁 Upload Document")
    st.markdown("Upload multiple documents at once (PDF, PNG, JPG, JPEG)")

    file_folder = st.file_uploader(
        "Select multiple documents", 
        type=["pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        help="You can select multiple files at once"
    )

    if file_folder:
        st.write(f"Selected {len(file_folder)} files:")
        
        # Check for new files that haven't been processed
        new_files = []
        for file in file_folder:  # Fixed: was 'files', now 'file'
            # Create unique identifier for each file
            file_id = f"{file.name}_{file.size}_{hash(file.getvalue())}"  # Fixed: use 'file' not 'file_folder'
            if file_id not in st.session_state.processed_files:
                new_files.append((file, file_id))
            
            # Show file button (for display purposes)
            st.write(f"- {file.name} ({file.type})")  # Changed from button to write for better UX
        
        if new_files:
            st.write(f"Processing {len(new_files)} new files...")
            
            loading_progression = st.progress(0)
            loading_status = st.empty()

            for i, (file, file_id) in enumerate(new_files):  # Process only new files
                try:
                    loading_status.text(f"Processing {file.name}...")

                    # Show preview for images
                    if file.type.startswith("image/"):
                        col1, col2 = st.columns([1, 3])
                        with col1:
                            st.image(file, caption=file.name, width=150)
                    
                    # Upload file
                    folder_payload = {"file": (file.name, file, file.type)}
                    response = requests.post(f"{BACKEND_URL}/upload", files=folder_payload)

                    if response.status_code == 200:
                        answer = response.json()
                        
                        file_information = {
                            'filename': file.name,
                            'doc_id': answer.get('doc_id'),
                            'status': 'success',
                            'text_length': answer.get('text_length', 0)
                        }
                    
                        st.session_state.id_doc.append(file_information)
                        st.session_state.processing_status[file.name] = 'Successful!'
                        
                        # Mark file as processed
                        st.session_state.processed_files.add(file_id)
                    else:
                        st.session_state.processing_status[file.name] = f'Failure: {response.text}'

                    # Update progress
                    loading_progression.progress((i + 1) / len(new_files))

                except Exception as wrong:
                    st.session_state.loading_progression[file.name] = f'Error: {wrong}'

            loading_status.text("✅ Processing complete!")
            st.success(f"Processed {len(new_files)} new documents!")
            
        else:
            st.info("All selected files have already been processed!")

if st.button("Clear All", type="secondary"):
    st.session_state.id_doc = []
    st.session_state.loading_progression = {}
    st.session_state.processed_files = set()  # Clear processed files tracking
    st.rerun()

# For the Camera's capturing functionality tab 
with tab2:
    st.header("Snap a Pic")
    pic_taken = st.camera_input("Capture it now!")

    if pic_taken is not None:
        st.image(pic_taken, caption="Captured Image", width=300)

        if st.button("Process the Snap"):
            with st.spinner("Processing Captured image..."):
                
                
                try:
                
                    pic = Image.open(pic_taken)
                    pic_buffer = io.BytesIO()
                    pic.save(pic_buffer, format='PNG')
                    pic_buffer.seek(0)

                    Pic_files = {
                        "file": ("captured_snap.png", pic_buffer, "image/png")
                    }

                    # Send the picture to the backend/main.py
                    response = requests.post(f"{BACKEND_URL}/upload", files=Pic_files)

                    if response.status_code == 200:
                        answer = response.json()
                        file_information = {
                            'filename': 'captured_snap.png',
                            'doc_id': answer.get('doc_id'),
                            'status': 'success',
                            'text_length': answer.get('text_length', 0)
                        }
                        st.session_state.id_doc.append(file_information)
                        st.success("Snap is successfully processed")
                    else:
                        st.error(f"Snap Failure: {response.text}")

                except Exception as wrong:
                    st.error(f"Complete failure: {wrong}")

# Tab 3: Documents Generator                 
with tab3: 
    st.header("Mock files")
    st.markdown("**Generating Document sample**")

    
    
    col1, col2 = st.columns(2)

    
    
    with col1:
        doc_nos = st.slider("Number of Documents", 10, 100, 25)
        file_titles = st.multiselect(
            "Select topics for documents",
            ["Technology", "Healthcare", "Finance", "Education", "Environment", "Sports", "Politics", "Science"],
            default=["Technology", "Healthcare", "Finance"]
        )

    
    
    with col2:
        if st.button("Documents Generator", type="secondary"):
            with st.spinner(f"Generating {doc_nos} example Files..."):
                try:
                    sol = requests.post(f"{BACKEND_URL}/generate_samples",
                        json={"num_docs": doc_nos, "topics": file_titles}
                    )

                    if sol.status_code == 200:
                        answer = sol.json()
                        for doc in answer.get('documents', []):
                            st.session_state.id_doc.append(doc)
                        st.success(f"Created {len(answer.get('documents', []))} example files!")
                        st.rerun()
                    else:
                        st.error("Failed to generate files")
                except Exception as wrong:
                    st.error(f"Error in generating examples: {wrong}")



# Tab 4: Document Manager
with tab4:
    st.header("File Organizer")

    
    if st.session_state.id_doc:
        st.subheader(f"Uploaded Files ({len(st.session_state.id_doc)})")

        # Dataframe Creation for display
        df_data = []
        for doc in st.session_state.id_doc:
            df_data.append({
                'Filename': doc['filename'],
                'Status': doc['status'],
                'Text Length': doc.get('text_length', 'N/A'),
                'Doc ID': doc['doc_id'][:8] + '...' if doc.get('doc_id') else 'N/A'
            })

        df = pd.DataFrame(df_data)
        st.dataframe(df, use_container_width=True)

        # Clear all files and documents button
        if st.button("Clear All", type="secondary", key="clear_all_global"):
            st.session_state.id_doc = []
            st.session_state.loading_progression = {}
            st.rerun()
    
    
    else:
        st.info("No documents uploaded yet. Use the other tabs to upload documents.")

    
    
    # Question & answering interface
    st.markdown("---")
    if st.session_state.id_doc:
        st.header("ASK ME ANYTHING ACROSS ANY FILES!")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Ask me anything about your files",
                                 placeholder="What are the main points in all documents?")
            
        with col2:
            find_mode = st.selectbox("Search Mode", ["All Documents", "Smart Themes", "Detailed Citations"])

        if st.button("🔍 Search & Analyze", type="primary") and query:
            with st.spinner("Analyzing across all documents..."):
                try:
                    # Get all doc_ids
                    doc_ids = [doc['doc_id'] for doc in st.session_state.id_doc if doc.get('doc_id')]
                    
                    # Search across all documents
                    response = requests.post(
                        f"{BACKEND_URL}/search_multiple",
                        json={
                            "query": query,
                            "doc_ids": doc_ids,
                            "mode": find_mode.lower().replace(" ", "_")
                        },
                        timeout=60
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display results in the required format
                        st.markdown("---")
                        
                        # PART 1: Individual Document Answers (TABLE FORMAT)
                        st.subheader("📋 Individual Document Answers")
                        
                        individual_answers = result.get("individual_answers", [])
                        if individual_answers:
                            # Create the required table format
                            table_data = []
                            for answer in individual_answers:
                                table_data.append({
                                    "Document ID": answer["doc_id"][:8] + "...",
                                    "Document Name": answer["filename"],
                                    "Extracted Answer": answer["extracted_answer"],
                                    "Citation": answer["citation"],
                                    "Confidence": f"{answer['confidence']:.2f}"
                                })
                            
                            # Display as DataFrame
                            df_answers = pd.DataFrame(table_data)
                            st.dataframe(df_answers, use_container_width=True)
                            
                            # Download option for the table
                            csv = df_answers.to_csv(index=False)
                            st.download_button(
                                label="📥 Download Individual Answers (CSV)",
                                data=csv,
                                file_name=f"individual_answers_{query[:20]}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.info("No individual answers found from the documents.")
                        
                        st.markdown("---")
                        
                        # PART 2: Theme Analysis & Synthesized Answer (CHAT FORMAT)
                        st.subheader("🧠 Synthesized Analysis")
                        
                        # Display synthesized answer
                        synthesized = result.get("synthesized_answer", "No synthesis available")
                        st.success(f"**Comprehensive Answer:** {synthesized}")
                        
                        # Display themes in chat-style format
                        themes = result.get("themes", [])
                        if themes:
                            st.subheader("🎯 Identified Themes")
                            for i, theme in enumerate(themes, 1):
                                with st.expander(f"Theme {i}: {theme.get('title', 'Unnamed Theme')}"):
                                    st.write(f"**Description:** {theme.get('description', 'No description')}")
                                    
                                    supporting_docs = theme.get('supporting_docs', [])
                                    if supporting_docs:
                                        st.write(f"**Supporting Documents:** {', '.join(supporting_docs[:5])}")
                                        if len(supporting_docs) > 5:
                                            st.write(f"*...and {len(supporting_docs) - 5} more documents*")
                        
                        # Metadata
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Documents Searched", result.get("total_documents_searched", 0))
                        with col2:
                            st.metric("Documents with Answers", result.get("documents_with_answers", 0))
                        with col3:
                            confidence = result.get("confidence_score", 0)
                            st.metric("Overall Confidence", f"{confidence:.1%}")
                            
                    else:
                        st.error(f"Search failed: {response.text}")
                        
                except Exception as e:
                    st.error(f"Search error: {str(e)}")
    else:
        st.info("Please upload documents first to start asking questions!")

# Sidebar with status
with st.sidebar:
    st.header("📊 System Status")
    
    # Document count
    doc_count = len(st.session_state.id_doc)
    if doc_count >= 75:
        st.success(f"✅ {doc_count} documents loaded (Target: 75+ ✓)")
    elif doc_count >= 10:
        st.warning(f"⚠️ {doc_count} documents loaded (Target: 75+)")
    else:
        st.error(f"❌ {doc_count} documents loaded (Target: 75+)")

    # Processing status
    if st.session_state.loading_progression:
        st.subheader("Last Processing Status")
        for filename, status in list(st.session_state.loading_progression.items())[-5:]:
            st.text(f"{filename[:20]}... {status}")

    # Quick stats
    if st.session_state.id_doc:
        total_chars = sum(doc.get('text_length', 0) for doc in st.session_state.id_doc)
        st.metric("Total Text Length", f"{total_chars:,} chars")

        success_count = sum(1 for doc in st.session_state.id_doc if doc['status'] == 'success')
        st.metric("Successfully Processed", f"{success_count}/{doc_count}")

st.markdown("""
    <style>
        .stButton > button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            font-weight: bold;
            border: none;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #45a049;
            transform: translateY(-2px);
        }
        
        .stButton > button[kind="primary"] {
            background-color: #2196F3;
        }
        
        .stButton > button[kind="primary"]:hover {
            background-color: #1976D2;
        }
        
        .metric-container {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin: 1rem 0;
        }
        
        .stProgress > div > div > div > div {
            background-color: #4CAF50;
        }
    </style>
""", unsafe_allow_html=True)

# The Footer 
st.markdown("""
    <div style='text-align: center; padding: 20px 0; font-size: 14px; color: gray;'>
        © 2025 Srivathsan | Document Research Chatbot | All Rights Reserved | Experimental UI <br>
        Built with Tomorrow's Intelligence for the Future of Documents
    </div>
""", unsafe_allow_html=True)