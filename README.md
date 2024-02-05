# Study Transcriber
This is a work in progress. The goal is to automatically transcribe lecture videos using Whisper, improve the transcripts using some local LLM, and embed the results into a vector database. This can then be used to easily access lecture contents for review.

## Install packages:
```bash
# Pytorch, e.g.:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install langchain pypdf fpdf sentence_transformers chromadb cryptography==3.1
```

## Run the code (Example)
```bash
python ./semantic_search.py ./path/to/input_folder/
```