# Study Transcriber
This is a work in progress. The goal is to automatically transcribe lecture videos using Whisper, improve the transcripts using some local LLM, and embed the results into a vector database. This can then be used to easily access lecture contents for review.

## Install packages:
```bash
# Pytorch, e.g.:
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install langchain pypdf fpdf sentence_transformers chromadb cryptography==3.1
```

## Run semantic search
```bash
python ./semantic_search.py [OPTIONS] ./path/to/input_folder/
```
**Options:**
- `--types [type1] [type2] [...]`: File types to index. The only supported types are pdf, txt. Default is `pdf txt`.
- `--lang [language]`: Language for stopword removal in the query. This is only relevant for the highlighting. Use the full language name, like `english`, `german`, etc. Supported languages: arabic, azerbaijani, basque, bengali, catalan, chinese, danish, dutch, english, finnish, french, german, greek, hebrew, hinglish, hungarian, indonesian, italian, kazakh, nepali, norwegian, portuguese, romanian, russian, slovene, spanish, swedish, tajik, turkish.
- `--print`: Also print the results to stdout.

Note that highlighting is always done in green color (as yellow is too mainstream) and is still experimental.