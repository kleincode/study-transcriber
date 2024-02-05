#!/usr/bin/env python3
"""
Performs semantic search on a folder of PDF and TXT files.
"""
from argparse import ArgumentParser
from typing import List, Dict
from pathlib import Path
import os
from io import BytesIO
from itertools import chain
from langchain.docstore.document import Document
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from pypdf import PdfWriter, PdfReader
import textwrap
import subprocess
import sys
from fpdf import FPDF


class SemanticSearcher:
    def __init__(self) -> None:
        print("Initializing... ", end="", flush=True)
        self.embeddings = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-base-en-v1.5",
            model_kwargs={"device": "cuda"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self.store = Chroma(
            collection_name="full_documents", embedding_function=self.embeddings
        )
        self.doc_store = InMemoryStore()
        self.full_doc_retriever = ParentDocumentRetriever(
            vectorstore=self.store,
            docstore=self.doc_store,
            child_splitter=RecursiveCharacterTextSplitter(chunk_size=400),
            parent_splitter=RecursiveCharacterTextSplitter(
                chunk_size=2000, add_start_index=True
            ),
            search_kwargs={"k": 10},
        )
        print("Done!")

    def index(self, file_path: Path) -> None:
        print(f"Indexing {file_path}")
        if file_path.suffix == ".pdf":
            self._index_pdf(file_path)
        elif file_path.suffix == ".txt":
            self._index_txt(file_path)

    def _index_pdf(self, file_path: Path) -> None:
        loader = PyPDFLoader(str(file_path), extract_images=False)
        pages = loader.load_and_split()
        preprocessed_pages = []
        page_count = 0

        def perform_add():
            nonlocal preprocessed_pages, page_count
            self.full_doc_retriever.add_documents(preprocessed_pages, ids=None)
            page_count += len(preprocessed_pages)
            preprocessed_pages.clear()
            print(f"Added {page_count} meaningful pages to the index...")

        for page in pages:
            content = page.page_content.replace("PFIWiSe21/22 TEIL I I", "").replace(
                "PSYCHOLOGIE FÜR INGENIEURINNEN  UND INGENIEURE (TEIL I I)", ""
            )
            if len(content) > 120:
                preprocessed_pages.append(
                    Document(page_content=content, metadata=page.metadata)
                )
            if len(preprocessed_pages) >= 100:
                perform_add()
        perform_add()
        print(f"Added all {page_count} relevant pages of {file_path}.")

    def _index_txt(self, file_path: Path) -> None:
        def remove_timestamps(line: str):
            end_of_timestamp = line.find("] ")
            if end_of_timestamp != -1:
                line = line[end_of_timestamp + 1 :]
            return line.strip()

        with open(file_path, "r", encoding="utf-8") as f:
            content = " ".join([remove_timestamps(line) for line in f.readlines()])
        content = content.replace(". ", ".\n")

        print("Adding 1 meaningful page to the index...")
        self.full_doc_retriever.add_documents(
            [
                Document(
                    page_content=content, metadata={"source": str(file_path), "page": 0}
                )
            ],
            ids=None,
        )

    def search(self, query: str) -> List[Document]:
        """Search for documents relevant to a query.
        Args:
            query: String to find relevant documents for
        Returns:
            List of relevant documents
        """
        return self.full_doc_retriever.get_relevant_documents(query)


class PDFCreator:
    def __init__(
        self, results: List[Document], txt_write_full_file: bool = False
    ) -> None:
        self.writer = PdfWriter()
        self.readers: Dict[str, PdfReader] = {}
        for result in results:
            page = int(result.metadata["page"])
            source_file = result.metadata["source"]
            if source_file.endswith(".pdf"):
                if source_file not in self.readers:
                    self.readers[source_file] = PdfReader(source_file)
                self.writer.add_page(self.readers[source_file].pages[page])
            elif source_file.endswith(".txt"):
                text = (
                    "\n".join(
                        f"{key.capitalize()}: {val}"
                        for key, val in result.metadata.items()
                    )
                    + "\n\n"
                )
                if txt_write_full_file:
                    with open(source_file, "r", encoding="utf-8") as f:
                        text += f.read()
                else:
                    text += result.page_content
                pdf_bytes = self._text_to_pdf(text)
                reader = PdfReader(BytesIO(pdf_bytes))
                self.writer.append_pages_from_reader(reader)
            else:
                print("Unsupported file type:", source_file)

    def write(self, output_path: Path, open_file: bool = False) -> None:
        self.writer.write(output_path)
        if open_file:
            if sys.platform == "darwin":
                subprocess.call(["open", output_path])
            else:
                os.startfile(output_path)

    def _text_to_pdf(self, text: str) -> bytearray:
        a4_width_mm = 210
        pt_to_mm = 0.35
        fontsize_pt = 10
        fontsize_mm = fontsize_pt * pt_to_mm
        margin_bottom_mm = 10
        character_width_mm = 7 * pt_to_mm
        width_text = a4_width_mm / character_width_mm

        pdf = FPDF(orientation="P", unit="mm", format="A4")
        pdf.set_auto_page_break(True, margin=margin_bottom_mm)
        pdf.add_page()
        pdf.set_font(family="Courier", size=fontsize_pt)
        splitted = text.split("\n")

        for line in splitted:
            lines = textwrap.wrap(line, width_text)

            if len(lines) == 0:
                pdf.ln()

            for wrap in lines:
                pdf.cell(0, fontsize_mm, wrap, ln=1)

        return pdf.output(dest="S").encode("latin-1")


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "input_folder", type=str, help="Input folder with pdf files to search"
    )
    parser.add_argument("--types", nargs="+", default=["pdf", "txt"])
    args = parser.parse_args()

    searcher = SemanticSearcher()
    input_folder = Path(args.input_folder)

    for file_path in chain(*[input_folder.glob(f"*.{t}") for t in args.types]):
        searcher.index(file_path)

    while True:
        try:
            query = input("Query > ").strip()
            if query.startswith('"""'):
                query = query[3:]
                while True:
                    query += "\n" + input("... ").strip()
                    if query.endswith('"""'):
                        query = query[:-3]
                        break
            if len(query.strip()) == 0:
                continue
            if query.lower() == "exit":
                break
            results = searcher.search(query)
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, start=1):
                print("#" * 64)
                print(f"# RESULT {i:02d}")
                print(
                    f"# Source: {Path(result.metadata['source']).relative_to(input_folder)}, page {result.metadata['page']}"
                )
                print("#" * 64)
                print()
                print(result.page_content)
                print()
            print()
            PDFCreator(results).write(Path("results.pdf"), open_file=True)
        except KeyboardInterrupt:
            print()
            if input("Do you really want to exit? (y/n)").lower().strip() == "y":
                break
        except Exception as e:
            print("An error occurred:", e)

    print("Bye!")


if __name__ == "__main__":
    main()
