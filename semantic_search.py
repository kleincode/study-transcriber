from argparse import ArgumentParser
from typing import List
from pathlib import Path
import os
from langchain.docstore.document import Document
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain.retrievers import ParentDocumentRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.storage import InMemoryStore
from langchain.document_loaders import PyPDFLoader
from pypdf import PdfMerger
from pypdf.pagerange import PageRange


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
        )
        print("Done!")

    def index(self, file_path: Path) -> None:
        print(f"Indexing {file_path}")
        loader = PyPDFLoader(str(file_path), extract_images=False)
        pages = loader.load_and_split()
        preprocessed_pages = []
        for page in pages:
            content = page.page_content.replace("PFIWiSe21/22 TEIL I I", "").replace(
                "PSYCHOLOGIE FÜR INGENIEURINNEN  UND INGENIEURE (TEIL I I)", ""
            )
            if len(content) > 120:
                preprocessed_pages.append(
                    Document(page_content=content, metadata=page.metadata)
                )

        print(f"Adding {len(preprocessed_pages)} meaningful pages to the index...")
        self.full_doc_retriever.add_documents(preprocessed_pages, ids=None)

    def search(self, query: str) -> List[Document]:
        """Search for documents relevant to a query.
        Args:
            query: String to find relevant documents for
        Returns:
            List of relevant documents
        """
        return self.full_doc_retriever.get_relevant_documents(query)


class PDFCreator:
    def __init__(self, results: List[Document]) -> None:
        self.merger = PdfMerger()
        for result in results:
            page = int(result.metadata["page"])
            self.merger.append(result.metadata["source"], pages=(page, page + 1))

    def write(self, output_path: Path, open_file: bool = False) -> None:
        self.merger.write(output_path)
        if open_file:
            os.startfile(output_path)


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "input_folder", type=str, help="Input folder with pdf files to search"
    )
    args = parser.parse_args()

    searcher = SemanticSearcher()
    input_folder = Path(args.input_folder)

    for file_path in input_folder.glob("*.pdf"):
        searcher.index(file_path)

    while True:
        query = input("Query > ").strip()
        if len(query) == 0 or query.lower() == "exit":
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

    print("Bye!")


if __name__ == "__main__":
    main()
