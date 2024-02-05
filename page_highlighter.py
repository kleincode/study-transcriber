from typing import List, Tuple
import re
from pypdf import PdfWriter
from pypdf.annotations import Highlight
from pypdf.generic import ArrayObject, FloatObject

Target = Tuple[float, float, float, float]


def get_stopwords(language: str) -> List[str]:
    import nltk
    from nltk.corpus import stopwords

    nltk.download("stopwords", quiet=True)
    if language not in stopwords._fileids:
        print("WARNING: Language not supported")
        language = "english"
    print(f"Using {language} stopwords")
    return stopwords.words(language)


class PageHighlighter:
    def __init__(
        self,
        writer: PdfWriter,
        query: str,
        min_matches: int | None = None,
        language: str = "german",
    ) -> None:
        self.writer = writer
        stopwords = get_stopwords(language)
        self.keywords = set(
            [
                w.strip().lower()
                for w in re.split(r"\b", query)
                if len(w.strip()) > 3 and w.strip().lower() not in stopwords
            ]
        )
        print("Highlighting keywords:", self.keywords)
        if min_matches is None:
            self.min_matches = 1 if len(self.keywords) < 4 else 2
        else:
            self.min_matches = min_matches
        self.targets: List[Target] = []

    def _check_for_keywords(self, text: str) -> bool:
        words = set(
            [w.strip().lower() for w in re.split(r"\b", text) if len(w.strip()) > 3]
        )
        hits = len(words.intersection(self.keywords))
        return hits >= self.min_matches

    def _visitor_body(self, text: str, cm, tm, font_dict, font_size: int):
        if self._check_for_keywords(text):
            self.targets.append(
                (tm[4], tm[5] - 2, 0.45 * font_size * len(text), font_size)
            )

    def _add_rect(self, target: Target):
        x, y, w, h = target
        rect = (x, y, x + w, y + h)
        quad_points = [
            rect[0],
            rect[1],
            rect[2],
            rect[1],
            rect[0],
            rect[3],
            rect[2],
            rect[3],
        ]

        # Add the highlight
        annotation = Highlight(
            rect=rect,
            quad_points=ArrayObject(
                [FloatObject(quad_point) for quad_point in quad_points]
            ),
            highlight_color="#00ff00",
        )
        self.writer.add_annotation(
            page_number=len(self.writer.pages) - 1, annotation=annotation
        )

    def highlight(self) -> None:
        self.targets.clear()
        page = self.writer.pages[-1]
        page.extract_text(visitor_text=self._visitor_body)
        for target in self.targets:
            self._add_rect(target)
