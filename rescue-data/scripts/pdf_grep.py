"""Print sentences matching a regex in a PDF, with page numbers (reads untrusted data as text only)."""
import re, sys
from pypdf import PdfReader
pat = re.compile(sys.argv[2], re.I)
for i, page in enumerate(PdfReader(sys.argv[1]).pages, 1):
    text = re.sub(r"\s+", " ", page.extract_text() or "")
    for s in re.split(r"(?<=[.!?])\s", text):
        if pat.search(s):
            print(f"p{i}: {s[:400]}")
