from html import unescape
from html.parser import HTMLParser
import re

class HTMLTextExtractor(HTMLParser):
    """Extract visible text from an HTML document while ignoring script/style blocks."""

    def __init__(self):
        # Initialize the parent HTML parser and keep a list of text fragments.
        super().__init__()
        self._parts = []
        # Track nested script/style tags so their content is not treated as visible text.
        self._skip_depth = 0

    def handle_starttag(self, tag, attrs):
        # Skip JavaScript and CSS content because it usually is not user-visible text.
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag):
        # Decrease the skip counter when the script/style block closes.
        if tag in {"script", "style"} and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data):
        # Keep only meaningful text outside script/style nodes.
        if self._skip_depth == 0 and data.strip():
            self._parts.append(data)

    def get_text(self):
        # Join the text fragments, decode HTML entities, and normalize spacing.
        text = " ".join(self._parts)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text