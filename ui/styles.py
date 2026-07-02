"""Load the Streamlit custom CSS."""

from pathlib import Path

import streamlit as st

CSS_PATHS = [
    Path("ui/styles/main.css"),
    Path("main.css"),
]

BASE_PREPAINT_CSS = """
<style>
html,
body,
#root,
.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] > .main {
    background: #0f1117 !important;
    color: #ffffff !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
[data-testid="stDecoration"] {
    background: transparent !important;
}
</style>
"""


def find_css_path():
    # First existing CSS path wins.
    for css_path in CSS_PATHS:
        if css_path.exists():
            return css_path

    return None


@st.cache_data(show_spinner=False)
def read_css_text(path_text, modified_ns):
    # Cache CSS text so every button rerun does not hit the file system.
    _ = modified_ns
    return Path(path_text).read_text(encoding="utf-8")


def load_css():
    # Inject dark prepaint first, then cached app CSS.
    css_path = find_css_path()

    if not css_path:
        st.warning("Custom CSS file was not found.")
        return

    try:
        modified_ns = css_path.stat().st_mtime_ns
    except OSError:
        modified_ns = 0

    css_text = read_css_text(str(css_path), modified_ns)

    st.markdown(
        BASE_PREPAINT_CSS + f"<style>{css_text}</style>",
        unsafe_allow_html=True,
    )
