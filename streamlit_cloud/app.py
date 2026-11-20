"""Streamlit Community Cloud entrypoint for the original Streamlit application.

Deployment: repo emotion-detector, branch main, file streamlit_cloud/app.py.
Original source is untouched in the repository root.
"""
from pathlib import Path
import runpy

root_app = Path(__file__).resolve().parents[1] / "app.py"
runpy.run_path(str(root_app), run_name="__main__")
