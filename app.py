"""Root entry point for the Streamlit demo.

Run from the project root with: streamlit run app.py
"""
from pathlib import Path
import runpy

dashboard_script = Path(__file__).resolve().parent / "dashboard" / "app.py"
runpy.run_path(str(dashboard_script), run_name="__main__")
