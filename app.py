import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from app_shared import setup_page, _render_screener

setup_page("screener", "Screener")
_render_screener()
