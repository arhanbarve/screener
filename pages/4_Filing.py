import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app_shared import setup_page, _render_filing_edge

setup_page("filing", "Filing Edge · Screener")
_render_filing_edge()
