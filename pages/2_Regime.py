import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from app_shared import setup_page, _render_regime

setup_page("regime", "Regime · Screener")
_render_regime()
