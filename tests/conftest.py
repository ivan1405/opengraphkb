import os
import sys

import pytest

# Ensure src/ is on the Python path for test discovery
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
