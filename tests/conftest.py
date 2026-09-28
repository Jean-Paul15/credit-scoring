from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TEST_RUNTIME_ROOT = Path(tempfile.mkdtemp(prefix="credit-risk-tests-"))

os.environ.setdefault("CREDIT_RISK_DATA_DIR", str(TEST_RUNTIME_ROOT / "data"))
os.environ.setdefault("CREDIT_RISK_ARTIFACTS_DIR", str(TEST_RUNTIME_ROOT / "artifacts"))

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
