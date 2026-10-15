"""Run one adapter in its declared Python environment, isolating backbone imports."""

import os, sys
from .io import read
from .stages import execute

if __name__ == "__main__":
    execute(read(sys.argv[1] if len(sys.argv) > 1 else os.environ["EVO_WAM_CONTEXT"]))
