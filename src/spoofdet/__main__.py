"""`python -m spoofdet` runs the same command line as the console script."""

import sys

from spoofdet.cli import main

if __name__ == "__main__":
    sys.exit(main())
