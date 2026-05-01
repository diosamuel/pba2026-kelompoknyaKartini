import os
import sys

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import PathResolver
from trainers import TrainingApplication


def main() -> None:
    app = TrainingApplication(config=PathResolver.create_config())
    app.run()

if __name__ == "__main__":
    main()
