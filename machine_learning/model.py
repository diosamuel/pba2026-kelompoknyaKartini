from config import PathResolver
from trainers import TrainingApplication


def main() -> None:
    app = TrainingApplication(config=PathResolver.create_config())
    app.run()

if __name__ == "__main__":
    main()
