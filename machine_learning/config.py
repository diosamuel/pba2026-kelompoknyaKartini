from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TrainingJob:
    model_code: str
    model_filename: str


@dataclass(frozen=True)
class AppConfig:
    dataset_path: Path
    model_dir: Path
    word2vec_model_path: Path
    text_column: str = "Pesan"
    label_column: str = "Kategori"
    train_size: float = 0.8
    session_id: int = 42
    jobs: tuple[TrainingJob, ...] = (
        TrainingJob("lr", "spam_model_lr"),
        TrainingJob("nb", "spam_model_nb"),
        TrainingJob("svm", "spam_model_svm"),
    )

    @property
    def allowed_labels(self) -> tuple[str, str]:
        return ("spam", "ham")

    @property
    def pycaret_include_models(self) -> list[str]:
        return [job.model_code for job in self.jobs]


class PathResolver:
    @staticmethod
    def running_in_colab() -> bool:
        return "COLAB_RELEASE_TAG" in os.environ

    @classmethod
    def create_config(cls) -> AppConfig:
        if cls.running_in_colab():
            dataset_path = Path("/content/email_spam_indo.csv")
            model_dir = Path("/content/model")
            word2vec_model_path = Path("/content/w2v_kamus.model")
        else:
            project_root = Path(__file__).resolve().parents[1]
            dataset_path = project_root / "dataset" / "email_spam_indo.csv"
            model_dir = project_root / "machine_learning" / "model"
            word2vec_model_path = project_root / "machine_learning" / "w2v_kamus.model"
        return AppConfig(
            dataset_path=dataset_path,
            model_dir=model_dir,
            word2vec_model_path=word2vec_model_path,
        )
