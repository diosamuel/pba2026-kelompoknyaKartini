from __future__ import annotations

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import AppConfig, TrainingJob
from pipeline import DatasetLoader, Word2VecVectorizer

try:
    from pycaret.classification import (
        compare_models,
        create_model,
        finalize_model,
        pull,
        save_model,
        setup,
    )

    PYCARET_AVAILABLE = True
    PYCARET_IMPORT_ERROR: Exception | None = None
except Exception as pycaret_error:
    PYCARET_AVAILABLE = False
    PYCARET_IMPORT_ERROR = pycaret_error
    compare_models = None
    create_model = None
    finalize_model = None
    pull = None
    save_model = None
    setup = None


class BaseTrainer:
    def __init__(self, config: AppConfig, data: pd.DataFrame) -> None:
        self.config = config
        self.data = data
        self.config.model_dir.mkdir(parents=True, exist_ok=True)

    def train_all(self) -> None:
        raise NotImplementedError


class PyCaretTrainer(BaseTrainer):
    def _setup(self) -> None:
        setup(
            data=self.data,
            target="label",
            text_features=["text"],
            train_size=self.config.train_size,
            session_id=self.config.session_id,
            verbose=False,
            html=False,
        )

    def create_comparison_table(self) -> None:
        self._setup()
        compare_models(include=self.config.pycaret_include_models, sort="F1", verbose=False)
        comparison_table = pull()
        print("\nPyCaret comparison table (LR, NB, SVM):")
        print(comparison_table)

    def train_model(self, job: TrainingJob) -> None:
        self._setup()
        model = create_model(job.model_code, verbose=False)
        metrics = pull()
        print(f"\n{job.model_code.upper()} validation metrics:")
        print(metrics)

        final_model = finalize_model(model)
        save_model(final_model, str(self.config.model_dir / job.model_filename))

    def train_all(self) -> None:
        self.create_comparison_table()
        for job in self.config.jobs:
            self.train_model(job)


class SklearnTrainer(BaseTrainer):
    def _build_pipeline(self, model_code: str) -> Pipeline:
        if model_code == "lr":
            classifier = LogisticRegression(max_iter=1000, random_state=self.config.session_id)
        elif model_code == "nb":
            classifier = GaussianNB()
        elif model_code == "svm":
            classifier = LinearSVC(random_state=self.config.session_id)
        else:
            raise ValueError(f"Unsupported model code: {model_code}")

        return Pipeline(
            [
                ("word2vec", Word2VecVectorizer(self.config.word2vec_model_path)),
                ("classifier", classifier),
            ]
        )

    def train_model(self, job: TrainingJob) -> None:
        X_train, X_test, y_train, y_test = train_test_split(
            self.data["text"],
            self.data["label"],
            test_size=1 - self.config.train_size,
            random_state=self.config.session_id,
            stratify=self.data["label"],
        )

        model = self._build_pipeline(job.model_code)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(classification_report(y_test, y_pred))

        output_path = self.config.model_dir / f"{job.model_filename}.pkl"
        joblib.dump(model, output_path)

    def train_all(self) -> None:
        for job in self.config.jobs:
            self.train_model(job)


class TrainerFactory:
    @staticmethod
    def create(config: AppConfig, data: pd.DataFrame) -> BaseTrainer:
        if PYCARET_AVAILABLE:
            return PyCaretTrainer(config=config, data=data)
        return SklearnTrainer(config=config, data=data)


class TrainingApplication:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.loader = DatasetLoader(config)

    def run(self) -> None:
        data = self.loader.load()
        trainer = TrainerFactory.create(config=self.config, data=data)
        if isinstance(trainer, PyCaretTrainer):
            print("Using PyCaret training pipeline.")
        else:
            print(f"PyCaret import error: {PYCARET_IMPORT_ERROR}")
        trainer.train_all()
