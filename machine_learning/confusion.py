from __future__ import annotations

import os
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config import PathResolver
from pipeline import DatasetLoader


class ConfusionMatrixGenerator:
    def __init__(self) -> None:
        self.config = PathResolver.create_config()
        self.dataset_loader = DatasetLoader(self.config)
        self.output_dir = self.config.model_dir / "images"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self, model_filename: str):
        model_path = self.config.model_dir / f"{model_filename}.pkl"
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        return joblib.load(model_path), model_path

    def _predict(self, model, X_test: pd.Series) -> pd.Series:
        try:
            y_pred = model.predict(X_test)
            return pd.Series(y_pred)
        except Exception:
            try:
                from pycaret.classification import predict_model
            except Exception as exc:
                raise RuntimeError(
                    "Model prediction failed. If this is a PyCaret model, install/import "
                    "PyCaret in this environment."
                ) from exc

            prediction_df = predict_model(model, data=pd.DataFrame({"text": X_test}))
            if "prediction_label" not in prediction_df.columns:
                raise RuntimeError("PyCaret prediction output missing 'prediction_label' column.")
            return prediction_df["prediction_label"]

    def _prepare_split(self) -> tuple[pd.Series, pd.Series]:
        data = self.dataset_loader.load()
        _, X_test, _, y_test = train_test_split(
            data["text"],
            data["label"],
            test_size=1 - self.config.train_size,
            random_state=self.config.session_id,
            stratify=data["label"],
        )
        return X_test, y_test

    def _save_confusion_image(
        self, y_true: pd.Series, y_pred: pd.Series, model_code: str, model_path: Path
    ) -> Path:
        labels = ["ham", "spam"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format="d")
        plt.title(f"Confusion Matrix - {model_code.upper()}")
        plt.tight_layout()

        output_path = self.output_dir / f"confusion_{model_code}.png"
        plt.savefig(output_path, dpi=150)
        plt.close()

        print(f"Model loaded: {model_path}")
        print(f"Saved confusion matrix image: {output_path}")
        return output_path

    def run(self) -> None:
        X_test, y_test = self._prepare_split()
        for job in self.config.jobs:
            model, model_path = self._load_model(job.model_filename)
            y_pred = self._predict(model, X_test)
            self._save_confusion_image(
                y_true=y_test.reset_index(drop=True),
                y_pred=y_pred.reset_index(drop=True),
                model_code=job.model_code,
                model_path=model_path,
            )


def main() -> None:
    generator = ConfusionMatrixGenerator()
    generator.run()


if __name__ == "__main__":
    main()
