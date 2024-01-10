import gdown
import hydra
import mlflow
import pandas as pd
from omegaconf import OmegaConf
from onnxmltools.convert import convert_sklearn
from onnxmltools.utils import save_model
from skl2onnx.common.data_types import FloatTensorType as ftp
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from utils import convert_to_num_data, create_folder


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: OmegaConf):
    # Создание папки
    create_folder("data")
    # Скачивание датасета с gdrive
    gdown.download(config.link.train_data, config.path.train_data, quiet=False)
    df = pd.read_csv("data/train_data.csv")
    # df = load_data(config.dvc.path, config.dvc.remote)

    # Преобразование текстовых данных в числовые
    tfidf = convert_to_num_data(df)

    mlflow.set_tracking_uri(uri=config.mlflow.uri)

    # Создание пайплайна
    pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy=config.imputer.strategy)),
            ("scaler", StandardScaler(with_mean=config.scaler.with_mean)),
            (
                "classifier",
                DecisionTreeClassifier(
                    max_depth=config.classifier.max_depth,
                    random_state=config.classifier.random_state,
                ),
            ),
        ]
    )

    # Обучение пайплайна
    pipe.fit(tfidf, df["label"].values)

    # Оценка точности
    accuracy = pipe.score(tfidf, df["label"].values)

    # Логгирование параметров, метрик и модели с использованием MLflow
    with mlflow.start_run():
        # Логгирование параметров
        mlflow.log_param("max_depth", pipe.named_steps["classifier"].max_depth)

        # Логгирование метрик
        mlflow.log_metric("accuracy", accuracy)

        # Логгирование модели
        mlflow.sklearn.log_model(pipe, "decision_tree_model")

        # Преобразование модели scikit-learn в формат ONNX
        onnx_model = convert_sklearn(
            pipe, initial_types=[("input", ftp([None, tfidf.shape[1]]))]
        )

        # Сохранение ONNX модели
        save_model(onnx_model, config.path.model)


if __name__ == "__main__":
    main()
