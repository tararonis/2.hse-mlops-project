import gdown
import nltk
import pandas as pd

nltk.download("stopwords")
import hydra
import mlflow
from nltk.corpus import stopwords
from omegaconf import OmegaConf
from onnxmltools.convert import convert_sklearn
from onnxmltools.utils import save_model
from skl2onnx.common.data_types import FloatTensorType
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: OmegaConf):

    # Скачивание датасета с gdrive
    file_content = gdown.download(
        config.links.train_data, "hse-mlops-project/data2/train_data.csv", quiet=False
    )
    df = pd.read_csv("data2/train_data.csv")

    df.fillna(" ", inplace=True)

    #  Преобразование текстовых данных в числовые
    transformer = TfidfTransformer(smooth_idf=False)
    count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    counts = count_vectorizer.fit_transform(df["content"].values)
    tfidf = transformer.fit_transform(counts)

    # mlflow.set_tracking_uri(uri=config.mlflow.uri)

    # Создание пайплайна
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler(with_mean=False)),
            ("classifier", DecisionTreeClassifier(max_depth=58, random_state=42)),
        ]
    )

    # Обучение пайплайна
    pipeline.fit(tfidf, df["label"].values)

    # Оценка точности
    accuracy = pipeline.score(tfidf, df["label"].values)

    # Логгирование параметров, метрик и модели с использованием MLflow
    with mlflow.start_run():
        # Логгирование параметров
        mlflow.log_param("max_depth", pipeline.named_steps["classifier"].max_depth)

        # Логгирование метрик
        mlflow.log_metric("accuracy", accuracy)

        # Логгирование модели
        mlflow.sklearn.log_model(pipeline, "decision_tree_model")

        # Сохранение модели в формате ONNX
        onnx_path = "models/decision_tree_model.onnx"
        # Преобразование модели scikit-learn в формат ONNX
        onnx_model = convert_sklearn(
            pipeline, initial_types=[("input", FloatTensorType([None, tfidf.shape[1]]))]
        )

        # Сохранение ONNX модели
        save_model(onnx_model, onnx_path)


if __name__ == "__main__":
    main()
