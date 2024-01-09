from pathlib import Path

import hydra
import numpy as np
import onnxruntime
import pandas as pd
from omegaconf import OmegaConf
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: OmegaConf):
    # Скачивание датасета с gdrive
    # file_content = gdown.download(cfg.links.test_data, "hse-mlops-project/data2/test_data.csv", quiet=False)
    df = pd.read_csv("data2/test_data.csv")

    df.fillna(" ", inplace=True)

    #  Преобразование текстовых данных в числовые
    transformer = TfidfTransformer(smooth_idf=False)
    count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    counts = count_vectorizer.fit_transform(df["content"].values)
    tfidf = transformer.fit_transform(counts)

    sess = onnxruntime.InferenceSession("models/decision_tree_model.onnx")

    input_data = {"input": tfidf.toarray().astype(np.float32).reshape(-1, 35786)}
    # Выполняем предсказание
    output = sess.run(None, input_data)

    # Преобразуем предсказания в DataFrame
    predictions_df = pd.DataFrame(data=output[0], columns=["Prediction"])

    # Сохраняем DataFrame в CSV файл
    directory_path = Path("data/tmp")
    directory_path.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv("data/tmp/predictions.csv", index=False)


if __name__ == "__main__":
    main()
