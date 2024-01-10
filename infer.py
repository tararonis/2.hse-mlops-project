import gdown
import hydra
import numpy as np
import onnxruntime
import pandas as pd
from omegaconf import OmegaConf

from utils import convert_to_num_data, create_folder


@hydra.main(config_path="config", config_name="config", version_base=None)
def main(config: OmegaConf):
    # Скачивание датасета с gdrive
    gdown.download(config.links.test_data, config.path.test_data, quiet=False)
    df = pd.read_csv(config.path.test_data)

    #  Преобразование текстовых данных в числовые
    tfidf = convert_to_num_data(df).toarray()
    tfidf = tfidf[:, :75541]

    session = onnxruntime.InferenceSession(config.path.model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Выполняем предсказание
    pred = session.run([output_name], {input_name: tfidf.astype(np.float32)})

    # Преобразуем предсказания в DataFrame
    predictions_df = pd.DataFrame(data=pred[0], columns=["Prediction"])

    # Сохраняем DataFrame в CSV файл
    create_folder("data/tmp")
    predictions_df.to_csv("data/tmp/predictions.csv", index=False)


if __name__ == "__main__":
    main()
