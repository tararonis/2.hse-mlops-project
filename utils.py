from pathlib import Path

import pandas as pd
import scipy
from dvc.api import open
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def convert_to_num_data(df) -> scipy.sparse._csr.csr_matrix:
    """Преобразование текстовых данных в числовые

    Args: df (Data Frame): Pandas Dataframe с
    тестовыви или тренировочными данными
    """
    df.fillna(" ", inplace=True)
    transformer = TfidfTransformer(smooth_idf=False)
    count_vectorizer = CountVectorizer(ngram_range=(1, 2))
    counts = count_vectorizer.fit_transform(df["content"].values)
    tfidf = transformer.fit_transform(counts)
    return tfidf


def load_data(path, remote) -> pd.DataFrame:
    """Скачивание данных с удаленного gdrive

    Args:
        - path (str): путь куда будет скачан файл
        - remote (str): ссылка до удаленного хранилища файла
    """
    with open(path, remote=remote) as f:
        df = pd.read_csv(f)
    return df


def create_folder(dir_name) -> None:
    """Создание папки

    Args: dir_name (str): название директории
    """
    directory_path = Path("dir_name")
    directory_path.mkdir(parents=True, exist_ok=True)
