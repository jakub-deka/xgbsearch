import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split


def get_titanic() -> pd.DataFrame:
    df = sns.load_dataset("titanic")
    df = df.assign(deck=lambda x: x.deck.astype(str).replace({"nan": "missing"}))
    df = df.fillna({"age": -1, "embarked": "missing", "deck": "missing"})
    df = pd.get_dummies(df, columns=["sex", "embarked", "class", "who", "deck"])

    df = df.drop(columns=["alive", "embark_town"])

    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)

    train_indices, test_indices = train_test_split(df.index, test_size=0.3)
    df = df.assign(population="train")
    df.loc[test_indices, "population"] = "test"

    x_train = df.query("population == 'train'").drop(columns=["survived", "population"])
    y_train = df.query("population == 'train'").survived
    x_test = df.query("population == 'test'").drop(columns=["survived", "population"])
    y_test = df.query("population == 'test'").survived

    return x_train, x_test, y_train, y_test
