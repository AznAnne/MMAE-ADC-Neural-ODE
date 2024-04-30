import pandas as pd
from sklearn.model_selection import train_test_split


def data_split(df, on_col, seed, test_size=0.2):
    """Simple data split along a specified column."""
    unique_ids = df[on_col].unique()
    train, test = train_test_split(unique_ids, random_state=seed, test_size=test_size, shuffle=True)
    train_df = df[df[on_col].isin(train)]
    test_df = df[df[on_col].isin(test)]
    return train_df, test_df


def data_split_cross(df):
    """
    Splits for cross-schedule prediction as follows:
     Train on 3-week dosing
     Test on 1-week dosing
    """
    train_df = df[(df['STUD'] > 19) | (df['STUD'] < 19)]
    test_df = df[df['STUD'] == 19]
    return train_df, test_df


def augment_data(train, num_augmentations=10):
    augment_data = pd.DataFrame(columns=train.columns)
    for ptnm in train.PTNM.unique():
        for i in range(1, num_augmentations + 1):
            time_upper_bound = (i + 1) * 21 * 24
            df = train[(train.PTNM == ptnm) & (train.TIME <= time_upper_bound) & (train.TIME >= 0)]
            df.loc[:, "PTNM"] = df["PTNM"] + f".{i}"
            augment_data = pd.concat([augment_data, df], ignore_index=True)

    train = pd.concat([train, augment_data], ignore_index=True).reset_index(drop=True)
    return train