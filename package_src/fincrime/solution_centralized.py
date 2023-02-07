
import os
import torch
import pickle
from pathlib import Path
from loguru import logger
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def one_hot_encoder(s: pd.Series) -> torch.Tensor:
    """
    Apply the same one-hot-encoder method to train data and test data
    Only used for 'Flags' feature in bank_data
    :param s: The 'Flags' column in bank_data
    :return: A tensor of shape n*11
    """
    df1 = pd.get_dummies(s)
    df1.columns = df1.columns.astype(int)
    df1.fillna(0, inplace=True)
    return torch.Tensor(df1[sorted(df1.columns)].values)


def one_hot_encoder_currency(s: pd.Series) -> pd.DataFrame:
    """
    One-hot encode the most frequent 10 kinds of currency as new features
    :param s: The 'currency' column in swift data
    :return: An n*10 dataframe
    """
    currency_list = ['EUR', 'USD', 'GBP', 'CHF', 'CAD', 'SEK', 'PLN', 'AUD', 'ZAR', 'JPY']
    tmp_df = pd.get_dummies(s)
    diff = set(currency_list).difference(set(tmp_df.columns))
    if len(diff) > 0:
        for f in list(diff):
            tmp_df[f] = 0
    return tmp_df[currency_list]


def prepare_XY(swift_data: pd.DataFrame, bank_data: pd.DataFrame, model_dir):
    """
    1.Merge swift_data and bank_data together.
    2.Then perform some feature engineering.
    3.Apply normalization to the final X.
    """
    logger.info("-" * 50 + "prepare_XY start...")
    # merge bank data into swift data by 'OrderingAccount' and 'BeneficiaryAccount'
    swift_ordering_account_set = set(swift_data['OrderingAccount'])
    swift_beneficiary_account_set = set(swift_data['BeneficiaryAccount'])

    bank_data_ordering = bank_data[[x in swift_ordering_account_set for x in bank_data['Account']]][
        ['Account', 'Flags']]
    bank_data_beneficiary = bank_data[[x in swift_beneficiary_account_set for x in bank_data['Account']]][
        ['Account', 'Flags']]

    bank_data_ordering_df = pd.DataFrame(one_hot_encoder(bank_data_ordering['Flags']).detach().numpy())
    bank_data_beneficiary_df = pd.DataFrame(one_hot_encoder(bank_data_beneficiary['Flags']).detach().numpy())
    bank_data_ordering_df.columns = [f"ordering_{x}" for x in bank_data_ordering_df.columns]
    bank_data_beneficiary_df.columns = [f"beneficiary_{x}" for x in bank_data_beneficiary_df.columns]
    f_list3 = list(bank_data_ordering_df.columns) + list(bank_data_beneficiary_df.columns )

    bank_data_ordering_df['OrderingAccount'] = bank_data_ordering['Account'].values
    bank_data_beneficiary_df['BeneficiaryAccount'] = bank_data_beneficiary['Account'].values

    bank_data_ordering_df.drop_duplicates(subset=['OrderingAccount'], inplace=True)
    bank_data_beneficiary_df.drop_duplicates(subset=['BeneficiaryAccount'], inplace=True)

    swift_data = pd.merge(swift_data, bank_data_ordering_df, on='OrderingAccount', how='left')
    swift_data = pd.merge(swift_data, bank_data_beneficiary_df, on='BeneficiaryAccount', how='left')

    # feature engineering
    swift_data['currency_equality'] = np.int_(swift_data['SettlementCurrency'] == swift_data['InstructedCurrency'])

    # one-hot encoder factor feature in swift data
    swift_num_cols = ['SettlementAmount', 'InstructedAmount']
    settlement = one_hot_encoder_currency(swift_data['SettlementCurrency'])
    settlement.columns = ["settlement_%s" % x for x in settlement.columns]
    instructed = one_hot_encoder_currency(swift_data['InstructedCurrency'])
    instructed.columns = ["instructed_%s" % x for x in instructed.columns]

    swift_data['hour'] = swift_data['Timestamp'].astype("datetime64[ns]").dt.hour

    swift_data["sender_hour"] = swift_data["Sender"] + swift_data["hour"].astype(str)
    vc = swift_data["sender_hour"].value_counts().reset_index().rename(columns={'index': 'sender_hour', 'sender_hour': 'sender_hour_freq'})
    swift_data = pd.merge(swift_data, vc, on='sender_hour', how='left').fillna(0)

    # Sender-Currency Frequency and Average Amount per Sender-Currency
    swift_data["sender_currency"] = swift_data["Sender"] + swift_data["InstructedCurrency"]
    vc = swift_data["sender_currency"].value_counts().reset_index().rename(columns={'index': 'sender_currency', 'sender_currency': 'sender_currency_freq'})
    swift_data = pd.merge(swift_data, vc, on='sender_currency', how='left').fillna(0)

    vm = swift_data.groupby(by='sender_currency')[['InstructedAmount']].agg(np.nanmean).reset_index().rename(columns={'InstructedAmount': 'sender_currency_amount_average'})
    swift_data = pd.merge(swift_data, vm, on='sender_currency', how='left').fillna(0)

    # Sender-Receiver Frequency
    swift_data["sender_receiver"] = swift_data["Sender"] + swift_data["Receiver"]
    vc = swift_data["sender_receiver"].value_counts().reset_index().rename(columns={'index': 'sender_receiver', 'sender_receiver': 'sender_receiver_freq'})
    swift_data = pd.merge(swift_data, vc, on='sender_receiver', how='left').fillna(0)

    hb_columns = ['sender_hour_freq', 'sender_currency_freq', 'sender_currency_amount_average', 'sender_receiver_freq', 'hour']

    if 'Label' in swift_data.columns:
        df = pd.concat([swift_data[['MessageId', 'currency_equality', 'Label'] + swift_num_cols + hb_columns + f_list3],
                        settlement, instructed], axis=1)
        X = df.drop(columns=['MessageId', 'Label']).values
        Y = df['Label'].values
    else:
        df = pd.concat([swift_data[['MessageId', 'currency_equality'] + swift_num_cols + hb_columns + f_list3],
                        settlement, instructed], axis=1)
        X = df.drop(columns=['MessageId']).values
        Y = None

    if 'Label' in swift_data.columns:
        scaler = StandardScaler()
        scaler.fit(X)
        with open(os.path.join(model_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(os.path.join(model_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

    X = scaler.transform(X)
    logger.info("-" * 50 + "prepare_XY Finished!")

    return df['MessageId'].values, X, Y


def fit(swift_data_path: Path, bank_data_path: Path, model_dir: Path):
    """
    The main function for training.
    1.Will load data and call the helper functions to preprocess.
    2.Use XGBClassifier to train a privacy-protected model.
    3.Save model into disk for evaluation.
    """
    logger.info("-" * 50 + "fit")
    swift_data = pd.read_csv(swift_data_path)
    bank_data = pd.read_csv(bank_data_path, dtype=pd.StringDtype())

    _, X, Y = prepare_XY(swift_data, bank_data, model_dir)
    X += np.random.normal(loc=0., scale=1e-6, size=X.shape)

    dp_ebm = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.3)
    logger.info("-" * 50 + "Start Training...")
    dp_ebm.fit(X, Y)
    logger.info("-" * 50 + "Training Finished!")
    # save model to disk
    with open(os.path.join(model_dir, "dp_ebm.model"), 'wb') as model_file:
        pickle.dump(dp_ebm, model_file)
    logger.info("-" * 50 + "Model saved to disk!")


def predict(swift_data_path: Path, bank_data_path: Path, model_dir: Path, preds_format_path: Path, preds_dest_path: Path):
    """
    The main function for testing.
    1.Will load data and call the helper functions to preprocess.
    2.Use saved model to predict the label the test data.
    3.Save prediction with a given format to a given path.
    """
    logger.info("-" * 50 + "predict")
    swift_data = pd.read_csv(swift_data_path)
    bank_data = pd.read_csv(bank_data_path, dtype=pd.StringDtype())
    logger.info(f"swift data columns: {swift_data.columns}")
    logger.info(f"bank data columns: {bank_data.columns}")
    logger.info(f"Flags: {bank_data['Flags'].unique()}")
    logger.info(f"swift_data shape: {swift_data.shape}")
    logger.info(f"bank_data shape: {bank_data.shape}")

    MessageId, X, Y = prepare_XY(swift_data, bank_data, model_dir)
    with open(os.path.join(model_dir, "dp_ebm.model"), 'rb') as model_file:
        dp_ebm = pickle.load(model_file)

    y_pred = dp_ebm.predict_proba(X)
    result = pd.DataFrame()
    result['MessageId'] = MessageId
    result['Score'] = y_pred[:, 1]

    preds_format_df = pd.read_csv(preds_format_path)
    formatted_result = pd.merge(preds_format_df[['MessageId']], result, on='MessageId', how='left')
    formatted_result.to_csv(preds_dest_path, index=False)
    logger.info('-' * 70 + "Centralized FL prediction done!")
