

import os
import sys

import torch
import pickle
import flwr as fl
import pandas as pd
import numpy as np
import copy

from loguru import logger
from torch.utils.data import DataLoader
from pathlib import Path
from collections import OrderedDict
from functools import reduce
from typing import Dict, Optional, List, Tuple
from interpret.privacy import DPExplainableBoostingClassifier as DPEBC

from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet

from flwr.server import ClientManager
from flwr.common import FitIns, FitRes, parameters_to_ndarrays, GetParametersIns, GetParametersRes, Status, Code, EvaluateIns, EvaluateRes, Scalar, Parameters, ndarrays_to_parameters, NDArrays
from flwr.server.client_proxy import ClientProxy
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler

GLOBAL_EPOCH = 5  # Global epoch number of auto-encoder training
ENCODER_DIM = 7  # The encoder dimension of 'Flags' feature in band_data


def ndarray_to_bytes(ndarray: np.ndarray) -> bytes:
    """
    Used for serialization of model parameters.
    """
    return ndarray.tobytes()


def bytes_to_ndarray(b: bytes, dtype) -> np.ndarray:
    """
    Used for deserialization of model parameters.
    """
    return np.frombuffer(b, dtype=dtype)


class AutoEncoderHalf(torch.nn.Module):
    """First half model of autoencoder"""
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(11, 9),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(9, ENCODER_DIM))

        self.decoder = torch.nn.Sequential(torch.nn.Linear(ENCODER_DIM, 9),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(9, 11),
                                           torch.nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        # decoded = self.decoder(encoded)
        return encoded


def model_encoder(model):
    """This is to generate the first half model of trained autoencoder to encode the filtered Flags from bank client"""
    new_model = AutoEncoderHalf()

    parameters = [val.cpu().numpy() for _, val in model.autoencoder.state_dict().items()]
    params_dict = zip(model.autoencoder.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})

    new_model.load_state_dict(state_dict, strict=True)

    return new_model


class AutoEncoder(torch.nn.Module):
    """
    The auto-encoder model for encoding the 'Flags' feature in bank_data
    """
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(torch.nn.Linear(11, 9),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(9, ENCODER_DIM))

        self.decoder = torch.nn.Sequential(torch.nn.Linear(ENCODER_DIM, 9),
                                           torch.nn.ReLU(),
                                           torch.nn.Linear(9, 11),
                                           torch.nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class BankModel:
    """
    Wrap auto-encoder model into this class to provide an aligned API (with SwiftModel).
    """
    def __init__(self):
        self.autoencoder = AutoEncoder()

    def fit(self, X):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=0.01, weight_decay=1e-8)
        for x in DataLoader(X, batch_size=128, shuffle=None):
            y_hat = self.autoencoder(x)
            loss = loss_fn(y_hat, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        return self


class SwiftModel:
    """
    Wrap XGBClassifier model into this class to provide an aligned API (with BankModel).
    """
    def __init__(self):
        self.dp_ebm = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.3)

    def fit(self, X, Y):
        self.dp_ebm.fit(X, Y)
        return self


def generate_asymmetric_keys(client_dir: Path):
    """
    Generate RSA private and public keys and save them to disk.
    """
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048, backend=default_backend())
    public_key = private_key.public_key()

    serial_private = private_key.private_bytes(encoding=serialization.Encoding.PEM,
                                               format=serialization.PrivateFormat.PKCS8,
                                               encryption_algorithm=serialization.NoEncryption())
    with open(os.path.join(client_dir, "private_key.rsa"), 'wb') as f:
        f.write(serial_private)

    serial_pub = public_key.public_bytes(encoding=serialization.Encoding.PEM,
                                         format=serialization.PublicFormat.SubjectPublicKeyInfo)
    with open(os.path.join(client_dir, "public_key.rsa"), 'wb') as f:
        f.write(serial_pub)


def read_private(client_dir: Path):
    """
    Load RSA private key from disk.
    """
    filename = os.path.join(client_dir, "private_key.rsa")
    with open(filename, "rb") as key_file:
        private_key = serialization.load_pem_private_key(key_file.read(), password=None, backend=default_backend())
    return private_key


def read_public(client_dir: Path):
    """
    Load RSA public key from disk.
    """
    filename = os.path.join(client_dir, "public_key.rsa")
    with open(filename, "rb") as key_file:
        public_key = serialization.load_pem_public_key(key_file.read(), backend=default_backend())
    return public_key


def asym_encrypt(public_key, message):
    """
    Use public_key to encrypt message.
    """
    encrypted = public_key.encrypt(message,
                                   padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(),
                                                label=None))
    return encrypted


def asym_decrypt(private_key, encrypted):
    """
    Use private_key to decrypt encrypted message.
    """
    decrypted = private_key.decrypt(encrypted,
                                    padding.OAEP(mgf=padding.MGF1(algorithm=hashes.SHA256()), algorithm=hashes.SHA256(),
                                                 label=None))
    return decrypted


def generate_server_symmetric_key(server_dir: Path):
    """
    Generate symmetric key on server and save it to disk.
    """
    sym_key = Fernet.generate_key()
    with open(os.path.join(server_dir, "server_symmetric.key"), "wb") as f:
        f.write(sym_key)


def generate_swift_symmetric_key(client_dir: Path):
    """
    Generate symmetric key on swift client and save it to disk.
    """
    sym_key = Fernet.generate_key()
    with open(os.path.join(client_dir, "swift_symmetric.key"), "wb") as f:
        f.write(sym_key)


def load_server_symmetric_key(client_dir: Path):
    """
    Load symmetric key generated on server.
    """
    with open(os.path.join(client_dir, "server_symmetric.key"), "rb") as f:
        return f.read()


def load_swift_symmetric_key(client_dir: Path):
    """
    Load symmetric key generated on swift client.
    """
    with open(os.path.join(client_dir, "swift_symmetric.key"), "rb") as f:
        return f.read()


def sym_encrypt(sym_key, message):
    """
    Use symmetric key to encrypt message.
    """
    fernet = Fernet(sym_key)
    return fernet.encrypt(message)


def sym_decrypt(sym_key, message):
    """
    Use symmetric key to decrypt encrypted message.
    """
    fernet = Fernet(sym_key)
    return fernet.decrypt(message)


def set_parameters(model, parameters: List[np.ndarray]):
    """
    Update model with new parameters.
    """
    params_dict = zip(model.autoencoder.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.autoencoder.load_state_dict(state_dict, strict=True)


def get_parameters(model):
    """
    Get the parameters from model.
    """
    return [val.cpu().numpy() for _, val in model.autoencoder.state_dict().items()]


def empty_parameters() -> Parameters:
    """
    Generates empty Flower Parameters dataclass instance.
    """
    return fl.common.ndarrays_to_parameters([])


def initialize_autoencoder_parameters() -> Parameters:
    """
    Initialize the auto-encoder parameters.
    """
    model = BankModel()
    return ndarrays_to_parameters(get_parameters(model))


def aggregate(results: List[Tuple[NDArrays, int]]) -> NDArrays:
    """
    Compute weighted average of model parameters from bank clients.
    """
    # Calculate the total number of examples used during training
    num_examples_total = sum([num_examples for _, num_examples in results])

    # Create a list of weights, each multiplied by the related number of examples
    weighted_weights = [[layer * num_examples for layer in weights] for weights, num_examples in results]

    # Compute average weights of each layer
    weights_prime = [reduce(np.add, layer_updates) / num_examples_total for layer_updates in zip(*weighted_weights)]
    return weights_prime


def merged_swift_df_to_XY(merged_swift_df, client_dir):
    """
    1.Perform some feature engineering.
    2.Apply normalization to the final X.
    """
    merged_swift_df = merged_swift_df.copy()
    merged_swift_df['currency_equality'] = np.int_(merged_swift_df['SettlementCurrency'] == merged_swift_df['InstructedCurrency'])

    merged_swift_df_num_cols = ['SettlementAmount', 'InstructedAmount'] + [x for x in merged_swift_df.columns if 'Encoder' in str(x)]

    settlement = one_hot_encoder_currency(merged_swift_df['SettlementCurrency'])
    settlement.columns = ["settlement_%s" % x for x in settlement.columns]
    instructed = one_hot_encoder_currency(merged_swift_df['InstructedCurrency'])
    instructed.columns = ["instructed_%s" % x for x in instructed.columns]

    # new feature engineering start
    merged_swift_df['hour'] = merged_swift_df['Timestamp'].astype("datetime64[ns]").dt.hour
    merged_swift_df["sender_hour"] = merged_swift_df["Sender"] + merged_swift_df["hour"].astype(str)
    vc = merged_swift_df["sender_hour"].value_counts().reset_index().rename(columns={'index': 'sender_hour', 'sender_hour': 'sender_hour_freq'})
    merged_swift_df = pd.merge(merged_swift_df, vc, on='sender_hour', how='left').fillna(0)

    merged_swift_df["sender_currency"] = merged_swift_df["Sender"] + merged_swift_df["InstructedCurrency"]
    vc = merged_swift_df["sender_currency"].value_counts().reset_index().rename(columns={'index': 'sender_currency', 'sender_currency': 'sender_currency_freq'})
    merged_swift_df = pd.merge(merged_swift_df, vc, on='sender_currency', how='left').fillna(0)

    vm = merged_swift_df.groupby(by='sender_currency')[['InstructedAmount']].agg(np.nanmean).reset_index().rename(columns={'InstructedAmount': 'sender_currency_amount_average'})
    merged_swift_df = pd.merge(merged_swift_df, vm, on='sender_currency', how='left').fillna(0)

    merged_swift_df["sender_receiver"] = merged_swift_df["Sender"] + merged_swift_df["Receiver"]
    vc = merged_swift_df["sender_receiver"].value_counts().reset_index().rename(columns={'index': 'sender_receiver', 'sender_receiver': 'sender_receiver_freq'})
    merged_swift_df = pd.merge(merged_swift_df, vc, on='sender_receiver', how='left').fillna(0)

    hb_columns = ['sender_hour_freq', 'sender_currency_freq', 'sender_currency_amount_average', 'sender_receiver_freq', 'hour']

    # new feature engineering end

    if 'Label' in merged_swift_df.columns:
        df = pd.concat([merged_swift_df[['MessageId', 'currency_equality', 'Label'] + merged_swift_df_num_cols + hb_columns],
                        settlement, instructed], axis=1)
        df.fillna(0, inplace=True)
        X = df.drop(columns=['MessageId', 'Label']).values
        Y = df['Label'].values
    else:
        df = pd.concat([merged_swift_df[['MessageId', 'currency_equality'] + merged_swift_df_num_cols + hb_columns],
                        settlement, instructed], axis=1)
        df.fillna(0, inplace=True)
        X = df.drop(columns=['MessageId']).values
        Y = None

    if 'Label' in merged_swift_df.columns:
        scaler = StandardScaler()
        scaler.fit(X)
        with open(os.path.join(client_dir, 'scaler.pkl'), 'wb') as f:
            pickle.dump(scaler, f)
    else:
        with open(os.path.join(client_dir, 'scaler.pkl'), 'rb') as f:
            scaler = pickle.load(f)

    X = scaler.transform(X)

    return df['MessageId'].values, X, Y


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


class TrainingSwiftClient(fl.client.Client):
    def __init__(self, cid, swift_df, model, client_dir):
        super(TrainingSwiftClient, self).__init__()
        self.cid = cid
        self.swift_df = swift_df
        self.model = model
        self.client_dir = client_dir

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        return GetParametersRes(status=Status(code=Code.OK, message="Success"), parameters=empty_parameters())

    def fit(self, ins: FitIns) -> FitRes:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"Swift client fit round {ins.config['round']} start...")
        if ins.config['round'] == 1:
            raise ValueError("Swift clients should not be selected in 0th round.")
        elif ins.config['round'] == 2:
            generate_swift_symmetric_key(self.client_dir)
            swift_symmetric_key = load_swift_symmetric_key(self.client_dir)

            # encrypt symmetric key
            bank_clients_public_key_dict = ins.config.copy()
            bank_clients_public_key_dict.pop('round')
            encrypted_swift_sym_key = {cid: asym_encrypt(serialization.load_pem_public_key(public_key, backend=default_backend()), swift_symmetric_key) for cid, public_key in bank_clients_public_key_dict.items()}

            # encrypt OrderingAccount and BeneficiaryAccount
            encrypted_ordering = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(self.swift_df['OrderingAccount'].values.astype('<U32')))
            encrypted_beneficiary = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(self.swift_df['BeneficiaryAccount'].values.astype('<U32')))

            encrypted_swift_sym_key['encrypted_ordering'] = encrypted_ordering
            encrypted_swift_sym_key['encrypted_beneficiary'] = encrypted_beneficiary

            fit_res = FitRes(status=Status(code=Code.OK, message="Success"),
                             parameters=empty_parameters(),
                             num_examples=0,
                             metrics=encrypted_swift_sym_key)
            logger.info("*" * 50 + f"fit_res size of round {ins.config['round']} from {self.cid}: {sys.getsizeof(fit_res) / 1024 ** 3}")
            return fit_res
        elif ins.config['round'] <= GLOBAL_EPOCH:
            raise ValueError(f"Swift clients should not be selected in {ins.config['round']} round.")
        elif ins.config['round'] == GLOBAL_EPOCH + 1:
            swift_symmetric_key = load_swift_symmetric_key(self.client_dir)

            ordering_account = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['ordering_account_bytes']), dtype='<U32')
            beneficiary_account = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['beneficiary_account_bytes']), dtype='<U32')
            ordering_encoder = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['ordering_encoder_bytes']), dtype='float16')
            beneficiary_encoder = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['beneficiary_encoder_bytes']), dtype='float16')

            ordering_ndarray = np.concatenate([ordering_account.reshape(-1, 1), ordering_encoder.reshape(-1, ENCODER_DIM)], axis=1)
            beneficiary_ndarray = np.concatenate([beneficiary_account.reshape(-1, 1), beneficiary_encoder.reshape(-1, ENCODER_DIM)], axis=1)

            logger.info(f"ordering_ndarray.shape: [ordering_ndarray.shape]")
            logger.info(f"ordering_ndarray[:5, :5]: {ordering_ndarray[:5, :5]}")

            ordering_df = pd.DataFrame(ordering_ndarray, columns=['OrderingAccount'] + ['OEncoder' + str(x) for x in range(ordering_ndarray.shape[1] - 1)])
            beneficiary_df = pd.DataFrame(beneficiary_ndarray, columns=['BeneficiaryAccount'] + ['BEncoder' + str(x) for x in range(beneficiary_ndarray.shape[1] - 1)])

            for column in ordering_df.columns:
                if column != 'OrderingAccount':
                    ordering_df[column] = ordering_df[column].astype('float16')

            for column in beneficiary_df.columns:
                if column != 'BeneficiaryAccount':
                    beneficiary_df[column] = beneficiary_df[column].astype('float16')

            swift_df = pd.merge(self.swift_df, ordering_df, on='OrderingAccount', how='left')
            swift_df = pd.merge(swift_df, beneficiary_df, on='BeneficiaryAccount', how='left')

            _, X, Y = merged_swift_df_to_XY(swift_df, self.client_dir)
            X += np.random.normal(loc=0., scale=1e-6, size=X.shape)
            self.model.fit(X, Y)

            pickle.dump(self.model, open(os.path.join(self.client_dir, 'SwiftModel.pkl'), 'wb'))
            logger.info('-' * 60 + "SwiftModel.pkl save to disk!")

            return FitRes(status=Status(code=Code.OK, message="Success"),
                          parameters=empty_parameters(),
                          num_examples=0,
                          metrics={})
        else:
            raise ValueError(f"This {ins.config['round']} round should not exist.")


class TrainingBankClient(fl.client.Client):
    def __init__(self, cid, bank_df, model, client_dir):
        super(TrainingBankClient, self).__init__()
        self.cid = cid
        self.bank_df = bank_df
        self.model = model
        self.client_dir = client_dir
        self.X = one_hot_encoder(bank_df['Flags'])

    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:
        """
        This func would only be called in the very first server_round if global model parameters are not initialized.
        """
        print(f"[Client {self.cid}] get_parameters")
        ndarrays: List[np.ndarray] = get_parameters(self.model)
        parameters = ndarrays_to_parameters(ndarrays)
        return GetParametersRes(status=Status(code=Code.OK, message="Success"), parameters=parameters)

    def fit(self, ins: FitIns) -> FitRes:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"Bank client (cid: {self.cid}) fit round {ins.config['round']} start...")
        if ins.config['round'] == 1:
            generate_asymmetric_keys(self.client_dir)
            with open(os.path.join(self.client_dir, "public_key.rsa"), "rb") as f:
                public_key_bytes = f.read()
            return FitRes(status=Status(code=Code.OK, message="Success"),
                          parameters=Parameters([], ''),
                          num_examples=0,
                          metrics={self.cid: public_key_bytes})
        elif ins.config['round'] < GLOBAL_EPOCH:
            if ins.config['round'] == 2:  # decrypt server's symmetric key
                encrypted_sym_key_server = ins.config['encrypted_sym_key_server']
                private_key = read_private(self.client_dir)
                sym_key_server = asym_decrypt(private_key, encrypted_sym_key_server)
                with open(os.path.join(self.client_dir, "server_symmetric.key"), "wb") as f:
                    f.write(sym_key_server)
            elif ins.config['round'] == 3:  # decrypt swift's symmetric key
                encrypted_sym_key_swift = ins.config['encrypted_sym_key_swift']
                private_key = read_private(self.client_dir)
                sym_key_swift = asym_decrypt(private_key, encrypted_sym_key_swift)
                with open(os.path.join(self.client_dir, "swift_symmetric.key"), "wb") as f:
                    f.write(sym_key_swift)
            else:
                pass
            # decrypt and deserialize initial autoencoder parameters
            sym_key_server = load_server_symmetric_key(self.client_dir)
            parameters = ins.parameters
            parameters.tensors = [sym_decrypt(sym_key_server, x) for x in parameters.tensors]  # decrypt
            ndarrays_original = parameters_to_ndarrays(parameters)  # deserialize
            set_parameters(self.model, ndarrays_original)

            for local_epoch in range(20):  # LocalEpoch hyperparameter will be fixed here
                self.model.fit(self.X)  # local train

            ndarrays_updated = get_parameters(self.model)
            parameters_updated = ndarrays_to_parameters(ndarrays_updated)  # serialize
            parameters_updated.tensors = [sym_encrypt(sym_key_server, x) for x in parameters_updated.tensors]  # encrypt
            return FitRes(status=Status(code=Code.OK, message="Success"),
                          parameters=parameters_updated,
                          num_examples=len(self.X),
                          metrics={})
        elif ins.config['round'] == GLOBAL_EPOCH:
            # save global model to disk
            sym_key_server = load_server_symmetric_key(self.client_dir)
            parameters = ins.parameters
            parameters.tensors = [sym_decrypt(sym_key_server, x) for x in parameters.tensors]  # decrypt
            ndarrays_original = parameters_to_ndarrays(parameters)  # deserialize
            set_parameters(self.model, ndarrays_original)
            pickle.dump(self.model, open(os.path.join(self.client_dir, 'BankModel.pkl'), 'wb'))
            logger.info("*" * 50 + f"{self.cid} BankModel has been save to disk")

            # decrypt and deserialize OrderingAccount and BeneficiaryAccount
            sym_key_swift = load_swift_symmetric_key(self.client_dir)
            ordering_encrypted_bytes = ins.config['encrypted_ordering']
            beneficiary_encrypted_bytes = ins.config['encrypted_beneficiary']

            ordering = bytes_to_ndarray(sym_decrypt(sym_key_swift, ordering_encrypted_bytes), dtype='<U32')  # decrypt and deserialize
            beneficiary = bytes_to_ndarray(sym_decrypt(sym_key_swift, beneficiary_encrypted_bytes), dtype='<U32')  # decrypt and deserialize

            # filter bank_df based on OrderingAccount and BeneficiaryAccount
            ordering_df = pd.DataFrame(ordering, columns=['Account'])
            beneficiary_df = pd.DataFrame(beneficiary, columns=['Account'])

            columns = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            bank_flag = pd.DataFrame(self.X.numpy().astype(int), columns=columns)
            bank_flag['Account'] = self.bank_df['Account']

            ordering_df = pd.merge(ordering_df, bank_flag, on='Account', how='inner')
            beneficiary_df = pd.merge(beneficiary_df, bank_flag, on='Account', how='inner')

            # get encrypted (with swift's symmetric key) encoder
            new_model = model_encoder(self.model)
            ordering_encoder = new_model(torch.Tensor(ordering_df.drop(columns=['Account']).values))
            beneficiary_encoder = new_model(torch.Tensor(beneficiary_df.drop(columns=['Account']).values))

            ordering_encoder_array = ordering_encoder.detach().numpy()
            beneficiary_encoder_array = beneficiary_encoder.detach().numpy()

            ordering_encoder_array_account = ordering_df['Account'].values.astype('<U32')
            beneficiary_encoder_array_account = beneficiary_df['Account'].values.astype('<U32')
            ordering_encoder_array_encoder = ordering_encoder_array.astype('float16')
            beneficiary_encoder_array_encoder = beneficiary_encoder_array.astype('float16')

            logger.info('*' * 70 + 'bank_fit at round 5')
            logger.info(f"ordering_encoder_array_account.shape: {ordering_encoder_array_account.shape}")
            logger.info(f"beneficiary_encoder_array_account.shape: {beneficiary_encoder_array_account.shape}")
            logger.info(f"ordering_encoder_array_encoder.shape: {ordering_encoder_array_encoder.shape}")
            logger.info(f"beneficiary_encoder_array_encoder.shape: {beneficiary_encoder_array_encoder.shape}")

            ordering_account_bytes = sym_encrypt(sym_key_swift, ndarray_to_bytes(ordering_encoder_array_account))
            beneficiary_account_bytes = sym_encrypt(sym_key_swift, ndarray_to_bytes(beneficiary_encoder_array_account))
            ordering_encoder_bytes = sym_encrypt(sym_key_swift, ndarray_to_bytes(ordering_encoder_array_encoder))
            beneficiary_encoder_bytes = sym_encrypt(sym_key_swift, ndarray_to_bytes(beneficiary_encoder_array_encoder))

            fit_res = FitRes(status=Status(code=Code.OK, message="Success"),
                             parameters=Parameters([], ''),
                             num_examples=len(self.X),
                             metrics={'ordering_account_bytes': ordering_account_bytes,
                                      'beneficiary_account_bytes': beneficiary_account_bytes,
                                      'ordering_encoder_bytes': ordering_encoder_bytes,
                                      'beneficiary_encoder_bytes': beneficiary_encoder_bytes})
            logger.info("=" * 50 + f"fit_res size of round {ins.config['round']} from {self.cid}: {sys.getsizeof(fit_res) / 1024 ** 3}")
            return fit_res
        else:
            raise ValueError(f"Bank clients should not be selected in {ins.config['round']} round.")


def train_client_factory(cid, data_path: Path, client_dir: Path):
    if cid == "swift":
        logger.info("-" * 50 + f"Train: Initialing Swift Client for {cid}")
        swift_df = pd.read_csv(data_path)
        model = SwiftModel()
        return TrainingSwiftClient(cid, swift_df=swift_df, model=model, client_dir=client_dir)
    else:
        logger.info("-" * 50 + f"Train: Initialing Bank Client for {cid}")
        bank_df = pd.read_csv(data_path, dtype=pd.StringDtype())
        bank_df['Flags'] = bank_df['Flags'].astype(int)
        model = BankModel()
        return TrainingBankClient(cid, bank_df=bank_df, model=model, client_dir=client_dir)


class TrainStrategy(fl.server.strategy.Strategy):
    def __init__(self, server_dir: Path):
        super(TrainStrategy, self).__init__()
        self.server_dir = server_dir
        self.bank_public_key = dict()
        self.encrypted_swift_sym_key = dict()
        self.encrypted_ordering = None
        self.encrypted_beneficiary = None
        self.ordering_account_bytes = dict()
        self.beneficiary_account_bytes = dict()
        self.ordering_encoder_bytes = dict()
        self.beneficiary_encoder_bytes = dict()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return initialize_autoencoder_parameters()

    def configure_fit(self, server_round: int, parameters: Parameters,
                      client_manager: ClientManager) -> List[Tuple[ClientProxy, FitIns]]:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"config_fit in round {server_round} start...")
        if server_round == 1:
            parameters = empty_parameters()
            config_dict = {"round": server_round}
            bank_fit_ins = FitIns(parameters, config_dict)

            client_dict: Dict[str, ClientProxy] = client_manager.all()

            fit_config = [(v, bank_fit_ins) for k, v in client_dict.items() if k != "swift"]
            return fit_config
        elif server_round == 2:
            generate_asymmetric_keys(self.server_dir)
            with open(os.path.join(self.server_dir, "public_key.rsa"), "rb") as f:
                public_key_bytes = f.read()

            generate_server_symmetric_key(self.server_dir)
            server_symmetric_key = load_server_symmetric_key(self.server_dir)

            encrypted_server_symmetric_key_dict = {cid: asym_encrypt(serialization.load_pem_public_key(public_key, backend=default_backend()), server_symmetric_key) for cid, public_key in self.bank_public_key.items()}
            client_dict: Dict[str, ClientProxy] = client_manager.all()

            # Encrypt parameters
            parameters = self.initialize_parameters(client_manager)
            sym_key_server = load_server_symmetric_key(self.server_dir)
            parameters.tensors = [sym_encrypt(sym_key_server, x) for x in parameters.tensors]

            fit_config = []
            for k, v in client_dict.items():
                if k != "swift":
                    config_dict = {"round": server_round, "encrypted_sym_key_server": encrypted_server_symmetric_key_dict[k]}
                    bank_fit_ins = FitIns(parameters, config_dict)
                    fit_config.append((v, bank_fit_ins))
                else:
                    config_dict = self.bank_public_key.copy()
                    config_dict['server'] = public_key_bytes
                    config_dict["round"] = server_round
                    swift_fit_ins = FitIns(empty_parameters(), config_dict)
                    fit_config.append((v, swift_fit_ins))
            return fit_config
        elif server_round == 3:
            sym_key_server = load_server_symmetric_key(self.server_dir)
            parameters.tensors = [sym_encrypt(sym_key_server, x) for x in parameters.tensors]

            client_dict: Dict[str, ClientProxy] = client_manager.all()
            fit_config = [(v, FitIns(parameters, {"round": server_round, 'encrypted_sym_key_swift': self.encrypted_swift_sym_key[k]})) for k, v in client_dict.items() if k != "swift"]
            return fit_config
        elif server_round < GLOBAL_EPOCH:
            sym_key_server = load_server_symmetric_key(self.server_dir)
            parameters.tensors = [sym_encrypt(sym_key_server, x) for x in parameters.tensors]

            client_dict: Dict[str, ClientProxy] = client_manager.all()
            bank_fit_ins = FitIns(parameters, {"round": server_round})
            fit_config = [(v, bank_fit_ins) for k, v in client_dict.items() if k != "swift"]
            return fit_config
        elif server_round == GLOBAL_EPOCH:
            sym_key_server = load_server_symmetric_key(self.server_dir)
            parameters.tensors = [sym_encrypt(sym_key_server, x) for x in parameters.tensors]

            client_dict: Dict[str, ClientProxy] = client_manager.all()
            config_dict = {"round": server_round, "encrypted_ordering": self.encrypted_ordering, "encrypted_beneficiary": self.encrypted_beneficiary}
            bank_fit_ins = FitIns(parameters, config_dict)
            fit_config = [(v, bank_fit_ins) for k, v in client_dict.items() if k != "swift"]
            return fit_config
        elif server_round == GLOBAL_EPOCH + 1:
            client_dict: Dict[str, ClientProxy] = client_manager.all()

            swift_symmetric_key = load_swift_symmetric_key(self.server_dir)

            dict1 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='<U32') for k, v in self.ordering_account_bytes.items()}
            dict2 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='<U32') for k, v in self.beneficiary_account_bytes.items()}
            dict3 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='float16') for k, v in self.ordering_encoder_bytes.items()}
            dict4 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='float16') for k, v in self.beneficiary_encoder_bytes.items()}

            array1 = np.concatenate(list(dict1.values()), axis=0)
            array2 = np.concatenate(list(dict2.values()), axis=0)
            array3 = np.concatenate(list(dict3.values()), axis=0)
            array4 = np.concatenate(list(dict4.values()), axis=0)

            logger.info("*" * 50 + f"array1 shape: {array1.shape}")
            logger.info("*" * 50 + f"array2 shape: {array2.shape}")
            logger.info("*" * 50 + f"array3 shape: {array3.shape}")
            logger.info("*" * 50 + f"array4 shape: {array4.shape}")

            ordering_ndarray = np.concatenate([array1.reshape(-1, 1), array3.reshape(-1, ENCODER_DIM)], axis=1)
            beneficiary_ndarray = np.concatenate([array2.reshape(-1, 1), array4.reshape(-1, ENCODER_DIM)], axis=1)

            logger.info(f"ordering_ndarray shape: {ordering_ndarray.shape}")
            logger.info(f"ordering_ndarray[:4, :4]: {ordering_ndarray[:4, :4]}")

            ordering_df = pd.DataFrame(ordering_ndarray, columns=['OrderingAccount'] + ['OEncoder' + str(x) for x in range(ordering_ndarray.shape[1] - 1)])
            beneficiary_df = pd.DataFrame(beneficiary_ndarray, columns=['BeneficiaryAccount'] + ['BEncoder' + str(x) for x in range(beneficiary_ndarray.shape[1] - 1)])

            for column in ordering_df.columns:
                if column != 'OrderingAccount':
                    ordering_df[column] = ordering_df[column].astype('float16')

            for column in beneficiary_df.columns:
                if column != 'BeneficiaryAccount':
                    beneficiary_df[column] = beneficiary_df[column].astype('float16')

            ordering_df = ordering_df.groupby(by='OrderingAccount').agg(np.nanmean).reset_index()
            beneficiary_df = beneficiary_df.groupby(by='BeneficiaryAccount').agg(np.nanmean).reset_index()

            ordering_account_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(ordering_df['OrderingAccount'].values.astype('<U32')))
            beneficiary_account_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(beneficiary_df['BeneficiaryAccount'].values.astype('<U32')))
            ordering_encoder_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(ordering_df.drop(columns=['OrderingAccount']).values.astype('float16')))
            beneficiary_encoder_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(beneficiary_df.drop(columns=['BeneficiaryAccount']).values.astype('float16')))

            config_dict = {"round": server_round,
                           'ordering_account_bytes': ordering_account_bytes,
                           'beneficiary_account_bytes': beneficiary_account_bytes,
                           'ordering_encoder_bytes': ordering_encoder_bytes,
                           'beneficiary_encoder_bytes': beneficiary_encoder_bytes}

            swift_fit_ins = FitIns(empty_parameters(), config_dict)
            fit_config = [(v, swift_fit_ins) for k, v in client_dict.items() if k == "swift"]

            logger.info("*" * 50 + f"fit_config size of round {server_round} from server: {sys.getsizeof(config_dict) / 1024 ** 3}GB")
            return fit_config
        else:
            raise ValueError(f"Round {server_round} should not exist.")

    def aggregate_fit(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]],
                      failures) -> Tuple[Optional[Parameters], dict]:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"aggregate_fit in round {server_round} start...")
        if (n_failures := len(failures)) > 0:
            logger.info("!" * 50 + f"Had {n_failures} failures in round {server_round}")
        if server_round == 1:
            for _, fit_res in results:
                self.bank_public_key = self.bank_public_key | fit_res.metrics
            return empty_parameters(), {}
        elif server_round == 2:
            print("*" * 50, [client_proxy.cid for client_proxy, fit_res in results])
            weights_results: List[Tuple[List[np.ndarray], int]] = []
            for client_proxy, fit_res in results:
                cid = client_proxy.cid
                if cid == "swift":
                    self.encrypted_swift_sym_key = {k: v for k, v in fit_res.metrics.items() if k not in {'encrypted_ordering', 'encrypted_beneficiary'}}
                    self.encrypted_ordering = fit_res.metrics['encrypted_ordering']
                    self.encrypted_beneficiary = fit_res.metrics['encrypted_beneficiary']

                    private_key = read_private(self.server_dir)
                    sym_key_swift = asym_decrypt(private_key, self.encrypted_swift_sym_key['server'])
                    with open(os.path.join(self.server_dir, "swift_symmetric.key"), "wb") as f:
                        f.write(sym_key_swift)
                else:
                    server_symmetric_key = load_server_symmetric_key(self.server_dir)
                    parameters = copy.copy(fit_res.parameters)
                    parameters.tensors = [sym_decrypt(server_symmetric_key, x) for x in parameters.tensors]
                    decrypted_weights = parameters_to_ndarrays(parameters)
                    weights_results.append((decrypted_weights, fit_res.num_examples))
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
            return parameters_aggregated, {}
        elif server_round < GLOBAL_EPOCH:
            server_symmetric_key = load_server_symmetric_key(self.server_dir)
            weights_results: List[Tuple[List[np.ndarray], int]] = []
            for client_proxy, fit_res in results:
                parameters = copy.copy(fit_res.parameters)
                parameters.tensors = [sym_decrypt(server_symmetric_key, x) for x in parameters.tensors]
                decrypted_weights = parameters_to_ndarrays(parameters)
                weights_results.append((decrypted_weights, fit_res.num_examples))
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))
            return parameters_aggregated, {}
        elif server_round == GLOBAL_EPOCH:
            for client_proxy, fit_res in results:
                cid = client_proxy.cid
                self.ordering_account_bytes = self.ordering_account_bytes | {cid: fit_res.metrics['ordering_account_bytes']}
                self.beneficiary_account_bytes = self.beneficiary_account_bytes | {cid: fit_res.metrics['beneficiary_account_bytes']}
                self.ordering_encoder_bytes = self.ordering_encoder_bytes | {cid: fit_res.metrics['ordering_encoder_bytes']}
                self.beneficiary_encoder_bytes = self.beneficiary_encoder_bytes | {cid: fit_res.metrics['beneficiary_encoder_bytes']}
            return empty_parameters(), {}
        elif server_round == GLOBAL_EPOCH + 1:
            # Since there is only one SWIFT client, no aggregation is needed
            return empty_parameters(), {}
        else:
            raise ValueError(f"Round {server_round} should not exist.")

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Not running any federated evaluation."""
        return []

    def aggregate_evaluate(self, server_round, results, failures):
        """Not aggregating any evaluation."""
        return None

    def evaluate(self, server_round, parameters):
        """Not running any centralized evaluation."""
        return None


def train_strategy_factory(server_dir: Path):
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federated learning rounds to run.
    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.
    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    training_strategy = TrainStrategy(server_dir=server_dir)
    num_rounds = GLOBAL_EPOCH + 1
    return training_strategy, num_rounds


class TestSwiftClient(fl.client.Client):
    def __init__(self, cid, swift_df, model, client_dir, preds_format_path, preds_dest_path):
        super(TestSwiftClient, self).__init__()
        self.cid = cid
        self.swift_df = swift_df
        self.model = model
        self.client_dir = client_dir
        self.preds_format_path = preds_format_path
        self.preds_dest_path = preds_dest_path

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"Swift client evaluate round {ins.config['round']} start...")
        if ins.config['round'] == 1:
            swift_symmetric_key = load_swift_symmetric_key(self.client_dir)

            encrypted_ordering = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(self.swift_df['OrderingAccount'].values.astype('<U32')))
            encrypted_beneficiary = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(self.swift_df['BeneficiaryAccount'].values.astype('<U32')))

            evaluate_res = EvaluateRes(status=Status(code=Code.OK, message="Success"),
                                       loss=0,
                                       num_examples=len(self.swift_df),
                                       metrics={'encrypted_ordering': encrypted_ordering,
                                                'encrypted_beneficiary': encrypted_beneficiary})
            logger.info("*" * 50 + f"evaluate_res size of round {ins.config['round']} from {self.cid}: {sys.getsizeof(evaluate_res) / 1024 ** 3}")
            return evaluate_res
        elif ins.config['round'] == 2:
            raise ValueError("Swift client should not be selected in round 1")
        elif ins.config['round'] == 3:
            swift_symmetric_key = load_swift_symmetric_key(self.client_dir)

            ordering_account = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['ordering_account_bytes']), dtype='<U32')
            beneficiary_account = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['beneficiary_account_bytes']), dtype='<U32')
            ordering_encoder = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['ordering_encoder_bytes']), dtype='float16')
            beneficiary_encoder = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, ins.config['beneficiary_encoder_bytes']), dtype='float16')

            ordering_ndarray = np.concatenate([ordering_account.reshape(-1, 1), ordering_encoder.reshape(-1, ENCODER_DIM)], axis=1)
            beneficiary_ndarray = np.concatenate([beneficiary_account.reshape(-1, 1), beneficiary_encoder.reshape(-1, ENCODER_DIM)], axis=1)

            ordering_df = pd.DataFrame(ordering_ndarray, columns=['OrderingAccount'] + ['OEncoder' + str(x) for x in range(ordering_ndarray.shape[1] - 1)])
            beneficiary_df = pd.DataFrame(beneficiary_ndarray, columns=['BeneficiaryAccount'] + ['BEncoder' + str(x) for x in range(beneficiary_ndarray.shape[1] - 1)])

            for column in ordering_df.columns:
                if column != 'OrderingAccount':
                    ordering_df[column] = ordering_df[column].astype('float16')

            for column in beneficiary_df.columns:
                if column != 'BeneficiaryAccount':
                    beneficiary_df[column] = beneficiary_df[column].astype('float16')

            swift_df = pd.merge(self.swift_df, ordering_df, on='OrderingAccount', how='left')
            swift_df = pd.merge(swift_df, beneficiary_df, on='BeneficiaryAccount', how='left')

            MessageId, X, Y = merged_swift_df_to_XY(swift_df, self.client_dir)

            y_pred = self.model.dp_ebm.predict_proba(X)
            result = pd.DataFrame()
            result['MessageId'] = MessageId
            result['Score'] = y_pred[:, 1]

            preds_format_df = pd.read_csv(self.preds_format_path)
            formatted_result = pd.merge(preds_format_df[['MessageId']], result, on='MessageId', how='left')
            formatted_result.to_csv(self.preds_dest_path, index=False)

            return EvaluateRes(status=Status(code=Code.OK, message="Success"),
                               loss=0,
                               num_examples=len(self.swift_df),
                               metrics={})
        else:
            raise ValueError(f"Round {ins.config['round']} should not exist.")


class TestBankClient(fl.client.Client):
    def __init__(self, cid, bank_df, model, client_dir):
        super(TestBankClient, self).__init__()
        self.cid = cid
        self.bank_df = bank_df
        self.model = model
        self.client_dir = client_dir
        self.X = one_hot_encoder(bank_df['Flags'])

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"Bank client (cid: {self.cid}) evaluate round {ins.config['round']} start...")
        if ins.config['round'] == 1:
            raise ValueError("Bank clients should not be selected in round 1")
        elif ins.config['round'] == 2:
            encrypted_ordering = ins.config['encrypted_ordering']
            encrypted_beneficiary = ins.config['encrypted_beneficiary']

            swift_symmetric_key = load_swift_symmetric_key(self.client_dir)
            decrypted_ordering = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, encrypted_ordering), dtype='<U32')
            decrypted_beneficiary = bytes_to_ndarray(sym_decrypt(swift_symmetric_key, encrypted_beneficiary), dtype='<U32')

            ordering_df = pd.DataFrame(decrypted_ordering, columns=['Account'])
            beneficiary_df = pd.DataFrame(decrypted_beneficiary, columns=['Account'])

            columns = [0, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11]
            bank_flag = pd.DataFrame(self.X.numpy().astype(int), columns=columns)
            bank_flag['Account'] = self.bank_df['Account']

            ordering_df = pd.merge(ordering_df, bank_flag, on='Account', how='inner')
            beneficiary_df = pd.merge(beneficiary_df, bank_flag, on='Account', how='inner')

            # get encrypted (with swift's symmetric key) encoder
            new_model = model_encoder(self.model)
            ordering_encoder = new_model(torch.Tensor(ordering_df.drop(columns=['Account']).values))
            beneficiary_encoder = new_model(torch.Tensor(beneficiary_df.drop(columns=['Account']).values))

            ordering_encoder_array = ordering_encoder.detach().numpy()
            beneficiary_encoder_array = beneficiary_encoder.detach().numpy()

            ordering_encoder_array_account = ordering_df['Account'].values.astype('<U32')
            beneficiary_encoder_array_account = beneficiary_df['Account'].values.astype('<U32')
            ordering_encoder_array_encoder = ordering_encoder_array.astype('float16')
            beneficiary_encoder_array_encoder = beneficiary_encoder_array.astype('float16')

            ordering_account_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(ordering_encoder_array_account))
            beneficiary_account_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(beneficiary_encoder_array_account))
            ordering_encoder_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(ordering_encoder_array_encoder))
            beneficiary_encoder_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(beneficiary_encoder_array_encoder))

            evaluate_res = EvaluateRes(status=Status(code=Code.OK, message="Success"),
                                       loss=0,
                                       num_examples=len(self.X),
                                       metrics={'ordering_account_bytes': ordering_account_bytes,
                                                'beneficiary_account_bytes': beneficiary_account_bytes,
                                                'ordering_encoder_bytes': ordering_encoder_bytes,
                                                'beneficiary_encoder_bytes': beneficiary_encoder_bytes})
            logger.info("=" * 50 + f"evaluate_res size of round {ins.config['round']} from {self.cid}: {sys.getsizeof(evaluate_res) / 1024 ** 3}")
            return evaluate_res

        elif ins.config['round'] == 3:
            raise ValueError("Bank clients should not be selected in round 2")
        else:
            raise ValueError(f"Round {ins.config['round']} should not exist.")


def test_client_factory(cid: str, data_path: Path, client_dir: Path, preds_format_path: Path, preds_dest_path: Path):
    if cid == "swift":
        logger.info("-" * 50 + f"Test: Initialing Swift Client for {cid}")
        swift_df = pd.read_csv(data_path)
        model = pickle.load(open(os.path.join(client_dir, 'SwiftModel.pkl'), 'rb'))
        return TestSwiftClient(cid, swift_df=swift_df, model=model, client_dir=client_dir, preds_format_path=preds_format_path, preds_dest_path=preds_dest_path)
    else:
        logger.info("-" * 50 + f"Test: Initialing Bank Client for {cid}")
        bank_df = pd.read_csv(data_path)
        model = pickle.load(open(os.path.join(client_dir, 'BankModel.pkl'), 'rb'))
        return TestBankClient(cid, bank_df, model, client_dir)


class TestStrategy(fl.server.strategy.Strategy):
    def __init__(self, server_dir: Path):
        super(TestStrategy, self).__init__()
        self.server_dir = server_dir
        self.encrypted_ordering = None
        self.encrypted_beneficiary = None
        self.ordering_account_bytes = dict()
        self.beneficiary_account_bytes = dict()
        self.ordering_encoder_bytes = dict()
        self.beneficiary_encoder_bytes = dict()

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters:
        """Do nothing. Return empty Flower Parameters dataclass."""
        return initialize_autoencoder_parameters()

    def configure_fit(self, server_round, parameters, client_manager):
        """Not running any federated fitting."""
        return []

    def aggregate_fit(self, server_round, results, failures):
        """Not aggregating any fitting."""
        return []

    def configure_evaluate(self, server_round: int, parameters: Parameters, client_manager: ClientManager) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"configure_evaluate in round {server_round} start...")
        if server_round == 1:
            parameters = empty_parameters()
            config_dict = {"round": server_round}
            swift_evaluate_ins = EvaluateIns(parameters, config_dict)

            client_dict: Dict[str, ClientProxy] = client_manager.all()

            evaluate_config = [(v, swift_evaluate_ins) for k, v in client_dict.items() if k == "swift"]
            return evaluate_config
        elif server_round == 2:
            client_dict: Dict[str, ClientProxy] = client_manager.all()
            config_dict = {"round": server_round, "encrypted_ordering": self.encrypted_ordering, "encrypted_beneficiary": self.encrypted_beneficiary}
            bank_evaluate_ins = EvaluateIns(parameters, config_dict)
            evaluate_config = [(v, bank_evaluate_ins) for k, v in client_dict.items() if k != "swift"]
            return evaluate_config
        elif server_round == 3:
            client_dict: Dict[str, ClientProxy] = client_manager.all()
            swift_symmetric_key = load_swift_symmetric_key(self.server_dir)

            dict1 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='<U32') for k, v in self.ordering_account_bytes.items()}
            dict2 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='<U32') for k, v in self.beneficiary_account_bytes.items()}
            dict3 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='float16') for k, v in self.ordering_encoder_bytes.items()}
            dict4 = {k: bytes_to_ndarray(sym_decrypt(swift_symmetric_key, v), dtype='float16') for k, v in self.beneficiary_encoder_bytes.items()}

            array1 = np.concatenate(list(dict1.values()), axis=0)
            array2 = np.concatenate(list(dict2.values()), axis=0)
            array3 = np.concatenate(list(dict3.values()), axis=0)
            array4 = np.concatenate(list(dict4.values()), axis=0)

            ordering_ndarray = np.concatenate([array1.reshape(-1, 1), array3.reshape(-1, ENCODER_DIM)], axis=1)
            beneficiary_ndarray = np.concatenate([array2.reshape(-1, 1), array4.reshape(-1, ENCODER_DIM)], axis=1)

            ordering_df = pd.DataFrame(ordering_ndarray, columns=['OrderingAccount'] + ['OEncoder' + str(x) for x in range(ordering_ndarray.shape[1] - 1)])
            beneficiary_df = pd.DataFrame(beneficiary_ndarray, columns=['BeneficiaryAccount'] + ['BEncoder' + str(x) for x in range(beneficiary_ndarray.shape[1] - 1)])

            for column in ordering_df.columns:
                if column != 'OrderingAccount':
                    ordering_df[column] = ordering_df[column].astype('float16')

            for column in beneficiary_df.columns:
                if column != 'BeneficiaryAccount':
                    beneficiary_df[column] = beneficiary_df[column].astype('float16')

            ordering_df = ordering_df.groupby(by='OrderingAccount').agg(np.nanmean).reset_index()
            beneficiary_df = beneficiary_df.groupby(by='BeneficiaryAccount').agg(np.nanmean).reset_index()

            ordering_account_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(ordering_df['OrderingAccount'].values.astype('<U32')))
            beneficiary_account_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(beneficiary_df['BeneficiaryAccount'].values.astype('<U32')))
            ordering_encoder_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(ordering_df.drop(columns=['OrderingAccount']).values.astype('float16')))
            beneficiary_encoder_bytes = sym_encrypt(swift_symmetric_key, ndarray_to_bytes(beneficiary_df.drop(columns=['BeneficiaryAccount']).values.astype('float16')))

            config_dict = {"round": server_round,
                           'ordering_account_bytes': ordering_account_bytes,
                           'beneficiary_account_bytes': beneficiary_account_bytes,
                           'ordering_encoder_bytes': ordering_encoder_bytes,
                           'beneficiary_encoder_bytes': beneficiary_encoder_bytes}

            swift_evaluate_ins = EvaluateIns(empty_parameters(), config_dict)
            evaluate_config = [(v, swift_evaluate_ins) for k, v in client_dict.items() if k == "swift"]
            logger.info("*" * 50 + f"evaluate_config size of round {server_round} from server: {sys.getsizeof(evaluate_config) / 1024 ** 3}GB")
            return evaluate_config
        else:
            raise ValueError(f"Round {server_round} should not exist.")

    def aggregate_evaluate(self, server_round: int, results: List[Tuple[ClientProxy, FitRes]], failures) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """
        Please refer to the Communication Flow for details of different rounds.
        """
        logger.info("-" * 50 + f"aggregate_evaluate in round {server_round} start...")
        if server_round == 1:
            for client_proxy, evaluate_res in results:
                cid = client_proxy.cid
                if cid == "swift":
                    self.encrypted_ordering = evaluate_res.metrics['encrypted_ordering']
                    self.encrypted_beneficiary = evaluate_res.metrics['encrypted_beneficiary']
            return None, {}
        elif server_round == 2:
            for client_proxy, evaluate_res in results:
                cid = client_proxy.cid
                self.ordering_account_bytes = self.ordering_account_bytes | {cid: evaluate_res.metrics['ordering_account_bytes']}
                self.beneficiary_account_bytes = self.beneficiary_account_bytes | {cid: evaluate_res.metrics['beneficiary_account_bytes']}
                self.ordering_encoder_bytes = self.ordering_encoder_bytes | {cid: evaluate_res.metrics['ordering_encoder_bytes']}
                self.beneficiary_encoder_bytes = self.beneficiary_encoder_bytes | {cid: evaluate_res.metrics['beneficiary_encoder_bytes']}
            return None, {}
        elif server_round == 3:
            return None, {}
        else:
            raise ValueError(f"Round {server_round} should not exist.")

    def evaluate(self, server_round, parameters):
        """Not running any centralized evaluation."""
        return None


def test_strategy_factory(server_dir: Path):
    """
    Factory function that instantiates and returns a Flower Strategy, plus the
    number of federation rounds to run.
    Args:
        server_dir (Path): Path to a directory specific to the server/aggregator
            that is available over the simulation. The server can use this
            directory for saving and reloading server state. Using this
            directory is required for the trained model to be persisted between
            training and test stages.
    Returns:
        (Strategy): Instance of Flower Strategy.
        (int): Number of federated learning rounds to execute.
    """
    test_strategy = TestStrategy(server_dir=server_dir)
    num_rounds = 3
    return test_strategy, num_rounds
