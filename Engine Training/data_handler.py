import pickle as pkl
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm import tqdm
import torch
from utils import ListUtils as LU
import et_literals as l
import sys

class HumanDataHandler:
    """
    This class is responsible for loading the data and preparing it for the model.
    """

    def __init__(self, pairs_path: str, user_results_path: str, samples_path: str = None, filters: dict = None):
        self.pairs_path: str = pairs_path  # Path to the pairs (questions) file
        self.user_results_path = user_results_path  # Path to the user results file
        self.samples_path = samples_path  # Path to the samples file
        self.filters = filters  # Filters to apply on the data

        self.merged_data: pd.DataFrame = None  # The merged data - users results with questions data
        self.samples: np.ndarray = None  # list of sample (dict)
        self.samples_index: pd.DataFrame = None  # For each user (index) and question (columns) - the index of the sample in the list.

        self.decision_features_dim: int = None  # The dimension of the decision features
        self.ada_v2_dim = None  # The dimension of the question features

        self.load_data()

    def load_data(self) -> None:
        """
        Load the data from the files and filter and merge them
        :return: None
        """
        with open(self.pairs_path, 'rb') as f:
            pairs = pkl.load(f)
        with open(self.user_results_path, 'rb') as f:
            data = pkl.load(f)
        if self.filters is not None:
            for k, v in self.filters.items():
                data = data[data[k].isin(v if type(v) == list else [v])]
                pairs = pairs[pairs[k].isin(v if type(v) == list else [v])]

        pairs = pairs.set_index(['sch_id_1', 'sch_id_2'])
        merged = data.join(pairs[['ada_v2']], on=['sch_id_1', 'sch_id_2'], how='left')
        self.merged_data = merged

    def merged_to_samples(
            self,
            torch_samples: bool = False,
            device: torch.device = None,
            load_samples: bool = False,
            save_samples: bool = False,
    ):
        """
        Convert the merged data to samples.
        samples is a dictionary with user_id as key and a list of samples as value.
        Each sample is a tuple of (history, next_question, real_conf, meta_data)
        :param: torch_samples: Whether to convert the samples to torch tensors
        :return: None
        """
        user_ids = self.merged_data['user_id'].unique()
        samples = []
        samples_index = pd.DataFrame(index=user_ids, columns=range(2, self.merged_data['order'].max() + 1))
        first_load = True
        for user_id in tqdm(user_ids):
            v = self.merged_data[self.merged_data['user_id'] == user_id].sort_values(by='order')
            for i in range(2, v['order'].max() + 1):
                if load_samples:
                    sample = torch.load(f"{self.samples_path}/{user_id}_{i}.pt")
                    if first_load:
                        first_load = False
                        self.ada_v2_dim = sample[l.NEXT_QUESTION].shape[0]
                        self.decision_features_dim = sample[l.HISTORY].shape[1] - self.ada_v2_dim

                else:
                    sample_df = v[v['order'] <= i]
                    sample = self.head_to_sample(sample_df, torch_samples=torch_samples, device=device)
                    if save_samples:
                        torch.save(sample, f"{self.samples_path}/{user_id}_{i}.pt")

                samples.append(sample)
                samples_index.loc[user_id, i] = len(samples) - 1
        self.samples = np.array(samples)
        self.samples_index = samples_index

    def __getitem__(self, item) -> list:
        """
        Get the samples of a specific user, if the samples are not yet created - create them
        :param item: The user id
        :return: A list of samples for the user
        """
        if self.samples is None:
            self.merged_to_samples()

        if isinstance(item, int):
            # item is a user id
            if item not in self.samples_index.index:
                raise IndexError(f'User {item} not in data')
            return self.samples[self.samples_index.loc[item].tolist()].tolist()

        elif isinstance(item, tuple):
            # item is a tuple of (user_id, question_id)
            if item[0] not in self.samples_index.index:
                raise IndexError(f'User {item[0]} not in data')
            if item[1] not in self.samples_index.columns:
                raise IndexError(f'Question {item[1]} not in data')
            return self.samples[self.samples_index.loc[item[0], item[1]]]

    def head_to_sample(self, df: pd.DataFrame, torch_samples: bool = False, device: torch.device = None) -> dict:
        """
        Convert the head of a user's data to a sample, consisting of:
        1. history - previous decisions featues and question question_embeddings
        2. next_question - embedding
        3. real_conf - the real answer for the next question
        4. user_ans_is_match - whether the user's answer was a match
        4. meta_data - which users, which question, etc.
        :param df: the prefix of the user's data
        :param torch_samples: Whether to convert the sample to torch tensors
        :return: tuple of (history, next_question, real_conf, meta_data)
        """
        sample = {}
        last_row = df.iloc[-1, :]
        df = df.iloc[:-1, :]
        decision_features = self.get_decision_features(df)
        previous_questions = np.array(df['ada_v2'].to_list())

        if not self.ada_v2_dim:
            self.ada_v2_dim = previous_questions.shape[1]

        sample[l.HISTORY] = np.concatenate([previous_questions, decision_features], axis=1)
        sample[l.NEXT_QUESTION] = np.array(last_row['ada_v2'])
        sample[l.NQ_REAL_CONF] = last_row['realconf']
        sample[l.NQ_IS_MATCH] = last_row['user_ans_is_match']
        sample[l.USERCONF] = last_row[l.USERCONF]

        sample[l.META_DATA] = {'user_id': last_row['user_id'],
                               'order': last_row['order'],
                               'history_acc': (df['user_ans_is_match'] == df['realconf']).mean()}

        if torch_samples:
            for i in [l.HISTORY, l.NEXT_QUESTION]:
                if type(sample[i]) == np.ndarray:
                    sample[i] = torch.from_numpy(sample[i]).float().to(device=device)

        return sample

    def get_decision_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get the decision features from the data.
        There could be several ways to do so, and each one defines a different decision features dimension.
        Currently, there's only one way implemented.
        :param df: The user's data
        :return:
        """
        decision_deatures = df[
            ['userconf', 'user_sub_val', 'user_ans_is_match', 'time', 'token_path', 'term_match', 'word_net']].values

        if not self.decision_features_dim:
            self.decision_features_dim = decision_deatures.shape[1]

        return decision_deatures

    def matchers_split(self, melted_samples_index, train_portion: float = 0.8):
        """
        Split the data by matchers.
        :param melted_samples_index: The melted samples index
        :param train_portion: The portion of the data to be in the train set.
        :return: train and test sets
        """
        matcher_ids = melted_samples_index['user_id'].unique()
        m_train_set, m_test_set = train_test_split(matcher_ids, train_size=train_portion)
        return m_train_set.tolist(), m_test_set.tolist()

    def questions_split(self, melted_samples_index, method, param):
        """
        Split the data by questions.
        :param melted_samples_index: The melted samples index
        :param method: One of [l.RANDOM, l.LESS]:
        l.RANDOM - some questions are in train and some are in test, with train portion of param
        l.LESS - all questions with order <= param are in train, and the rest are in test.
        :param param: The parameter for the method
        :return: train and test sets
        """
        assert method in [l.LESS, l.RANDOM], f'Unknown split by question method: {method}'
        train_set, test_set = [], []
        if method == l.LESS:
            train_set = melted_samples_index[melted_samples_index['order'] <= param]['pair']
            test_set = melted_samples_index[melted_samples_index['order'] > param]['pair']

        elif method == l.RANDOM:
            train_set, test_set = train_test_split(melted_samples_index['pair'], train_size=param)

        return train_set.tolist(), test_set.tolist()

    def train_test_split(self,
                         split_by: str = l.RANDOM,
                         train_portion: float = 0.8,
                         by_question: tuple = ('less', 30),
                         get_data: bool = False):
        """
        Split the data to train and test sets.
        :param split_by: One of [l.RANDOM, l.MATCHERS, l.QUESTIONS]:
        l.RANDOM - each sample is either in train or in test.
        l.MATCHERS - each matcher (all of its decisions [=samples]) is either in train or in test .
        l.QUESTIONS - each question (all the decisions [=samples] of the question) is either in train or in test.
        :param train_portion: for split_by = l.RANDOM or l.MATCHERS - the portion of the data to be in the train set.
        :param by_question: the method to split by questions, if split_by = l.QUESTIONS:
        :param get_data: Whether to return the data or the indices
        :return: train and test sets
        """
        assert split_by in [l.RANDOM, l.MATCHERS, l.QUESTIONS], f'Unknown split method: {split_by}'

        melted_samples_index = self.samples_index.reset_index().melt("index")
        melted_samples_index = melted_samples_index.rename(
            columns={'index': 'user_id', 'variable': 'order', 'value': 'index'})
        melted_samples_index = melted_samples_index[melted_samples_index['index'].notna()]
        melted_samples_index['pair'] = list(zip(melted_samples_index['user_id'], melted_samples_index['order']))

        by_question_all = (split_by == 'questions') and (by_question[0] == 'random') and (by_question[1] == 1)
        if (train_portion == 1 and split_by in [l.RANDOM, l.MATCHERS]) or by_question_all:
            train_set, test_set = melted_samples_index['pair'].tolist(), []

        else:
            if split_by == l.RANDOM:
                train_set, test_set = train_test_split(melted_samples_index['pair'], train_size=train_portion)
                train_set, test_set = train_set.tolist(), test_set.tolist()

            elif split_by == l.MATCHERS:
                train_set, test_set = self.matchers_split(melted_samples_index, train_portion=train_portion)

            elif split_by == l.QUESTIONS:
                train_set, test_set = self.questions_split(melted_samples_index, *by_question)

        if get_data:
            train_data = LU.shufflee([self[sample] for sample in train_set])
            test_data = LU.shufflee([self[sample] for sample in test_set])
            return train_data, test_data

        return train_set, test_set

    @staticmethod
    def process_batch(batch, device: torch.device = None):
        """
        Process a batch of samples to be fed to the model.
        :param device: The device to use
        :param batch: list of samples
        :return: tuple of (histories, next_questions, real_confs, lengths, meta_data)
        """
        d_batch = {k: [] for k in batch[0].keys()}
        for sample in batch:
            for k, v in sample.items():
                d_batch[k].append(v)

        out = dict()
        out[l.LENGTHS] = torch.tensor([len(t) for t in d_batch[l.HISTORY]])
        out[l.HISTORIES] = pack_padded_sequence(pad_sequence(d_batch[l.HISTORY]), out[l.LENGTHS], enforce_sorted=False)
        out[l.NEXT_QUESTIONS] = torch.stack(d_batch[l.NEXT_QUESTION])
        out[l.NQ_REAL_CONFS] = torch.tensor(d_batch[l.NQ_REAL_CONF], dtype=torch.float, device=device)
        out[l.NQ_IS_MATCHES] = torch.tensor(d_batch[l.NQ_IS_MATCH], dtype=torch.float, device=device)
        out[l.USERCONF] = torch.tensor(d_batch[l.USERCONF], dtype=torch.float, device=device)
        out[l.META_DATAS] = d_batch[l.META_DATA]

        return out
