import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from tqdm import tqdm
import torch
from utils import ListUtils as LU
import et_literals as l
import json
import sys

from openai import AzureOpenAI
import os

sys.path.append("Evaluation Framework")
from strategy import Strategy
from decision_mediator import DecisionMediator
import ef_literals as efl
from inputs_handler import InputsHandler
from experiment_manager import ExperimentManager
from sample_selector import sample_selectors

sys.path.append("Chat Protocol")


class ChatbotDataHandler:
    def __init__(self, pairs_path: str,
                 chat_samples_path: str,
                 filters: dict,
                 protocols_path: str,
                 chat_protocol: str):

        self.ih = InputsHandler(engines_path="", engine_name="",
                                protocols_path=protocols_path,
                                protocol_name=chat_protocol,
                                embedded_pairs_path=pairs_path)
        # Questions
        self.pairs_path = pairs_path
        self.question_pool = self.ih.load_question_pool(filters=filters)

        # Protocol
        self.protocols_path = protocols_path
        self.chat_protocol = chat_protocol

        # Data
        self.samples_path = f"{chat_samples_path}/{self.chat_protocol}"
        self.existing_samples = pd.read_pickle(f"{self.samples_path}/samples.pkl")
        self.agents_samples = pd.read_pickle(f"{self.samples_path}/agents_samples.pkl")
        self.samples_selector = None

        if not os.path.exists(self.samples_path):
            os.makedirs(self.samples_path)
        self.merged_data: pd.DataFrame = None  # The merged data - users results with questions data
        self.samples: np.ndarray = None  # list of sample (dict)
        self.samples_index: pd.DataFrame = None  # For each user (index) and question (columns) - the index of the sample in the list.

        # Params
        self.decision_features_dim: int = None  # The dimension of the decision features
        self.ada_v2_dim = None  # The dimension of the question features

    def update_existing_samples(self, existing_samples):
        self.existing_samples = existing_samples
        self.samples_selector.existing_samples = existing_samples

    def update_agents_samples(self, agents_samples):
        self.agents_samples = agents_samples
        self.samples_selector.agents_samples = agents_samples

    def prepare_data(self, sample_selector_method, sample_selector_args, creation_dict):
        # Get samples indices
        sample_selector_args[l.EXISTING_SAMPLES] = self.existing_samples
        sample_selector_args[l.AGENTS_SAMPLES] = self.agents_samples

        # Set sample selector and select samples
        questions = self.question_pool['order'].tolist()
        self.samples_selector = sample_selectors[sample_selector_method](questions=questions, **sample_selector_args)
        samples = self.samples_selector.select()

        # Which samples already exist, and add the missing ones
        samples_df = self.samples_selector.samples_to_df(samples)
        if sample_selector_method == l.SS.RANDOM_EXISTING and not sample_selector_args[l.CREATE_MISSING]:
            assert samples_df['Exist'].all(), "Not all samples exist"
        missing_samples = samples_df[samples_df['sample_id'].isna()]
        # In case we randomly generated not-familiar samples:
        if len(missing_samples) > 0:
            missing_samples = self.build_missing_samples_index(missing_samples)
            self.update_existing_samples(existing_samples=pd.concat([self.existing_samples, missing_samples],
                                                                    ignore_index=True))
            # Assert no missing samples and save
            samples_df = self.samples_selector.samples_to_df(samples)
            assert samples_df['sample_id'].isna().sum() == 0, "There are missing samples"
            self.existing_samples.to_pickle(f"{self.samples_path}/samples.pkl")

        missing_agents_samples = samples_df[~samples_df['Exist']].drop(columns='Exist')
        # In case we randomly generated not-familiar agents-samples paths
        if len(missing_agents_samples) > 0:
            missing_agents_samples = self.create_missing_samples(missing_agents_samples=missing_agents_samples,
                                                                 creation_dict=creation_dict)
            samples_df = self.samples_selector.samples_to_df(samples)
            assert samples_df['Exist'].all(), "Not all samples exist"
            self.existing_samples.to_pickle(f"{self.samples_path}/samples.pkl")
            self.agents_samples.to_pickle(f"{self.samples_path}/agents_samples.pkl")

        self.load_samples(samples_df)

    def create_missing_samples(self, missing_agents_samples, creation_dict):
        client = AzureOpenAI(api_key=os.getenv("OPENAI_API_KEY"),
                             api_version="2023-12-01-preview",
                             azure_endpoint="https://baropenairesource.openai.azure.com/")
        decision_mediator = self.load_decision_mediator(creation_dict['decision_mediator_method'], client=client)

        missing_agents_samples['facilitator'] = self.ih.load_protocol(client=client,
                                                                      strings_path=creation_dict['strings_path'],
                                                                      wandb_track=creation_dict['wandb_track'],
                                                                      seeds=missing_agents_samples['agent'].tolist())

        manager = ExperimentManager(question_pool=self.question_pool,
                                    inputs_handler=self.ih,
                                    decision_mediator=decision_mediator,
                                    demo_strategy=None)

        results = []
        for i, row in enumerate(missing_agents_samples.iterrows()):
            # Extract row's instances
            row = row[1]
            questions_list, sample_id, facilitator = row['questions_list'], row['sample_id'], row['facilitator']

            # Relevant questions and existing prefix sample
            questions_to_ask = questions_list.copy()
            prefix_sample = self.get_prefix_sample(agent=row['agent'],
                                                   questions_list=questions_list)

            # Step 1 - Reset the facilitator (past questions)
            asked_questions = None
            if prefix_sample is not None:
                facilitator.reset_facilitator(checkpoint=prefix_sample['facilitator_status'])
                prefix_asked = prefix_sample['questions_list']
                questions_to_ask = questions_to_ask[len(prefix_asked):]
                asked_questions = prefix_sample['facilitator_status']['strategy_asked_questions']

            # Step 2 - Define the strategy (future questions only if there's a prefix sample, else all questions)
            strategy = Strategy(question_pool=self.question_pool,
                                first_question_method=efl.FQS.GIVEN_ORDER,
                                next_question_method=efl.NQS.GIVEN_ORDER,
                                fq_kwargs={'order': questions_to_ask},
                                nq_kwargs={'order': questions_to_ask},
                                asked_questions=asked_questions)

            # Step 3 - Run the experiment
            questions_to_ask_df = self.question_pool[self.question_pool['order'].isin(questions_to_ask)]
            error_count = manager.run_experiment(facilitator=facilitator,
                                                 strategy=strategy,
                                                 remaining_questions=questions_to_ask_df,
                                                 introduction=prefix_sample is None)
            results.append(error_count)
            if facilitator.dead_facilitator:
                continue

            # Step 4 - Generate the sample, and all the prefixes samples
            samples = self.generate_samples(prefix_sample=prefix_sample,
                                            sample_row=row,
                                            strategy=strategy,
                                            torch_samples=creation_dict[
                                                'torch_samples'],
                                            device=creation_dict['device'])

            # Step 5 - Save the samples
            if i % 10 == 0:
                print(f"\nSaving index... ({i + 1}/{len(missing_agents_samples)})", end=' ')
                self.existing_samples.to_pickle(f"{self.samples_path}/samples.pkl")
                self.agents_samples.to_pickle(f"{self.samples_path}/agents_samples.pkl")
            for name, sample in samples.items():
                torch.save(sample['sample'], sample['path'])
            print()

        missing_agents_samples['error_counts'] = results
        return missing_agents_samples

    def get_prefix_sample(self, agent, questions_list):
        # Check if subsample exists
        for sub in range(len(questions_list) - 1, 1, -1):
            sub_sample = questions_list[:sub]
            subsample_code = "_".join(map(str, sub_sample))
            match = self.existing_samples[self.existing_samples['sample_code'] == subsample_code]
            if len(match) > 0:
                match_id = match.iloc[0]['sample_id']
                agent_match = self.agents_samples[
                    (self.agents_samples['agent_id'] == agent) & (self.agents_samples['sample_id'] == match_id)]
                if len(agent_match) > 0:
                    loaded_sample = torch.load(f"{self.samples_path}/{agent}_{match_id}.pt")
                    return loaded_sample

        # Check if first question exists
        i_filepath = f"{self.samples_path}/{agent}_i{questions_list[0]}.pt"
        if os.path.exists(i_filepath):
            loaded_sample = torch.load(i_filepath)
            return loaded_sample
        else:
            return None

    def generate_samples(self, prefix_sample, sample_row, strategy, torch_samples, device):
        # Extract sample_row's instances
        agent, questions_list, name = sample_row['agent'], sample_row['questions_list'], sample_row['name']
        facilitator = sample_row['facilitator']

        # Global objects to use
        samples = {}
        chat_checkpoints = self.break_facilitator_chat(facilitator)
        merged = self.process_facilitator_conversation(facilitator)
        prefix_len = 0 if prefix_sample is None else len(prefix_sample['questions_list'])

        # Saving the initial question sample
        if prefix_len == 0:
            name = f"{agent}_i{questions_list[0]}"
            path = f"{self.samples_path}/{name}.pt"
            sample = {'facilitator_status': {'chat_messages': facilitator.chat_messages[:chat_checkpoints[0]],
                                             'strategy_asked_questions': strategy.asked_questions[:1],
                                             'decisions': facilitator.decisions[:1],
                                             'questions': facilitator.questions[:1]},
                      'questions_list': questions_list[:1], }
            samples[name] = {'path': path, 'sample': sample}

        samples_ids = []
        temp_samples = []
        for i in range(max(prefix_len + 1, 2), len(merged) + 1):
            # Sample id
            samples_ids.append({'questions_list': questions_list[:i],
                                'sample_code': "_".join(map(str, questions_list[:i]))})

            # Sample
            sub_merged = merged.iloc[:i]
            decision_features = np.array([d['decision_features'] for d in strategy.asked_questions[:i]])
            sub_sample = self.head_to_sample(df=sub_merged, decision_features=decision_features[:-1, :],
                                             torch_samples=torch_samples,
                                             device=device)
            # Facilitator status
            sub_chat_messages = facilitator.chat_messages[:chat_checkpoints[i - 1]]

            sample = {'facilitator_status': {'chat_messages': sub_chat_messages,
                                             'strategy_asked_questions': strategy.asked_questions[:i],
                                             'decisions': facilitator.decisions[:i],
                                             'questions': facilitator.questions[:i]},
                      'questions_list': questions_list[:i],
                      'sample': sub_sample}
            temp_samples.append(sample)

        samples_ids = pd.DataFrame(samples_ids)
        samples_ids = samples_ids.join(self.existing_samples.set_index('sample_code')[['sample_id']],
                                       how='left',
                                       on='sample_code')
        missing_samples = samples_ids[samples_ids['sample_id'].isna()]
        missing_samples = self.build_missing_samples_index(missing_samples)
        self.update_existing_samples(existing_samples=pd.concat([self.existing_samples, missing_samples],
                                                                ignore_index=True))

        samples_ids = samples_ids.join(missing_samples.set_index('sample_code')['sample_id'],
                                       how='left', on='sample_code', rsuffix='_missing')
        samples_ids['sample_id'] = (samples_ids[['sample_id', 'sample_id_missing']].
                                    bfill(axis=1)['sample_id'].
                                    astype(int))
        samples_ids = samples_ids.drop(columns='sample_id_missing')
        for i, s in enumerate(temp_samples):
            sid = samples_ids.iloc[i]['sample_id']
            name = f"{agent}_{sid}"
            path = f"{self.samples_path}/{name}.pt"
            samples[name] = {'path': path, 'sample': s, 'agent_id': agent, 'sample_id': sid}

        new_agents_samples = samples_ids[['sample_id']].copy()
        new_agents_samples.loc[:, 'agent_id'] = agent
        new_agents_samples = pd.concat([self.agents_samples, new_agents_samples])
        new_agents_samples = new_agents_samples.reset_index(drop=True)
        self.update_agents_samples(agents_samples=new_agents_samples)

        return samples

    def head_to_sample(self, df, decision_features=None, torch_samples=False, device=None) -> dict:
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
        if decision_features is None:
            decision_features = self.get_decision_features(df)
        if not self.decision_features_dim:
            self.decision_features_dim = decision_features.shape[1]

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
        if 'question_index' in last_row:
            sample[l.META_DATA]['question_index'] = last_row['question_index']

        if torch_samples:
            for i in [l.HISTORY, l.NEXT_QUESTION]:
                if type(sample[i]) == np.ndarray:
                    sample[i] = torch.from_numpy(sample[i]).float().to(device=device)

        return sample

    def load_samples(self, samples_to_load) -> None:
        samples = []
        samples_index = pd.DataFrame(columns=['agent_id', 'sample_id', 'index'])
        first_load = True
        for i, sample_row in tqdm(samples_to_load.iterrows(), total=len(samples_to_load)):
            agent_id, sample_id, name = sample_row['agent'], sample_row['sample_id'], sample_row['name']
            sample = torch.load(f"{self.samples_path}/{name}.pt")['sample']
            if first_load:
                first_load = False
                self.ada_v2_dim = sample[l.NEXT_QUESTION].shape[0]
                self.decision_features_dim = sample[l.HISTORY].shape[1] - self.ada_v2_dim
            samples.append(sample)
            samples_index.loc[len(samples_index)] = [agent_id, sample_id, len(samples) - 1]

        self.samples = np.array(samples)
        self.samples_index = samples_index

    @staticmethod
    def load_decision_mediator(decision_mediator_method, client):
        decision_mediator = DecisionMediator(client=client,
                                             decision_mediator_method=decision_mediator_method)
        return decision_mediator

    def build_missing_samples_index(self, missing_samples):
        missing_samples = missing_samples[['sample_code', 'questions_list']]
        missing_samples = missing_samples.drop_duplicates(subset='sample_code')
        es_len = len(self.existing_samples)
        missing_samples['sample_id'] = np.arange(es_len, es_len + len(missing_samples), dtype=int)
        return missing_samples

    @staticmethod
    def break_facilitator_chat(facilitator):
        full_answers = [a['full_answer'] for a in facilitator.decisions]
        current, checkpoints = 0, []
        for i, c in enumerate(facilitator.chat_messages):
            if c['content'] == full_answers[current]:
                checkpoints.append(i + 1)
                current += 1
        return checkpoints

    @staticmethod
    def process_facilitator_conversation(facilitator):
        for d in facilitator.decisions:
            d['user_id'] = facilitator.seed
        questions = pd.DataFrame.from_dict(facilitator.questions)
        decisions = pd.DataFrame.from_dict(facilitator.decisions)
        merged = questions.merge(decisions, how='left', on='order')
        merged = merged.rename(columns={'order': 'question_index'})
        merged = merged.reset_index(names='order')
        merged['order'] += 1
        merged = merged[['user_id', 'exp_id', 'sch_id_1', 'sch_id_2', 'realConf'] +
                        [c for c in decisions.columns if c not in ['user_id', 'order']] +
                        ['order', 'question_index', 'token_path', 'term_match', 'word_net', 'ada_v2']]
        merged = merged.rename(columns={'realConf': 'realconf',
                                        'normalized_conf': 'userconf',
                                        'sub_val': 'user_sub_val',
                                        'binary_decision': 'user_ans_is_match'},
                               errors='ignore')
        return merged

    def get_decision_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Get the decision features from the data.
        There could be several ways to do so, and each one defines a different decision features dimension.
        Currently, there's only one way implemented.
        :param df: The user's data
        :return:
        """
        decision_features = df[
            ['userconf', 'user_sub_val', 'user_ans_is_match', 'time', 'token_path', 'term_match', 'word_net']].values

        return decision_features

    def matchers_split(self, train_portion: float = 0.8):
        """
        Split the data by matchers.
        :param melted_samples_index: The melted samples index
        :param train_portion: The portion of the data to be in the train set.
        :return: train and test sets
        """
        matcher_ids = self.samples_index['agent_id'].unique()
        train_set, test_set = train_test_split(matcher_ids, train_size=train_portion)
        train_set = self.samples_index[self.samples_index['agent_id'].isin(train_set)]
        test_set = self.samples_index[self.samples_index['agent_id'].isin(test_set)]
        return train_set, test_set

    def train_test_split(self,
                         split_by: str = l.RANDOM,
                         train_portion: float = 0.8,
                         get_data: bool = False,
                         **kwargs):
        """
        Split the data to train and test sets.
        :param split_by: One of [l.RANDOM, l.MATCHERS, l.QUESTIONS]:
        l.RANDOM - each sample is either in train or in test.
        l.MATCHERS - each matcher (all of its decisions [=samples]) is either in train or in test .
        :param train_portion: for split_by = l.RANDOM or l.MATCHERS - the portion of the data to be in the train set.
        :param by_question: the method to split by questions, if split_by = l.QUESTIONS:
        :param get_data: Whether to return the data or the indices
        :return: train and test sets
        """
        # assert split_by in [l.RANDOM, l.MATCHERS, l.QUESTIONS], f'Unknown split method: {split_by}'
        assert split_by in [l.RANDOM, l.MATCHERS], ""

        if split_by == l.RANDOM:
            train_set, test_set = train_test_split(self.samples_index, train_size=train_portion)

        elif split_by == l.MATCHERS:
            train_set, test_set = self.matchers_split(train_portion=train_portion)

        else:
            print(f'Unknown split method: {split_by} (Maybe not supported by a ChatbotDataHandler)')

        if get_data:
            train_data = LU.shufflee(self.samples[train_set['index'].to_list()].tolist())
            test_data = LU.shufflee(self.samples[test_set['index'].to_list()].tolist())
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

    # def merged_to_samples(
    #         self,
    #         torch_samples: bool = False,
    #         device: torch.device = None,
    #         save_samples: bool = False,
    # ):
    #     """
    #     Convert the merged data to samples.
    #     samples is a dictionary with user_id as key and a list of samples as value.
    #     Each sample is a tuple of (history, next_question, real_conf, meta_data)
    #     :param: torch_samples: Whether to convert the samples to torch tensors
    #     :return: None
    #     """
    #     user_ids = self.merged_data['user_id'].unique()
    #     samples = []
    #     samples_index = pd.DataFrame(index=user_ids, columns=range(2, self.merged_data['order'].max() + 1))
    #     for user_id in tqdm(user_ids):
    #         v = self.merged_data[self.merged_data['user_id'] == user_id].sort_values(by='order')
    #         for i in range(2, v['order'].max() + 1):
    #             sample_df = v[v['order'] <= i]
    #             sample = self.head_to_sample(sample_df, torch_samples=torch_samples, device=device)
    #             if save_samples:
    #                 torch.save(sample, f"{self.samples_path}/{user_id}_{i}.pt")
    #
    #             samples.append(sample)
    #             samples_index.loc[user_id, i] = len(samples) - 1
    #     self.samples = np.array(samples)
    #     self.samples_index = samples_index

    # def load_strategy(self):
    #     path = f"{self.strategy_path}/{self.strategy}/config_{self.strategy}.json"
    #     with open(path, 'rb') as cf:
    #         config = json.load(cf)
    #
    #     strategy = Strategy(question_pool=self.question_pool, **config)
    #     assert not strategy.nq_selector.engine_required, "The strategy should not require the engine"
    #     return strategy

    # def process_results(self, results):
    #     seeds = [v['facilitator'].seed for v in results.values()]
    #     seeds_exists = all([seed is not None for seed in seeds])
    #
    #     all_merged = []
    #     for k, v in results.items():
    #         facilitator = v['facilitator']
    #         for d in facilitator.decisions:
    #             d['user_id'] = facilitator.seed if seeds_exists else k
    #         questions = pd.DataFrame.from_dict(facilitator.questions)
    #         decisions = pd.DataFrame.from_dict(facilitator.decisions)
    #         merged = questions.merge(decisions, how='left', on='order')
    #         merged = merged.rename(columns={'order': 'question_index'})
    #         merged = merged.reset_index(names='order')
    #         merged['order'] += 1
    #         merged = merged[['user_id', 'exp_id', 'sch_id_1', 'sch_id_2', 'realConf'] +
    #                         [c for c in decisions.columns if c not in ['user_id', 'order']] +
    #                         ['order', 'question_index', 'token_path', 'term_match', 'word_net', 'ada_v2']]
    #         merged = merged.rename(columns={'realConf': 'realconf',
    #                                         'normalized_conf': 'userconf',
    #                                         'sub_val': 'user_sub_val',
    #                                         'binary_decision': 'user_ans_is_match'},
    #                                errors='ignore')
    #         all_merged.append(merged)
    #
    #     self.merged_data = pd.concat(all_merged)
