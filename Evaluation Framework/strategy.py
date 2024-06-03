import json
import os

import numpy as np
import torch

import ef_literals as l
import sys

sys.path.append('../Engine Training/')
import et_literals as etl


class FirstQuestionSelection:
    """
    The FirstQuestionSelection class is used to select the first question to be asked.
    It implements several methods to select the first question.
    """

    @staticmethod
    def random(question_pool, **kwargs):
        """
        * A FQS method *
        Select the first question randomly
        :param question_pool: DataFrame with the questions to be asked
        """
        return question_pool.sample(1).squeeze()

    @staticmethod
    def original(question_pool, **kwargs):
        """
        * A FQS method *
        Select the first question to be from the original order.
        That is, from the remaining questions, select the one with the lowest order.
        :param question_pool: DataFrame with the questions to be asked
        """
        return question_pool[question_pool[l.ORDER] == question_pool[l.ORDER].min()].squeeze()

    @staticmethod
    def given_order(question_pool, order, **kwargs):
        """
        * A FQS method *
        Select the first question to be from a given order.
        :param question_pool: DataFrame with the questions to be asked
        :param order: The order of the question to be selected
        """
        return question_pool[question_pool['order'] == order[0]].squeeze()

    methods = {l.FQS.RANDOM: random,
               l.FQS.ORIGINAL: original,
               l.FQS.GIVEN_ORDER: given_order}


class NextQuestionSelection:
    """
    The NextQuestionSelection class is used to select the next question to be asked, after the first question.
    It implements several methods to select the next question, and it holds the method to be used.
    """

    def __init__(self, next_question_method):
        """
        :param next_question_method: Method to be used to select the next question
        """
        self.methods = {l.NQS.RANDOM: {'method': self.random, 'engine_required': False},
                        l.NQS.ORIGINAL: {'method': self.original, 'engine_required': False},
                        l.NQS.GIVEN_ORDER: {'method': self.given_order, 'engine_required': False},
                        l.NQS.LOWEST_BINARY_ENTROPY: {'method': self.lowest_binary_entropy, 'engine_required': True},
                        l.NQS.HIGHEST_BINARY_ENTROPY: {'method': self.highest_binary_entropy, 'engine_required': True},
                        l.NQS.HIGHEST_USER_CONFIDENCE: {'method': self.highest_user_confidence,
                                                        'engine_required': True},
                        l.NQS.LOWEST_USER_CONFIDENCE: {'method': self.lowest_user_confidence, 'engine_required': True},
                        l.NQS.LOWEST_TO_BHIGHEST: {'method': self.lowest_to_bhighest, 'engine_required': True},
                        l.NQS.LOWEST_TO_RHIGHEST: {'method': self.lowest_to_rhighest, 'engine_required': True},
                        l.NQS.LOWEST_TO_RLOWEST: {'method': self.lowest_to_rlowest, 'engine_required': True}}

        self.next_question_method = next_question_method
        self.method = self.methods[self.next_question_method]['method']
        self.engine_required = self.methods[self.next_question_method]['engine_required']

        self.engine = None
        self.device = None

    def set_engine(self, engine, device):
        """
        Set the engine and device to be used by the method to select the next question
        :param engine: Engine to be used to predict the next question
        :param device: Device to be used by the engine
        """
        self.engine = engine
        self.device = device

    def choose_next_question(self, current_pool, questions_embedding, decisions_features, **kwargs):
        """
        Choose the next question to be asked:
        1. If the engine is required, assert that it is not None
        2. Apply the method to select the next question
        """
        if self.engine_required:
            assert self.engine is not None, "Engine is required for this method"
            assert self.device is not None, "Device is required for this method"

        question = self.method(current_pool,
                               questions_embedding=questions_embedding,
                               decisions_features=decisions_features,
                               **kwargs)
        return question

    def predict_all_nqs(self, question_pool, questions_embedding, decisions_features):
        """
        Predict the probability of each decision for all the questions in the pool
        :param question_pool: DataFrame with the questions to be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        """
        question_pool = question_pool.copy()
        history = self.process_engine_history(previous_questions=questions_embedding,
                                              decisions_features=decisions_features)
        duplicated_history = history.repeat(len(question_pool), 1, 1)
        duplicated_history = duplicated_history.permute(1, 0, 2)
        nqs = torch.tensor(question_pool['ada_v2'].to_list()).to(device=self.device, dtype=torch.float32)
        prediction = self.predict(history=duplicated_history, nq=nqs)
        if etl.BINARY in prediction:
            question_pool[['p0', 'p1']] = prediction[etl.BINARY]
        if etl.REGRESSION in prediction:
            question_pool['pred_conf'] = prediction[etl.REGRESSION]
        return question_pool

    def process_engine_history(self, previous_questions, decisions_features):
        """
        Craft a torch tensor with the history of the previous questions and decisions features
        :param previous_questions: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        :return: Torch tensor with the history of the previous questions and decisions features
        """
        torch_qe = torch.tensor(previous_questions)
        torch_df = torch.tensor(decisions_features)
        history = torch.concat([torch_qe, torch_df], dim=1).to(device=self.device, dtype=torch.float32)
        return history

    def predict(self, history, nq):
        """
        Use the engine to predict the probability of each decision for the next question
        """
        prediction = self.engine(history, nq)
        prediction = {k: v.detach().cpu().numpy() for k, v in prediction.items()}
        return prediction

    # From this point, the methods to select the next question are implemented

    def random(self, question_pool, **kwargs):
        """
        * A NQS method *
        Select the next question randomly
        :param question_pool: DataFrame with the questions that can be asked
        """
        return question_pool.sample(1).squeeze()

    def original(self, question_pool, **kwargs):
        """
        * A NQS method *
        Select the next question to be from the original order.
        That is, from the remaining questions, select the one with the lowest order.
        :param question_pool: DataFrame with the questions that can be asked
        """
        return question_pool[question_pool[l.ORDER] == question_pool[l.ORDER].min()].squeeze()

    def given_order(self, question_pool, order, **kwargs):
        """
        * A NQS method *
        Select the next question to be from a given order.
        :param question_pool: DataFrame with the questions that can be asked
        :param order: The order of the question to be selected
        """
        question_pool = question_pool[question_pool['order'].isin(order)]
        question_pool = question_pool.sort_values(by='order', key=lambda c: c.map(lambda e: order.index(e)))
        return question_pool.iloc[0]

    def lowest_binary_entropy(self, question_pool, questions_embedding, decisions_features, **kwargs):
        """
        * A NQS method *
        Select the next question to be the one with the highest confidence of one of the binary decisions.
        I.e., the highest probability is in one of the decisions (p0 or p1).
        :param question_pool: DataFrame with the questions that can be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        """
        question_pool = self.predict_all_nqs(question_pool=question_pool,
                                             questions_embedding=questions_embedding,
                                             decisions_features=decisions_features)
        highest = question_pool[['p0', 'p1']].max(axis=1).max()
        highest_question = question_pool[(question_pool['p0'] == highest) | (question_pool['p1'] == highest)].iloc[0]
        return highest_question

    def highest_user_confidence(self, question_pool, questions_embedding, decisions_features, **kwargs):
        """
        * A NQS method *
        Select the next question to be the one with the highest (continuous) user confidence.
        :param question_pool: DataFrame with the questions that can be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        """
        question_pool = self.predict_all_nqs(question_pool=question_pool,
                                             questions_embedding=questions_embedding,
                                             decisions_features=decisions_features)
        highest = question_pool[l.PRED_CONF].max()
        highest_question = question_pool[question_pool[l.PRED_CONF] == highest].iloc[0]
        return highest_question

    def lowest_user_confidence(self, question_pool, questions_embedding, decisions_features, **kwargs):
        """
        * A NQS method *
        Select the next question to be the one with the lowest (continuous) user confidence.
        :param question_pool: DataFrame with the questions that can be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        """
        question_pool = self.predict_all_nqs(question_pool=question_pool,
                                             questions_embedding=questions_embedding,
                                             decisions_features=decisions_features)
        lowest = question_pool[l.PRED_CONF].min()
        lowest_question = question_pool[question_pool[l.PRED_CONF] == lowest].iloc[0]
        return lowest_question

    def highest_binary_entropy(self, question_pool, questions_embedding, decisions_features, **kwargs):
        """
        * A NQS method *
        Select the next question to be the one with the lowest confidence of one of the decisions.
        I.e., the lowest probability is in one of the decisions (p0 or p1).
        :param question_pool: DataFrame with the questions that can be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        """
        question_pool = self.predict_all_nqs(question_pool=question_pool,
                                             questions_embedding=questions_embedding,
                                             decisions_features=decisions_features)
        lowest = question_pool[['p0', 'p1']].max(axis=1).min()
        lowest_question = question_pool[(question_pool['p0'] == lowest) | (question_pool['p1'] == lowest)].iloc[0]
        return lowest_question

    def lowest_to_bhighest(self, question_pool, questions_embedding, decisions_features, rule, **kwargs):
        """
        * A NQS method *
        Start from selecting the questions with the lowest confidence and then, after <rule> questions, select the
        questions with the highest confidence. The idea is that the first low confidence questions will help to
        teach the engine, and then, the engine will be able to predict the next questions with higher confidence.
        :param question_pool: DataFrame with the questions that can be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        :param rule: The amount of questions to select with the lowest confidence
        """
        index = len(questions_embedding)
        if index < rule:
            return self.highest_binary_entropy(question_pool=question_pool,
                                               questions_embedding=questions_embedding,
                                               decisions_features=decisions_features)
        else:
            return self.lowest_binary_entropy(question_pool=question_pool,
                                              questions_embedding=questions_embedding,
                                              decisions_features=decisions_features)

    def lowest_to_rhighest(self, question_pool, questions_embedding, decisions_features, rule, **kwargs):
        """
        * A NQS method *
        Start from selecting the questions with the lowest binary confidence (highest entropy)
        and then, after <rule> questions, select the questions with the highest regression confidence.
        The idea is that the first low confidence questions will help to teach the engine, and then,
        when the engine is improved, it will predict the continuous confidence better.
        We'll select the highest continuous confidence since the participant is more likely to give the correct answer.
        :param question_pool: DataFrame with the questions that can be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        :param rule: The amount of questions to select with the lowest confidence
        """
        index = len(questions_embedding)
        if index < rule:
            return self.highest_binary_entropy(question_pool=question_pool,
                                               questions_embedding=questions_embedding,
                                               decisions_features=decisions_features)
        else:
            return self.highest_user_confidence(question_pool=question_pool,
                                                questions_embedding=questions_embedding,
                                                decisions_features=decisions_features)

    def lowest_to_rlowest(self, question_pool, questions_embedding, decisions_features, rule, **kwargs):
        """
        * A NQS method *
        Start from selecting the questions with the lowest binary confidence (highest entropy)
        and then, after <rule> questions, select the questions with the lowest regression confidence.
        The idea is that the first low confidence questions will help to teach the engine, and then,
        when the engine is improved, it will predict the continuous confidence better.
        The lowest continuous confidence is selected since the lowest_user_confidence showed the highest acc during
        the experiments.
        :param question_pool: DataFrame with the questions that can be asked
        :param questions_embedding: List with the embedding of the previous questions
        :param decisions_features: List with the decision features of the previous questions
        :param rule: The amount of questions to select with the lowest confidence
        """
        index = len(questions_embedding)
        if index < rule:
            return self.highest_binary_entropy(question_pool=question_pool,
                                               questions_embedding=questions_embedding,
                                               decisions_features=decisions_features)
        else:
            return self.lowest_user_confidence(question_pool=question_pool,
                                               questions_embedding=questions_embedding,
                                               decisions_features=decisions_features)


class Strategy:
    def __init__(self, question_pool,
                 first_question_method, next_question_method, fq_kwargs=None, nq_kwargs=None,
                 save_path=None, run=None, asked_questions=None):
        """
        The strategy class is used to hold the information and manage the selection of NQ during the experiment.
        It holds the initial question pool and track the asked questions.
        Furthermore, it holds the methods to select the first question and the next question.
        :param question_pool: DataFrame with the questions to be asked
        :param first_question_method: Method to select the first question
        :param next_question_method: Method to select the next question
        """
        self.questions_pool = question_pool
        self.asked_questions = asked_questions if asked_questions is not None else []

        # First question selection
        self.first_question_method = first_question_method
        self.fq_kwargs = fq_kwargs

        # Next question selection
        self.next_question_method = next_question_method
        self.nq_selector = NextQuestionSelection(next_question_method=next_question_method)
        self.nq_kwargs = nq_kwargs

        if save_path is not None:
            self.save_strategy(save_path, run)

    def __dict__(self):
        return {'first_question_method': self.first_question_method,
                'next_question_method': self.next_question_method,
                'nq_kwargs': self.nq_kwargs,
                'fq_kwargs': self.fq_kwargs, }

    def is_engine_required(self):
        return self.nq_selector.engine_required

    def save_strategy(self, path, run):
        """
        Save the strategy
        @param path: The path to save the strategy
        @param run: The wandb run to save the protocol for.
        """
        rname = run.name
        if not os.path.exists(f"{path}/{rname}"):
            os.mkdir(f"{path}/{rname}")

        with open(f'{path}/{rname}/config_{rname}.json', 'w') as fp:
            json.dump(self.__dict__(), fp)

    @classmethod
    def from_other(cls, other,
                   question_pool=None,
                   first_question_method=None,
                   next_question_method=None,
                   nq_kwargs=None,
                   fq_kwargs=None):
        """
        Create a new Strategy object from another Strategy object.
        :param other: Strategy object to copy
        The other parameters are used to override the parameters of the other Strategy object
        :param question_pool: DataFrame with the questions to be asked
        :param first_question_method: Method to select the first question
        :param next_question_method: Method to select the next question
        :param nq_kwargs: Keyword arguments to be used by the method that selects the next question
        :param fq_kwargs: Keyword arguments to be used by the method that selects the first question
        """
        return cls(question_pool=question_pool if question_pool is not None else other.questions_pool,
                   first_question_method=first_question_method if first_question_method is not None else other.first_question_method,
                   next_question_method=next_question_method if next_question_method is not None else other.next_question_method,
                   nq_kwargs=nq_kwargs if nq_kwargs is not None else other.nq_kwargs,
                   fq_kwargs=fq_kwargs if fq_kwargs is not None else other.fq_kwargs)

    def set_engine(self, engine, device):
        """
        Set the engine and device to be used by the NextQuestionSelection object
        :param engine: Engine to be used to predict the next question
        :param device: Device to be used by the engine
        """
        self.nq_selector.set_engine(engine=engine, device=device)

    def choose_question(self):
        """
        Choose the next question to be asked: If no question has been asked, it selects the first question,
        otherwise it selects the next question
        """
        if len(self.asked_questions) == 0:
            question = self.choose_first_question()
        else:
            question = self.choose_next_question()
        self.asked_questions.append({'question_order': question[l.ORDER],
                                     'question': question,
                                     'decision_features': None})
        return question

    def choose_first_question(self):
        """
        Choose the first question to be asked
        """
        method = FirstQuestionSelection.methods[self.first_question_method]
        question = method(self.questions_pool, **self.fq_kwargs)
        return question

    def choose_next_question(self):
        """
        Choose the next question to be asked:
        1. Calculate the current pool of questions
        2. Calculate the embedding of the previous questions
        3. Select and return the next question
        """
        # Questions pool
        asked_orders = [q['question_order'] for q in self.asked_questions]
        current_pool = self.questions_pool[~self.questions_pool[l.ORDER].isin(asked_orders)]
        # Previous questions embedding
        questions_embedding = self.questions_pool[self.questions_pool[l.ORDER].isin(asked_orders)]
        questions_embedding = questions_embedding.sort_values(by='order', key=lambda column: column.map(
            lambda e: asked_orders.index(e)))
        questions_embedding = questions_embedding['ada_v2'].to_list()

        question = self.nq_selector.choose_next_question(current_pool=current_pool,
                                                         questions_embedding=questions_embedding,
                                                         decisions_features=[q['decision_features'] for q in
                                                                             self.asked_questions],
                                                         **self.nq_kwargs)
        return question

    def append_decision(self, question, decision_features):
        """
        After the decision is made, append the decision features to the asked question
        """
        q_order = question['order']
        for d in self.asked_questions:
            if d['question_order'] == q_order:
                d['decision_features'] = decision_features
                break
        else:
            raise ValueError(f"Question {q_order} wasn't asked")
