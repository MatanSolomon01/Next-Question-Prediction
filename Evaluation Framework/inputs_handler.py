import pickle as pkl
import json
import torch

import ef_consts as c
import sys
import ef_literals as l

sys.path.append(f"{c.project_dir}/Chat Protocol")
from facilitator import Facilitator

sys.path.append(f"{c.project_dir}/Engine Training")
from models.compund import NextQuestionPrediction
from models.history_processor import HistoryProcessor
from models.performance_predictor import predictors


class InputsHandler:
    """
    This class is responsible for handling the inputs of the evaluation framework.
    It is responsible for loading the chat protocol, the question pool, and the engine model.
    """

    def __init__(self,
                 engines_path: str,
                 engine_name: str,
                 protocols_path: str,
                 protocol_name: str,
                 embedded_pairs_path: str):
        self.engines_path = engines_path  # Path to the engine models directory
        self.engine_name: str = engine_name  # The engine model to use
        self.protocol_path: str = protocols_path  # Path to the chat protocol directory
        self.protocol_name: str = protocol_name  # The chat protocol to use
        self.embedded_pairs_path: str = embedded_pairs_path  # Path to the pairs (questions) file

        # Protocol handling
        self.chat_protocol = None  # The chat protocol

        # Engine handling
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # The device to use
        self.model_config = None  # The model config

        self.decision_features_dim: int = -1  # The dimension of the decision features
        self.ada_v2_dim = 1536  # The dimension of the question features

    def load_protocol(self, agent_amount=0,
                      seeds=None, seeds_method=l.SD.NONE,
                      client=None, strings_path=None, wandb_track=False):
        """
        This method is responsible for loading the chat protocol from a file, and optionally creating facilitators.
        The file path is stored in the instance variable `self.protocol_path`.
        :param agent_amount: The amount of facilitators to create, if any.
        :param seeds: The seeds to use when creating facilitators.
        :param seeds_method: If seeds are not provided, the method to use when generating them.
        :param client: The OpenAI client to use when creating facilitators.
        :param strings_path: The path to the strings file to use when creating facilitators.
        :param wandb_track: Whether to track the live messages of the first facilitator using wandb.
        :return: A list of facilitators, if any.
        """
        path = f"{self.protocol_path}/{self.protocol_name}/config_{self.protocol_name}.json"
        with open(path, 'rb') as f:
            protocol = json.load(f)
        self.chat_protocol = protocol
        facilitators = None

        if agent_amount > 0 or (seeds is not None and len(seeds) > 0):
            if seeds is None:
                seeds = InputsHandler.generate_seeds(seeds_method=seeds_method, agent_amount=agent_amount)
            assert client is not None, "Client must be provided when returning facilitators"
            assert strings_path is not None, "Strings path must be provided when returning facilitators"
            facilitators = []
            for i, seed in enumerate(seeds):
                facilitator = Facilitator(client=client,
                                          completion_args=self.chat_protocol[l.COMPLETION_ARGS],
                                          strings_path=strings_path,
                                          inspector_logic=self.chat_protocol[l.INSPECTOR_LOGIC],
                                          log_live_messages=self.chat_protocol[
                                              l.LOG_LIVE_MESSAGES] if i == 0 and wandb_track else False,
                                          seed=seed)
                facilitator.set_sys_msg(self.chat_protocol[l.SYS_MSG])
                facilitator.set_instructions(self.chat_protocol[l.OVERVIEW_INSTRUCTIONS],
                                             self.chat_protocol[l.TASK_INSTRUCTIONS],
                                             self.chat_protocol[l.Q_INSTRUCTIONS])
                facilitator.set_error_lines(self.chat_protocol[l.ERROR_LINES])
                facilitators.append(facilitator)

        return facilitators

    @staticmethod
    def generate_seeds(seeds_method, agent_amount):
        """
        This method is responsible for generating the seeds for the agents.
        :param seeds_method: The seeds methods to use
        :param agent_amount: The amount of agents
        :return: The seeds
        """
        assert seeds_method in [l.SD.NONE, l.SD.EVEN, l.SD.ODD], "Invalid seeds method"
        if seeds_method == l.SD.NONE:
            return [None] * agent_amount
        elif seeds_method == l.SD.EVEN:
            # Agent amount even numbers
            return list(range(0, 2 * agent_amount, 2))
        elif seeds_method == l.SD.ODD:
            # Agent amount odd numbers
            return list(range(1, 2 * agent_amount, 2))

    def load_question_pool(self, filters=None):
        """
        This method is responsible for loading the question pool from a file.
        The file path is stored in the instance variable `self.pairs_path`. The file is expected to be in pickle format.
        :return: The question pool
        """
        with open(self.embedded_pairs_path, 'rb') as f:
            question_pool = pkl.load(f)

        if filters is not None:
            for k, v in filters.items():
                question_pool = question_pool[question_pool[k].isin(v if type(v) == list else [v])]

        return question_pool

    def load_engine(self):
        """
        This method is responsible for loading the engine model.
        The model is expected to be stored in two files: a config file and a model file.
        The paths to these files are stored in the instance variable `self.engines_path`.
        :return: The loaded model and the device to use
        """
        config_path = f"{self.engines_path}/{self.engine_name}/config_{self.engine_name}.json"
        model_path = f"{self.engines_path}/{self.engine_name}/model_{self.engine_name}.pt"

        # Load config
        with open(config_path, 'rb') as f:
            model_config = json.load(f)
        self.model_config = model_config

        # Build model
        history_processor = HistoryProcessor(input_dim=self.ada_v2_dim + self.decision_features_dim,
                                             hidden_dim=self.model_config[l.HP_HIDDEN_DIM],
                                             bidirectional=self.model_config[l.BIDIRECTIONAL])
        performance_predictor = predictors[self.model_config[l.PP_MODEL]](
            user_profile_dim=(self.model_config[l.BIDIRECTIONAL] + 1) * self.model_config[l.HP_HIDDEN_DIM],
            nq_dim=self.ada_v2_dim,
            hidden_dim=self.model_config[l.PP_HIDDEN_DIM])
        model = NextQuestionPrediction(history_processor, performance_predictor)
        model = model.to(device=self.device)

        state_dict = torch.load(model_path)
        load_state_dict = model.load_state_dict(state_dict)
        print(load_state_dict, end=' ')
        return model, self.device
