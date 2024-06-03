import torch
from torch import nn
from models.history_processor import HistoryProcessor
from models.performance_predictor import ConcatenationPredictor, InteractionPredictor


class NextQuestionPrediction(nn.Module):
    """
    This class combines the history processor and the performance predictor to a single model.
    """

    def __init__(self, history_processor: HistoryProcessor, performance_predictor) -> None:
        """
        Initiate a combined model that predicts the next question values.
        :param history_processor: a history processor model
        :param performance_predictor: a performance predictor model
        """
        super(NextQuestionPrediction, self).__init__()
        self.his_proc = history_processor
        self.perf_pred = performance_predictor

    def forward(self, history, nq, return_user_profile=False):
        """
        Forward pass of the model
        Send the history to the history processor to get a user profile.
        Then, use the user profile with the next question embedding to get the next question values via the performance predictor.
        :param history: user history - previous questions question_embeddings and decision features
        :param nq: next question embedding
        :param return_user_profile: Whether to return the user profile
        :return: next question values, user profile (optional)
        """
        user_profile = self.his_proc(history)
        prediction = self.perf_pred(user_profile, nq)
        if return_user_profile:
            return prediction, user_profile

        return prediction
