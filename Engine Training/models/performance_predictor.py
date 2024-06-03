from typing import Tuple, Any, Dict

import torch
from torch import nn
import et_literals as l


class ConcatenationPredictor(nn.Module):
    """
    This class is responsible for predicting the next question values.
    """

    def __init__(self, user_profile_dim: int, nq_dim: int, hidden_dim: int, **kwargs) -> None:
        """
        Initiate the PP model
        :param user_profile_dim: user profile dimension
        :param nq_dim: next question embedding dimension
        :param hidden_dim: hidden dimension
        """
        super(ConcatenationPredictor, self).__init__()
        self.linear1 = nn.Linear(user_profile_dim + nq_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=1)

        BCE = nn.BCELoss(reduction='mean')
        self.loss = lambda predicted, true_values: BCE(predicted[l.BINARY][:, 1], true_values[l.BINARY])

    def forward(self, user_profile: torch.Tensor, nq: torch.Tensor) -> dict[Any, Any]:
        """
        Forward pass to predict next question values
        :param user_profile: user profile that modules the user's thinking and matching process
        :param nq: embedding of the next (yet unseen) question
        :return: predicted next question values
        """
        x = torch.cat((user_profile, nq), dim=1)
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return {l.BINARY: x}


class InteractionPredictor(nn.Module):
    """
    This class is responsible for predicting the next question values.
    """

    def __init__(self, user_profile_dim: int, nq_dim: int, hidden_dim: int, **kwargs) -> None:
        """
        Initiate the PP model
        :param user_profile_dim: user profile dimension
        :param nq_dim: next question embedding dimension
        :param hidden_dim: hidden dimension
        """
        super(InteractionPredictor, self).__init__()
        self.linear1 = nn.Linear(nq_dim, user_profile_dim)
        self.linear2 = nn.Linear(user_profile_dim, hidden_dim)

        self.activation1 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, 2)
        self.softmax = nn.Softmax(dim=1)

        BCE = nn.BCELoss(reduction='mean')
        self.loss = lambda predicted, true_values: BCE(predicted[l.BINARY][:, 1], true_values[l.BINARY])

    def forward(self, user_profile: torch.Tensor, nq: torch.Tensor) -> dict[Any, Any]:
        """
        Forward pass to predict next question values
        :param user_profile: user profile that modules the user's thinking and matching process
        :param nq: embedding of the next (yet unseen) question
        :return: predicted next question values
        """
        nq = self.linear1(nq)
        x = torch.mul(user_profile, nq)
        x = self.linear2(x)
        x = self.activation1(x)
        x = self.linear3(x)
        x = self.softmax(x)
        return {l.BINARY: x}


class RegressionHeadPredictor(nn.Module):
    """
    This class is responsible for predicting the next question values.
    """

    def __init__(self, user_profile_dim: int, nq_dim: int, hidden_dim: int, classification_weight=0.5,
                 **kwargs) -> None:
        """
        Initiate the PP model
        :param user_profile_dim: user profile dimension
        :param nq_dim: next question embedding dimension
        :param hidden_dim: hidden dimension
        """
        super(RegressionHeadPredictor, self).__init__()
        self.linear1 = nn.Linear(nq_dim, user_profile_dim)
        self.linear2 = nn.Linear(user_profile_dim, hidden_dim)

        self.activation1 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, out_features=3)
        self.softmax = nn.Softmax(dim=1)
        self.activation2 = nn.ReLU()

        self.loss = RegressionHeadPredictor.craft_loss(classification_weight=classification_weight)

    def forward(self, user_profile: torch.Tensor, nq: torch.Tensor) -> dict[str, Any]:
        """
        Forward pass to predict next question values
        :param user_profile: user profile that modules the user's thinking and matching process
        :param nq: embedding of the next (yet unseen) question
        :return: predicted next question values
        """
        nq = self.linear1(nq)
        x = torch.mul(user_profile, nq)
        x = self.linear2(x)
        x = self.activation1(x)
        x = self.linear3(x)
        binary = x[:, :2]
        binary = self.softmax(binary)

        regression = x[:, 2]
        regression = self.activation2(regression)
        return {l.BINARY: binary,
                l.REGRESSION: regression}

    @staticmethod
    def craft_loss(classification_weight):
        def loss(predicted_values, true_values):
            """
            Calculate the loss of the model
            :param predicted_values: predicted values
            :param true_values: true values
            """
            predicted_probs, predicted_conf = predicted_values[l.BINARY], predicted_values[l.REGRESSION]
            binary_decision, conf = true_values[l.BINARY], true_values[l.REGRESSION]

            classification_loss = nn.BCELoss(reduction='mean')(predicted_probs[:, 1], binary_decision)
            regression_loss = nn.MSELoss(reduction='mean')(predicted_conf, conf)
            return classification_weight * classification_loss + (1 - classification_weight) * regression_loss

        return loss


predictors = {l.CONCATENATION_PREDICTOR: ConcatenationPredictor,
              l.INTERACTION_PREDICTOR: InteractionPredictor,
              l.REGRESSION_HEAD_PREDICTOR: RegressionHeadPredictor}
