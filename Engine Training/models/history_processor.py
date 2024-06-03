import torch
from torch import nn


class HistoryProcessor(nn.Module):
    """
    This class is responsible for processing the history of the user.
    The History is a sequence of previous questions user's answers.
    The HistoryProcessor uses an LSTM to process the history and output a user profile.
    """

    def __init__(self, input_dim: int, hidden_dim: int, bidirectional=True) -> None:
        """
        Initialize the HistoryProcessor
        :param input_dim: Input dimension of the LSTM
        :param hidden_dim: Hidden dimension of the LSTM
        :param bidirectional: Whether to use a bidirectional LSTM
        """
        super(HistoryProcessor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, bidirectional=bidirectional)

    def forward(self, history) -> torch.Tensor:
        """
        Forward pass of the HistoryProcessor, process the history and output a user profile
        :param history: The history to process
        :return: The user profile
        """
        output, (h, c) = self.lstm(history)
        h = h.permute(1, 0, 2)
        h = h.reshape(h.shape[0], -1)
        return h
