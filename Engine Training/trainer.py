import numpy as np
import pandas as pd
import torch

import et_consts
import wandb

from data_handler import HumanDataHandler
import et_literals as l


class Trainer:
    """
    This class is responsible for training the model.
    """

    def __init__(self, model, label_keys, optimizer, criterion, device, epochs, batch_size, test_each):
        self.model = model  # The model to train
        self.label_keys = label_keys  # The key of the label in the data
        self.optimizer = optimizer  # The optimizer to use for training
        self.criterion = criterion  # The loss function to use for training
        self.device = device  # The device to train on

        self.epochs = epochs  # Number of epochs to train the model
        self.batch_size = batch_size  # Number of samples in each batch
        self.test_each = test_each  # After how many epochs to test the model, 0 for no testing

        self.examples_seen = 0  # Number of examples seen so far during training

    def train(self, train_data: list, test_data: list) -> None:
        """
        Main functionality to train the model
        :param train_data: list of samples to train on
        :param test_data: list of samples to test on
        :return: None
        """
        for epoch_index in range(self.epochs):

            # Train
            self.train_epoch(train_data)

            if self.test_each and epoch_index % self.test_each == 0:
                print(f'Epoch {epoch_index}', end='\t')
                self.test(test_data=test_data, log_predictions=False, name="test")
                self.test(test_data=train_data, log_predictions=False, name="train")
                print()

        self.test(test_data, log_predictions=True, name="test")
        self.test(train_data, log_predictions=True, name="train")

    def train_epoch(self, train_data: list) -> None:
        """
        Train the model for one epoch
        :param train_data: list of samples to train on
        :return: None
        """
        for batch_index in range(0, len(train_data), self.batch_size):
            batch = train_data[batch_index:batch_index + self.batch_size]
            self.train_batch(batch)

    def train_batch(self, batch: list) -> None:
        """
        Train the model for one batch
        :param batch: list of samples to train on
        :return: None
        """
        out = HumanDataHandler.process_batch(batch, device=self.device)
        true_values = {k: out[v] for k, v in self.label_keys.items()}

        self.model.train()
        self.model.zero_grad()
        predicted_values = self.model(out[l.HISTORIES], out[l.NEXT_QUESTIONS])
        loss = self.criterion(predicted_values, true_values)
        loss.backward()
        self.optimizer.step()

        self.examples_seen += len(batch)
        wandb.log({"loss": loss.item()}, step=self.examples_seen)

        # print(round(loss.item(), 3), end=' ')

    def test(self, test_data: list, log_predictions: bool = False, name: str = "test") -> None:
        """
        Test the model
        :param test_data: list of samples to test on
        :return: None
        """
        self.model.eval()
        with torch.no_grad():
            out = HumanDataHandler.process_batch(test_data, device=self.device)
            true_values = {k: out[v] for k, v in self.label_keys.items()}

            # Feed the model
            predicted_values = self.model(out[l.HISTORIES], out[l.NEXT_QUESTIONS])
            # Calculate loss and accuracy
            loss = self.criterion(predicted_values, true_values)
            print(f'{name} loss: {round(loss.item(), 3)}', end='\t')

            wandb_log = {f"{name}_loss": loss.item()}
            if l.BINARY in predicted_values:
                y_hat = predicted_values[l.BINARY]
                y_hat_binary = torch.argmax(y_hat, dim=1)
                binary_label = self.label_keys[l.BINARY]
                acc = torch.sum(y_hat_binary == out[binary_label]).item() / len(out[binary_label])
                wandb_log[f'{name}_acc'] = acc
                print(f'{name} accuracy: {round(acc, 3)}', end='\t')

            if l.REGRESSION in predicted_values:
                y_hat = predicted_values[l.REGRESSION]
                regression_label = self.label_keys[l.REGRESSION]
                MAE = torch.mean(torch.abs(y_hat - out[regression_label]))
                wandb_log[f'{name}_MAE'] = MAE.item()
                print(f'{name} MAE: {round(MAE.item(), 3)}', end='\t')

            wandb.log(wandb_log)

            # Log the test histories
            if log_predictions:
                meta_datas = pd.DataFrame(out[l.META_DATAS])[['user_id', 'order']]
                lengths = pd.DataFrame(out[l.LENGTHS], columns=['length'])
                others = {}
                for prediction_type, attribute in self.label_keys.items():
                    predicted = predicted_values[prediction_type].cpu()
                    columns = predicted.shape[1] if len(predicted.shape) > 1 else 1
                    column_names = [f"{prediction_type}_{i}" for i in range(columns)]
                    others[f"{prediction_type}_pred"] = pd.DataFrame(predicted, columns=column_names)
                    others[f"{prediction_type}_true"] = pd.DataFrame(out[attribute].cpu(), columns=[attribute])

                df = pd.concat([meta_datas, lengths] + list(others.values()), axis=1)
                wandb.log({f"{name}_histories": wandb.Table(dataframe=df)})
