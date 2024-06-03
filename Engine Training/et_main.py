import pandas as pd
import torch
import wandb
from torch import nn, optim
from wandb.sdk.lib.disabled import RunDisabled
import os
from data_handler import HumanDataHandler
from chatbot_data_handler import ChatbotDataHandler
from models.compund import NextQuestionPrediction
from models.history_processor import HistoryProcessor
from models.performance_predictor import predictors
from trainer import Trainer
import et_literals as l
import et_consts as c
from et_consts import config as config
import json
import sys

sys.path.append("Evaluation Framework")
import ef_literals as efl


def initiate_data_handler(device):
    if config[l.AGENTS_TYPE] == l.HUMANS:
        dh = HumanDataHandler(pairs_path=c.embedded_pairs,
                              user_results_path=c.user_results,
                              samples_path=c.samples,
                              filters=config[l.FILTERS])
        dh.merged_to_samples(torch_samples=True,
                             load_samples=config[l.LOAD_SAMPLES],
                             save_samples=config[l.SAVE_SAMPLES],
                             device=device)
    elif config[l.AGENTS_TYPE] == l.CHATBOTS:
        dh = ChatbotDataHandler(pairs_path=c.embedded_pairs,
                                chat_samples_path=c.chat_samples,
                                filters=config[l.FILTERS],
                                protocols_path=c.chat_protocol_dir,
                                chat_protocol=config[efl.CHAT_PROTOCOL])
        creation_dict = {'strings_path': c.strings_path,
                         'wandb_track': config[l.AGENTS_WANDB_TRACK],
                         'decision_mediator_method': config[efl.DECISION_MEDIATOR],
                         'torch_samples': True,
                         'device': device}
        dh.prepare_data(sample_selector_method=config[l.SAMPLE_SELECTOR],
                        sample_selector_args=config[l.SAMPLE_SELECTOR_ARGS].copy(),
                        creation_dict=creation_dict)
    else:
        raise ValueError(f"Agents type {config[l.AGENTS_TYPE]} is not supported")

    return dh


def main():
    # Initiate wandb run
    run = wandb.init(project='NQP-1',
                     config=config,
                     mode=['disabled', 'online'][config[l.WANDB_TRACK]])

    # Load data
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    dh = initiate_data_handler(device)

    train_data, test_data = dh.train_test_split(
        split_by=config[l.SPLIT_BY],
        train_portion=config[l.TRAIN_PORTION],
        by_question=config[l.BY_QUESTION],
        get_data=True,
    )

    # Build a model
    history_processor = HistoryProcessor(
        input_dim=dh.ada_v2_dim + dh.decision_features_dim,
        hidden_dim=config[l.HP_HIDDEN_DIM],
        bidirectional=config[l.BIDIRECTIONAL],
    )
    kwargs = {}
    if l.CLASSIFICATION_WEIGHT in config:
        kwargs[l.CLASSIFICATION_WEIGHT] = config[l.CLASSIFICATION_WEIGHT]
    performance_predictor = predictors[config[l.PP_MODEL]](
        user_profile_dim=(config[l.BIDIRECTIONAL] + 1) * config[l.HP_HIDDEN_DIM],
        nq_dim=dh.ada_v2_dim,
        hidden_dim=config[l.PP_HIDDEN_DIM], **kwargs)

    model = NextQuestionPrediction(history_processor, performance_predictor)

    # For training
    loss = performance_predictor.loss
    optimizer = optim.Adam(model.parameters(), lr=config[l.LR])
    model = model.to(device=device)
    wandb.watch(model, loss, log='all', log_freq=50)

    # Train the model
    trainer = Trainer(model=model, label_keys=config[l.LABEL_KEYS], criterion=loss, optimizer=optimizer,
                      device=device,
                      epochs=config[l.EPOCHS], batch_size=config[l.BATCH_SIZE], test_each=config[l.TEST_EACH])
    trainer.train(train_data, test_data)

    if config[l.SAVE_MODEL]:
        rname = run.name
        path = f"{c.trained_models}/{rname}"
        if not os.path.exists(path):
            os.mkdir(path)

        torch.save(model.state_dict(), path + f'/model_{rname}.pt')
        user_ids = [(x[l.META_DATA]['user_id'], x[l.META_DATA]['order'], "train") for x in train_data] + [
            (x[l.META_DATA]['user_id'], x[l.META_DATA]['order'], "test") for x in test_data]
        user_ids = pd.DataFrame(user_ids, columns=['user_id', 'order', 'batch'])

        user_ids.to_csv(f"{path}/train_test_{rname}.csv", float_format='%.0f', index=False)

        if not isinstance(run, RunDisabled):
            run.save(f"{path}/'model_{rname}.pt", policy='now')

        with open(f'{path}/config_{rname}.json', 'w') as fp:
            json.dump(config, fp)

    run.finish()


if __name__ == '__main__':
    main()
