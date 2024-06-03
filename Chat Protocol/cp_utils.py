import json
import pandas as pd
import os
import wandb
import cp_consts as c
import pickle as pkl


def is_float(element: any) -> bool:
    """
    Check if an element is a float
    """
    # If you expect None to be passed:
    if element is None:
        return False
    try:
        float(element)
        return True
    except ValueError:
        return False


def load_question_pool(question_pool_path, filters=None):
    """
    Load the question pool from the given path and apply the given filters
    """
    with open(question_pool_path, 'rb') as f:
        question_pool = pkl.load(f)

    if filters is not None:
        for k, v in filters.items():
            question_pool = question_pool[question_pool[k].isin(v if type(v) == list else [v])]
    return question_pool


def log_results(facilitator, error_counts):
    """
    Log the results of the chat protocol:
    1. The error counts of the conversation
    2. The decisions made by the chat-agent
    3. The chat messages
    @param facilitator: The facilitator object
    @param error_counts: dictionary of error counts
    """
    wandb.log(error_counts)

    decisions = pd.DataFrame(facilitator.decisions)
    questions = pd.DataFrame(facilitator.questions)
    questions = questions.drop(
        columns=['alg', 'score', 'token_path', 'term_match', 'word_net', 'prompt', 'raw', 'n_tokens', 'ada_v2'])
    results = questions.merge(decisions, left_on='order', right_on='order', how='left')

    acc = (results['binary_decision'] == results['realConf']).mean()
    wandb.log({f"results": wandb.Table(dataframe=results)})
    wandb.log({f"mean_tries": decisions['tries'].mean()})
    wandb.log({f"accuracy": acc})

    chat_messages = pd.DataFrame(facilitator.chat_messages)
    wandb.log({f"chat_messages": wandb.Table(dataframe=chat_messages)})


def save_protocol(run):
    """
    Save the protocol of the chat
    @param run: The wandb run to save the protocol for.
    """
    rname = run.name
    path = f"{c.protocols}/{rname}"
    if not os.path.exists(path):
        os.mkdir(path)

    with open(f'{path}/config_{rname}.json', 'w') as fp:
        json.dump(c.config, fp)
