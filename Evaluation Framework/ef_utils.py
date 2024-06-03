import pandas as pd
import wandb
from warnings import warn
import ef_literals as l
import os
import ef_consts as c

def log_results(facilitator=None, error_counts=None, results=None, added_results=None):
    if results is None:
        results = {0: {'facilitator': facilitator, 'error_count': error_counts}}

    wandb_log = {}
    local_log = {}

    # Error counts
    eval = pd.DataFrame.from_dict({k: v['error_count'] for k, v in results.items()}, orient='index')
    eval = eval.fillna(0)

    for k, v in results.items():
        # Chat messages
        facilitator = v['facilitator']
        chat_messages = pd.DataFrame(facilitator.chat_messages)

        # Decisions
        decisions = pd.DataFrame(facilitator.decisions)
        questions = pd.DataFrame(facilitator.questions)
        questions = questions.drop(
            columns=['alg', 'score', 'token_path', 'term_match', 'word_net', 'prompt', 'raw', 'n_tokens', 'ada_v2'])
        merged = questions.merge(decisions, left_on='order', right_on='order', how='left')

        acc = (merged['binary_decision'] == merged['realConf']).mean()
        eval.loc[k, 'chat_accuracy'] = acc

        mean_tries = merged['tries'].mean()
        eval.loc[k, 'mean_tries'] = mean_tries

        if added_results is not None:
            if 'engine_eval' in added_results:
                nothing = True
                if l.P1 in merged.columns:
                    merged['engine_binary'] = (merged['p1'] > 0.5).astype(int)
                    engine_acc = (merged['engine_binary'] == merged['binary_decision']).mean()
                    eval.loc[k, 'engine_accuracy'] = engine_acc
                    nothing = False
                if l.PRED_CONF in merged.columns:
                    engine_mae = abs(merged[l.PRED_CONF] - merged['normalized_conf']).mean()
                    eval.loc[k, 'engine_mae'] = engine_mae
                    nothing = False
                if nothing:
                    warn("\nMissing required evaluation data! Is the strategy used an engine? Engine eval ignored.")

            if 'log_decisions' in added_results:
                local_log[f"chat_messages_{k}"] = chat_messages
                local_log[f"decisions_{k}"] = merged

    wandb_log['eval'] = wandb.Table(dataframe=eval)
    d = {f"avg_{k}": v for k, v in eval.mean().items()}
    wandb_log.update(d)

    # Log it all
    wandb.log(wandb_log)

    if len(local_log) > 0:
        name, id = wandb.run.name, wandb.run.id
        path = f"{c.project_dir}/Chat Protocol/experiments/agents_data/{id}/media/table/"
        if not os.path.exists(path):
            os.makedirs(path)
        for k, v in local_log.items():
            v.to_pickle(f"{path}/{k}.pkl")