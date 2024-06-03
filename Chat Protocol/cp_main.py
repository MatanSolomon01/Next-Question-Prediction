from collections import defaultdict

import sys
import os
from openai import AzureOpenAI
import wandb
from tqdm import tqdm
from cp_utils import save_protocol, load_question_pool
import cp_consts as c
import cp_literals as l
from facilitator import Facilitator

sys.path.append("Evaluation Framework")
from ef_utils import log_results


def main():
    """
    Main function to run the chat protocol
    This part of the project is responsible for experimenting with the chat model, and saving the protocol for
    the use of the evaluation framework.
    """
    # Initiate wandb run
    run = wandb.init(project='gpt_agent',
                     config=c.config,
                     mode=['disabled', 'online'][c.config[l.WANDB_TRACK]])

    # Initialize the OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint="https://baropenairesource.openai.azure.com/",
    )

    # Load the question pool
    question_pool = load_question_pool(question_pool_path=c.embedded_pairs, filters=c.config[l.QUESTION_POOL_FILTERS])

    results = {}
    for i, seed in enumerate(c.config[l.SEEDS]):
        # Initialize the facilitator
        facilitator = Facilitator(client=client,
                                  completion_args=c.config['completion_args'],
                                  strings_path=c.strings_path,
                                  log_live_messages=c.config[l.LOG_LIVE_MESSAGES],
                                  seed=seed)
        facilitator.set_sys_msg(c.config['system_message'])
        facilitator.set_instructions(c.config['overview_instructions'],
                                     c.config['task_instructions'],
                                     c.config['q_instructions'])
        facilitator.set_error_lines(c.config['error_lines'])

        # Start the chat flow
        error_counts = defaultdict(int)
        facilitator.start_conversation()
        if facilitator.dead_facilitator:
            print(f"Facilitator {i + 1} died in intro, moving on...")
            continue

        for index in tqdm(range(1, 31)):
            question = question_pool[question_pool['order'] == index].iloc[0]
            facilitator.handle_question(question=question,
                                        error_counts=error_counts,
                                        inspector_logic=c.config[l.INSPECTOR_LOGIC])
            if facilitator.dead_facilitator:
                print(f"Facilitator {i + 1} died ({index}), no more tries, moving on...")
                break

        if not facilitator.dead_facilitator:
            results[i] = {'facilitator': facilitator,
                          'error_count': error_counts}

    # Log results
    log_results(results=results, added_results=['log_decisions'])
    if c.config[l.SAVE_PROTOCOL]:
        save_protocol(run)

    run.finish()


if __name__ == '__main__':
    main()
