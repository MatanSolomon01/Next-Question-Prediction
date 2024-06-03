import os
import ef_consts as c
import ef_literals as l
from inputs_handler import InputsHandler
from openai import AzureOpenAI
from strategy import Strategy
from decision_mediator import DecisionMediator
import wandb
from ef_utils import log_results
from experiment_manager import ExperimentManager


def main():
    # Initiate wandb run
    run = wandb.init(project='strategy_evaluation',
                     config=c.config,
                     mode=['disabled', 'online'][c.config[l.WANDB_TRACK]])

    # Initialize the OpenAI client
    client = AzureOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        api_version="2023-12-01-preview",
        azure_endpoint="https://baropenairesource.openai.azure.com/")

    # Load inputs
    ih = InputsHandler(engines_path=c.engine_models, engine_name=c.config[l.ENGINE_MODEL],
                       protocols_path=c.chat_protocol_dir, protocol_name=c.config[l.CHAT_PROTOCOL],
                       embedded_pairs_path=c.embedded_pairs)
    facilitators = ih.load_protocol(agent_amount=c.config[l.AGENT_AMOUNT], client=client, strings_path=c.strings_path,
                                    wandb_track=c.config[l.WANDB_TRACK], seeds_method=c.config[l.SEEDS])
    question_pool = ih.load_question_pool(filters=c.config[l.QUESTION_POOL_FILTERS])

    # Strategy components
    save_path = c.strategy_dir if c.config[l.SAVE_STRATEGY] else None
    dm = DecisionMediator(client=client, decision_mediator_method=c.config[l.DECISION_MEDIATOR])
    strategy = Strategy(question_pool=question_pool,
                        first_question_method=c.config[l.FIRST_QUESTION_METHOD],
                        next_question_method=c.config[l.QUESTION_CHOOSING_METHOD],
                        fq_kwargs=c.config[l.FQ_KWARGS],
                        nq_kwargs=c.config[l.NQ_KWARGS],
                        save_path=save_path,
                        run=run)

    # Experiment manager
    manager = ExperimentManager(question_pool=question_pool,
                                inputs_handler=ih,
                                decision_mediator=dm,
                                demo_strategy=strategy)
    added_results = ['engine_eval'] if strategy.is_engine_required() else []
    results = manager.run_experiments(facilitators=facilitators, tries=3, log_each=c.config[l.LOG_EACH],
                                      log_results_args={'added_results': added_results})
    log_results(results=results, added_results=['engine_eval', 'log_decisions'])
    run.finish()


if __name__ == '__main__':
    main()
