from collections import defaultdict
from strategy import Strategy
from ef_utils import log_results


class ExperimentManager:
    """
    Manages the experiments process for multiple facilitators
    """

    def __init__(self, question_pool, demo_strategy, decision_mediator, inputs_handler):
        self.question_pool = question_pool  # DataFrame, the question pool to present to each chat agent
        self.demo_strategy = demo_strategy  # A strategy object to copy the strategy from
        self.dm = decision_mediator  # the decision mediator to use (for converting decisions to features)
        self.ih = inputs_handler  # the inputs handler to use (for loading the engine)
        self.engine_params = None  # Engine, device

    def run_experiments(self, facilitators, tries=1, log_each=0, log_results_args=None):
        """
        Run experiments for multiple facilitators (chat agents)
        @param facilitators: list of Facilitator objects
        @param tries: number of tries to run for each facilitator
        :return: dictionary of results
        """
        results = {}
        for i, facilitator in enumerate(facilitators):
            for try_index in range(tries):
                print(f"{i + 1}/{len(facilitators)} | ", end=' ')
                error_count = self.run_experiment(facilitator=facilitator)
                if facilitator.dead_facilitator:
                    if try_index < tries - 1:
                        print(f"Facilitator {i + 1} died, trying again ({try_index + 1}/{tries})")
                        facilitator.reset_facilitator()
                    else:
                        print(f"Facilitator {i + 1} died, no more tries, moving on...")
                else:
                    results[i] = {'facilitator': facilitator,
                                  'error_count': error_count}
                    break

            if log_each != 0 and (i + 1) % log_each == 0 and not facilitator.dead_facilitator:
                log_results_args = {} if log_results_args is None else log_results_args
                log_results(results=results, **log_results_args)
            print()

        return results

    def run_experiment(self, facilitator, strategy=None, remaining_questions=None, introduction=True):
        """
        Run an experiment for a single facilitator (chat agent)
        :param facilitator: Facilitator object
        :param strategy: Strategy object, if None, will use the demo strategy
        :param remaining_questions: DataFrame with the remaining questions to ask. If None, will use the question pool
        :param introduction: bool, whether to start the conversation with an introduction phase
        :return: dictionary of error counts
        """
        error_counts = defaultdict(int)
        if remaining_questions is None:
            remaining_questions = self.question_pool.copy()
        if strategy is None:
            strategy = Strategy.from_other(self.demo_strategy, question_pool=remaining_questions)

        if introduction:
            facilitator.start_conversation()
            if facilitator.dead_facilitator:
                return None
        else:
            facilitator.assert_ready()
        print(f"Asked questions: ", end='')
        while len(remaining_questions) > 0:
            print(len(strategy.asked_questions), end=' ')
            # Choose and ask question
            question = strategy.choose_question()
            decision = facilitator.handle_question(question=question, return_decision=True, error_counts=error_counts)
            if facilitator.dead_facilitator:
                return None
            # Record decision
            decision_features = self.dm.convert_decision(decision=decision, question=question)
            strategy.append_decision(question=question, decision_features=decision_features)
            # Load engine if needed
            if strategy.nq_selector.engine_required and strategy.nq_selector.engine is None:
                if self.engine_params is None:
                    self.ih.decision_features_dim = len(decision_features)
                    self.engine_params = self.ih.load_engine()
                # If engine is required and not loaded yet
                strategy.set_engine(*self.engine_params)  # Engine, device

            remaining_questions = remaining_questions[
                ~remaining_questions['order'].isin([q['question_order'] for q in strategy.asked_questions])]

        print(f"Done :) ", end='')
        return error_counts
