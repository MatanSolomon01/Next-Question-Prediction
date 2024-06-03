from pathlib import Path
import ef_literals as l
import json

# Paths
project_name = "Next Question Prediction"
project_dir = Path.home() / project_name
embedded_pairs = project_dir / "data" / "question_embeddings" / "embedded_pairs.pkl"
engine_models = project_dir / "Engine Training" / "trained_models"
chat_protocol_dir = project_dir / "Chat Protocol" / "protocols"
strings_path = project_dir / "data" / "strings.json"
strategy_dir = project_dir / "Evaluation Framework" / "strategies"

# TODO - add the option to load config
config = {}
# General
config[l.WANDB_TRACK] = True  # Whether to track the run with wandb
config[l.LOG_EACH] = 1  # After how many experiments to log the results
config[l.AGENT_AMOUNT] = 100  # The amount of agents to load
config[l.SEEDS] = l.SD.EVEN  # The seeds to use for the agents

# Inputs
config[l.ENGINE_MODEL] = "lively-firefly-108"  # See down below
config[l.CHAT_PROTOCOL] = "stellar-darkness-43"
config[l.QUESTION_POOL_FILTERS] = {'exp_id': [15]}
config[l.DECISION_MEDIATOR] = l.DM.HUMAN_FEATURES
# Warning! The decision mediator translates the decision to features that are then fed to the engine.
## Therefore, the decision mediator MUST match the engine model!

# Strategy
config[l.SAVE_STRATEGY] = False  # Whether to save the strategy
config[l.FIRST_QUESTION_METHOD] = l.FQS.RANDOM  # The method to choose the first question
config[l.QUESTION_CHOOSING_METHOD] = l.NQS.LOWEST_TO_RLOWEST  # The method to choose the next question
config[l.FQ_KWARGS] = {}
config[l.NQ_KWARGS] = {'rule': 7,  # l.NQS.LOWEST_TO_(B/R)HIGHEST, the point in which to switch from lowest to highest
                       }

# Engine Models
# lively-firefly-108 - Regression head predictor, trained on chat agents
# earthy-glade-93 - Regression head predictor
# easy-glitter-81 - Interactions Predictor

# Chat Protocols
# stellar-darkness-43 - For "<conf>, <time>" format
# divine-monkey-44 - Same, but with a positive temperature
