from pathlib import Path
import et_literals as l
import json
import sys

# Paths
project_name = "Next Question Prediction"
# PROJECT_DIR = Path.home() / "OneDrive - Technion" / "Research" / "Code" / "Human matchers representation" / PROJECT_NAME
project_dir = Path.home() / project_name
embedded_pairs = project_dir / "data" / "question_embeddings" / "embedded_pairs.pkl"
user_results = project_dir / "data" / "experiment_data" / "data.pkl"
samples = project_dir / "data" / "samples"
chat_samples = project_dir / "data" / "chat_samples"
trained_models = project_dir / "Engine Training" / "trained_models"
chat_protocol_dir = project_dir / "Chat Protocol" / "protocols"
strategy_dir = project_dir / "Evaluation Framework" / "strategies"
strings_path = project_dir / "data" / "strings.json"

sys.path.append(f"{project_dir}/Evaluation Framework")
import ef_literals as efl

# Run parameters
load_config = False
rname = "fearless-yogurt-64"
config_path = f"{trained_models}/{rname}/config_{rname}.json"

if load_config:
    with open(config_path, 'r') as f:
        config = json.load(f)
        print(f"config loaded from {rname}")
else:
    config = {}
    # General
    config[l.SAVE_MODEL] = False  # Whether to save the model after training
    config[l.WANDB_TRACK] = False  # Whether to track the run with wandb
    config[l.TEST_EACH] = 1  # After how many epochs to test the model, 0 for no testing
    config[l.FILTERS] = {'exp_id': [15]}

    # Data
    config[l.AGENTS_TYPE] = [l.HUMANS, l.CHATBOTS][1]  # The type of agents to include in the data
    # # In case of humans
    config[l.LOAD_SAMPLES] = True  # Note: if True, then save_samples will be ignored.
    config[l.SAVE_SAMPLES] = True  # In case load_samples is false, Whether to save the data to storage.
    # # In case of chatbots
    config[efl.CHAT_PROTOCOL] = "stellar-darkness-43"
    config[l.SAMPLE_SELECTOR] = l.SS.RANDOM_EXISTING  # The sample selector to use
    config[l.SAMPLE_SELECTOR_ARGS] = {
        efl.AGENT_AMOUNT: 50,
        efl.SEEDS_METHOD: efl.SD.ODD,  # Warning! If SEEDS_METHOD is NONE, then SAVE_SAMPLES will be ignored!
        l.SAMPLES_AMOUNT: 500,  # For random - amount of samples per agent, for whole_random - amount of sequences
        l.DEMO_STRATEGY: "dummy-4zm67j0v",  # For strategy - the strategy to use
        l.CREATE_MISSING: False,  # For random_existing - whether to create missing samples if there are not enough
    }  # The arguments for the sample selector
    config[efl.DECISION_MEDIATOR] = efl.DM.HUMAN_FEATURES
    config[l.AGENTS_WANDB_TRACK] = False

    # Model
    # # History processor
    config[l.HP_HIDDEN_DIM] = 128  # Hidden dimension of the history processor
    config[l.BIDIRECTIONAL] = True  # Whether to use a bidirectional LSTM in the history processor
    # # Performance predictor
    config[l.PP_MODEL] = [l.CONCATENATION_PREDICTOR,
                          l.INTERACTION_PREDICTOR,
                          l.REGRESSION_HEAD_PREDICTOR][2]  # The performance predictor model to use
    config[l.CLASSIFICATION_WEIGHT] = 0.99
    # WARNING! The REGRESSION_HEAD_PREDICTOR gives that weight to the classification loss
    config[l.PP_HIDDEN_DIM] = 1024  # Hidden dimension of the performance predictor

    # Training
    config[l.SPLIT_BY] = [l.MATCHERS, l.RANDOM, l.QUESTIONS][1]
    # Whether to split the data by questions or by matchers, or split all randomly
    config[l.BY_QUESTION] = [(l.RANDOM, 1), (l.LESS, 20)][1]
    config[l.TRAIN_PORTION] = 0.8  # Must be of type float for a portion
    attributes = [l.NQ_REAL_CONFS, l.NQ_IS_MATCHES, l.USERCONF]
    config[l.LABEL_KEYS] = {l.BINARY: attributes[1],
                            l.REGRESSION: attributes[2]}  # The keys of the label in the data
    # WARNING!  Make sure it matches the config[l.PP_MODEL] in the model section!!

    config[l.EPOCHS] = 30  # Number of epochs to train the model
    config[l.BATCH_SIZE] = 1500  # Number of samples in each batch
    config[l.LR] = 0.005  # Learning rate
