# Description: This file contains all the literals used in the project.

# General
SAVE_MODEL = "save_model"
WANDB_TRACK = "wandb_track"
LOG_EACH = "log_each"
LOG_LIVE_MESSAGES = "log_live_messages"

# Input - Engine
ENGINE_MODEL = "engine_model"
HP_HIDDEN_DIM = "hp_hidden_dim"
PP_HIDDEN_DIM = "pp_hidden_dim"
BIDIRECTIONAL = "bidirectional"
INTERACTION_PREDICTOR = "interaction_predictor"
CONCATENATION_PREDICTOR = "concatenation_predictor"
PP_MODEL = "pp_model"

# Input - Chat Protocol
CHAT_PROTOCOL = "chat_protocol"
COMPLETION_ARGS = "completion_args"
INSPECTOR_LOGIC = "inspector_logic"
## System messages
SYS_MSG = "system_message"
SYS_MSG_FIRST = "system_message_first"
## Overview Instructions
OVERVIEW_INSTRUCTIONS = "overview_instructions"
## Task Instructions
TASK_INSTRUCTIONS = "task_instructions"
## Questions Instructions
Q_INSTRUCTIONS = "q_instructions"
## Error lines
ERROR_LINES = "error_lines"

# Input - Question Pool / Embedded Pairs
QUESTION_POOL_FILTERS = "question_pool_filters"
FILTERS = "filters"

# Strategy
SAVE_STRATEGY = "save_strategy"
SAVE_PATH = "save_path"
ORDER = "order"
FIRST_QUESTION_METHOD = "first_question_method"
QUESTION_CHOOSING_METHOD = "question_choosing_method"
DECISION_MEDIATOR = "decision_mediator"
NQ_KWARGS = "nq_kwargs"
FQ_KWARGS = "fq_kwargs"
P0 = "p0"
P1 = "p1"
PRED_CONF = "pred_conf"


class DM:
    EMBED_EXPLANATION = "embed_explanation"
    HUMAN_FEATURES = "human_features"


class FQS:
    RANDOM = "random"
    ORIGINAL = "original"
    GIVEN_ORDER = "given_order"


class NQS:
    RANDOM = "random"
    ORIGINAL = "original"
    GIVEN_ORDER = "given_order"
    LOWEST_BINARY_ENTROPY = "lowest_binary_entropy"
    HIGHEST_BINARY_ENTROPY = "highest_binary_entropy"
    HIGHEST_USER_CONFIDENCE = "highest_user_confidence"
    LOWEST_USER_CONFIDENCE = "lowest_user_confidence"
    LOWEST_TO_BHIGHEST = "lowest_to_bhighest"
    LOWEST_TO_RHIGHEST = "lowest_to_rhighest"
    LOWEST_TO_RLOWEST = "lowest_to_rlowest"


# Evaluation
AGENT_AMOUNT = "agent_amount"
SEEDS_METHOD = "seeds_method"
SEEDS = "seeds"


class SD:
    NONE = "none"
    EVEN = "even"
    ODD = "odd"

# Data handler
# HISTORY = "history"
# NEXT_QUESTION = "next_question"
# NQ_IS_MATCH = "nq_is_match"
# NQ_REAL_CONF = "nq_real_conf"
# META_DATA = "meta_data"
# LENGTHS = "lengths"
# HISTORIES = "histories"
# NEXT_QUESTIONS = "next_questions"
# NQ_REAL_CONFS = "nq_real_confs"
# NQ_IS_MATCHES = "nq_is_matches"
# META_DATAS = "meta_datas"
# RANDOM = "random"
# MATCHERS = "matchers"
# SPLIT_BY = "split_by"
# BY_QUESTION = "by_question"
# LESS = "less"
