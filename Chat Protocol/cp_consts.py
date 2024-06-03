import json
from pathlib import Path
import cp_literals as l

# Paths
project_name = "Next Question Prediction"
# PROJECT_DIR = Path.home() / "OneDrive - Technion" / "Research" / "Code" / "Human matchers representation" / PROJECT_NAME
project_dir = Path.home() / project_name
embedded_pairs = project_dir / "data" / "question_embeddings" / "embedded_pairs.pkl"
protocols = project_dir / "Chat Protocol" / "protocols"
strings_path = project_dir / "data" / "strings.json"

# Configuration

config = {}

# General parameters
config[l.QUESTION_POOL_FILTERS] = {'exp_id': [15]}

# Run parameters
config[l.DESCRIPTION] = "test"

## System messages
config[l.SYS_MSG] = "basic"
config[l.SYS_MSG_FIRST] = True  # Otherwise, last
## Overview Instructions
config[l.OVERVIEW_INSTRUCTIONS] = "basic"
## Task Instructions
config[l.TASK_INSTRUCTIONS] = "short_explanation"
## Questions Instructions
config[l.Q_INSTRUCTIONS] = "short_explanation"
## Questions
# TODO: Add questions parameter
## Error lines
config[l.INSPECTOR_LOGIC] = l.INSPECTORS.SHORT_EXPLANATION  # Pay Attention: The inspector uses the error lines!
# So, make sure the appropriate error lines are defined. More over, make sure to coordinate it with the
# instructions sent to the chat.
config[l.ERROR_LINES] = {'not_digits': "short_explanation",
                         '50_conf': "basic",
                         'first': "basic",
                         'long_explanation': "basic",
                         'invalid_time': 'basic',
                         }

# Completion args
config[l.SEEDS] = [[None], [1], list(range(2, 19, 2))][2]
config[l.COMPLETION_ARGS] = {'model': "gpt-35-turbo-Bar",
                             'temperature': 0.8,
                             # 'top_p': 0,
                             # 'frequency_penalty': 0,
                             # 'presence_penalty': 0,
                             'stop': None}

# Logging
config[l.LOG_LIVE_MESSAGES] = True
config[l.SAVE_PROTOCOL] = True
config[l.WANDB_TRACK] = True

if __name__ == '__main__':
    """
    Which strings are available in the strings.json file?
    """
    # Load the strings.json file
    with open(strings_path, 'r') as f:
        strings_path = json.load(f)
    for k, v in strings_path.items():
        print(f"{k}: {', '.join(list(v.keys()))}")
