import json
import cp_literals as l
from Inspector import inspectors
import wandb


class Facilitator:
    """
    The Facilitator class is responsible for managing the conversation flow. It does not decide the order of questions.
    It receives a question and presents it according to the protocol.
    It is not aware of the question pool, but only of the questions that were already asked.
    """

    def __init__(self, client, completion_args, strings_path, inspector_logic=None, **kwargs):
        """
        Initializes the Facilitator class with the given parameters.

        :param client: The azure client to be used.
        :param completion_args: The arguments to be used for completion for the azure model.
        :param strings_path: The path to the strings file.
        :param inspector_logic: The logic to be used by the inspector - the inspector is responsible for validating the
        user's input according to some specific logic.
        :param kwargs: Additional keyword arguments.
        """
        # Message generators
        self.sys_msg = lambda x: {"role": "system", "content": x}
        self.usr_msg = lambda x: {"role": "user", "content": x}
        self.ast_msg = lambda x: {"role": "assistant", "content": x}

        # General
        ## Chat
        self.client = client  # Azure client
        self.completion_args = completion_args.copy()  # Completion arguments
        self.seed = kwargs.get(l.SEED, None)
        if self.seed is not None:
            self.completion_args[l.SEED] = self.seed

        self.decisions = []
        self.questions = []
        self.chat_messages = []
        ## Strings
        self.strings_path = strings_path
        with open(self.strings_path, 'r') as f:
            self.strings = json.load(f)

        # System messages
        self.system_message = kwargs.get(l.SYS_MSG, None)
        if self.system_message is not None:
            self.set_sys_msg(self.system_message)
        self.sys_msg_first = kwargs.get(l.SYS_MSG_FIRST, True)
        self.log_live_messages = kwargs.get(l.LOG_LIVE_MESSAGES, False)
        # If sys_msg_first is False, we cannot log live messages, therefore it cant be True
        if not self.sys_msg_first:
            assert not self.log_live_messages, "Cannot log live messages if sys_msg_first is False"

        # Instructions
        self.instructions = {l.OVERVIEW_INSTRUCTIONS: kwargs.get(l.OVERVIEW_INSTRUCTIONS, None),
                             l.TASK_INSTRUCTIONS: kwargs.get(l.TASK_INSTRUCTIONS, None),
                             l.Q_INSTRUCTIONS: kwargs.get(l.Q_INSTRUCTIONS, None)}
        self.pre_task_instructions = None
        if (self.instructions[l.OVERVIEW_INSTRUCTIONS] is not None and
                self.instructions[l.TASK_INSTRUCTIONS] is not None and
                self.instructions[l.Q_INSTRUCTIONS] is not None):
            self.set_instructions(self.instructions[l.OVERVIEW_INSTRUCTIONS],
                                  self.instructions[l.TASK_INSTRUCTIONS],
                                  self.instructions[l.Q_INSTRUCTIONS])

        # Evaluation & Error lines
        self.inspector_logic = inspector_logic
        self.error_lines = kwargs.get(l.ERROR_LINES, None)
        if self.error_lines is not None:
            self.set_error_lines(self.error_lines)

        # Dead facilitator
        # If a completion fails, the facilitator is considered dead
        self.dead_facilitator = False

    def reset_facilitator(self, checkpoint=None):
        if not checkpoint:
            self.decisions = []
            self.questions = []
            self.chat_messages = []
            self.dead_facilitator = False

        else:
            self.chat_messages = checkpoint['chat_messages']
            self.decisions = checkpoint['decisions']
            self.questions = checkpoint['questions']

    def set_sys_msg(self, message):
        """
        Sets the system message.
        The function gets the name of the system message, extracts the content from the strings file, and sets it.
        :param message: The message to be set.
        """
        message = self.strings[l.SYS_MSG][message]['content']
        self.system_message = self.sys_msg(message)

    def set_instructions(self, overview_instructions, task_instructions, q_instructions):
        """
        Sets the instructions for the conversation.
        The arguments are the names of the instructions in the strings file.
        The function extracts the content from the strings file and sets the instructions.
        :param overview_instructions: The overview instructions.
        :param task_instructions: The task instructions.
        :param q_instructions: The question instructions.
        """
        self.instructions[l.OVERVIEW_INSTRUCTIONS] = self.strings[l.OVERVIEW_INSTRUCTIONS][overview_instructions][
            'content']
        self.instructions[l.TASK_INSTRUCTIONS] = self.strings[l.TASK_INSTRUCTIONS][task_instructions]['content']
        self.instructions[l.Q_INSTRUCTIONS] = self.strings[l.Q_INSTRUCTIONS][q_instructions]['content']
        combined = self.instructions[l.OVERVIEW_INSTRUCTIONS] + "\n\n" + self.instructions[l.TASK_INSTRUCTIONS]
        self.pre_task_instructions = self.usr_msg(combined)

    def set_error_lines(self, error_lines):
        """
        Sets the error lines for the conversation.
        The error_lines argument is a dictionary with the names of the error lines in the strings file.
        The function extracts the content from the strings file and sets the error lines.
        :param error_lines: The error lines to be set.
        """
        self.error_lines = {k: self.strings[k][v]['content'] for k, v in error_lines.items()}

    def assert_ready(self):
        """
        Asserts that all necessary variables (for managing the conversation) are defined.
        """
        assert self.system_message is not None, "System message is not defined"
        assert self.instructions[l.OVERVIEW_INSTRUCTIONS] is not None, "Overview instructions are not defined"
        assert self.instructions[l.TASK_INSTRUCTIONS] is not None, "Task instructions are not defined"
        assert self.instructions[l.Q_INSTRUCTIONS] is not None, "Questions instructions are not defined"
        assert self.error_lines is not None, "Error lines are not defined"

    def start_conversation(self):
        """
        Starts the conversation by asserting readiness, logging the system message, and running the first message.
        """
        self.assert_ready()
        self.chat_messages.append(self.system_message)
        if self.log_live_messages:
            wandb.log(self.system_message)
        self._add_msg(self.pre_task_instructions)

        # First message
        self.run_first_message()
        if self.dead_facilitator:
            return None

    def run_first_message(self):
        """
        Runs the first message of the conversation.
        """
        completion = self.launch_completion()
        if completion is None:
            return

        message = self.ast_msg(completion.choices[0].message.content)
        self._add_msg(message)

    def launch_completion(self):
        """
        Launches a completion for the conversation.
        :return: The completion.
        """
        try:
            completion = self.client.chat.completions.create(messages=self.chat_messages, **self.completion_args)
            return completion
        except Exception as e:
            print("Error in completion! Facilitator down!")
            self.dead_facilitator = True
            return None

    def handle_question(self, question, inspector_logic=None, error_counts=None, return_decision=False):
        """
        Handles a question in the conversation.

        :param question: The question to be handled.
        :param inspector_logic: The logic to be used by the inspector:
            Uses the given logic if not None,
            otherwise uses the logic defined in the inspector_logic attribute,
            and otherwise the INTEGER_ONLY logic.
        :param error_counts: A dict of error counts, may be None if we don't want to count errors.
        :param return_decision: Whether to return the decision. Default is False.
        :return: The decision if return_decision is True.
        """
        # Set the inspector logic
        inspector_hierarchy = [inspector_logic, self.inspector_logic, l.INSPECTORS.INTEGER_ONLY]
        for il in inspector_hierarchy:
            if il is not None:
                inspector_logic = il
                break
        inspector = inspectors[inspector_logic](q_instructions=self.instructions[l.Q_INSTRUCTIONS],
                                                error_lines=self.error_lines,
                                                question=question)

        # start the chat for the question
        chat_content, message = None, None
        first, invalid = True, False

        while first or invalid:  # If it's the first iteration or the user's input is invalid
            if first:
                first = False

            user_content = inspector.process_iteration(chat_content=chat_content, error_counts=error_counts)
            if user_content is not None:
                invalid = True
                message = self.usr_msg(user_content)
                self._add_msg(message)
                completion = self.launch_completion()
                if completion is None:
                    return None
                # TODO - In the future, we might want to export the following to a separate module,
                ## to allow for more complex completion selection logic
                chat_content = completion.choices[0].message.content
                message = self.ast_msg(chat_content)
                self._add_msg(message)
            else:
                invalid = False

        # Get the decision and log it
        decision = inspector.get_decision(chat_content)
        decision['order'] = question['order']
        decision['full_answer'] = chat_content
        decision['tries'] = inspector.tries
        self.decisions.append(decision)
        self.questions.append(question.to_dict())

        if return_decision:
            return decision

    def _add_msg(self, message):
        """
        Adds a message to the conversation.
        In the beginning or the end of the conversation, according to the sys_msg_first attribute.
        :param message: The message to be added.
        """
        if self.sys_msg_first:
            # Add the new message at the end
            self.chat_messages.append(message)
        else:
            # Add the new message before the last message
            self.chat_messages.insert(-1, message)

        if self.log_live_messages:
            wandb.log(message)
