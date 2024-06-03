from abc import ABC, abstractmethod
import cp_literals as l
from cp_utils import is_float


class IInspector(ABC):
    """
    This is the abstract class for the inspectors. It contains the basic structure for the inspectors.
    An inspector is responsible for validating the content of a message, and returning a new message if the content is
    invalid. Furthermore, it is responsible for counting the errors and extracting the decision from the content.
    The inspector can be thought of as a real inspector, enforcing the required format of the content.
    """

    def __init__(self, q_instructions, error_lines, question):
        """
        Constructor for the inspector.
        :param q_instructions: The instructions that comes before the question.
        :param error_lines: The error messages that are used to inform the user about the errors in the content.
        :param question: The question that the inspector is responsible for.
        """
        self.q_instructions = q_instructions
        self.error_lines = error_lines
        self.question = question
        self.tries = 0

    def process_iteration(self, chat_content=None, error_counts=None):
        """
        Processes an iteration of the conversation.
        Firstly, checks if the chat content is valid (if it's the first iteration or if the content is None).
        If it's not, returns the user content with the error message.
        If it's valid, returns None, since no new content is needed.
        :param chat_content: The content of the chat. Default is None.
        :param error_counts: The error counts dictionary. Can be None, in case we don't want to count errors.
        :return: The user content if it's the first iteration or the content is invalid. Otherwise, None.
        """
        invalid = False

        if chat_content is not None:
            # There is content to validate
            flags, error_counts = self.evaluate_completion(content=chat_content, error_counts=error_counts)
            invalid = any(flags.values())
        else:
            # This is the first iteration
            flags = {'first': True}

        if flags.get('first', False) or invalid:
            user_content = self.prepare_question(flags=flags)
            return user_content
        else:
            return None  # It's not the first iteration and the content is valid

    def prepare_question(self, flags):
        """
        Prepares the question content for the conversation, according to the flags (error / first flags).

        :param flags: The flags for the question content.
        :return: The content of the question.
        """
        self.tries += 1

        error_line = self.error_message(flags=flags)
        question_text = '#'.join(self.question['prompt'].split('#')[:-1])
        content = f"{error_line}{self.q_instructions}\n\n{question_text}"
        return content

    @abstractmethod
    def error_message(self, flags):
        """
        Returns the error message for the conversation.
        Here, there would be implemented the logic for selecting the required error message, in case of multiple errors.
        :param flags: The flags for the conversation.
        :return: The error message.
        """
        pass

    @abstractmethod
    def evaluate_completion(self, content, error_counts=None):
        """
        Evaluates the completion of the content.
        :param content: The content of the chat's message.
        :param error_counts: The error counts dictionary. Can be None, in case we don't want to count errors.
        :return: The error flags and the error counts.
        """
        pass

    @abstractmethod
    def get_decision(self, content):
        """
        Gets the decision from the chat content.
        According to the inspector's logic, the decision can be the whole content, or a part of it.
        :param content: The content of the chat's message.
        :return: The decision from the chat content.
        """
        pass

    @staticmethod
    def count_errors(error_flags, error_counts):
        """
        Counts the errors in the conversation.

        :param error_flags: The error flags of the chat response.
        :param error_counts: The error counts dictionary.
        :return: The error counts dictionary, updated with the new errors.
        NOTE: This method is static, since it's not dependent on the instance of the inspector.
        NOTE: Dicts are mutable, so we don't need to actually return the error_counts, however, for clarity, we do.
        """
        for k, v in error_flags.items():
            error_counts[k] += v
        return error_counts


class IntegerOnly(IInspector):
    """
    This inspector is used to validate that the content is an integer only representing the confidence.
    """

    def error_message(self, flags):
        """
        See the abstract method for documentation.
        """
        error_line = ""
        hierarchy = ['first', 'not_digits', '50_conf']
        for error in hierarchy:
            if flags.get(error, False):
                error_line = self.error_lines[error]
                break
        return error_line

    def evaluate_completion(self, content, error_counts=None):
        """
        See the abstract method for documentation.
        """
        error_flags = {}

        content = self.get_decision(content, validated=False)['sub_val']
        # Validate message
        error_flags['not_digits'] = not content.isdigit()
        if not error_flags['not_digits']:
            error_flags['50_conf'] = int(content) == 50
        # Count errors
        if error_counts is not None:
            error_counts = self.count_errors(error_flags=error_flags, error_counts=error_counts)
            return error_flags, error_counts
        return error_flags, None

    def get_decision(self, content, validated=True):
        """
        See the abstract method for documentation.
        """
        if not validated:
            return {'sub_val': content}

        confidence = int(content)
        decision = {'sub_val': confidence,
                    'binary_decision': int(confidence > 50),
                    'normalized_conf': abs(confidence - 50) * 2}

        return decision


class ShortExplanation(IInspector):
    """
    This inspector is used to validate that the content is an integer, then ', ', then a short explanation.
    I.E, in the format of '<integer>, <explanation>'.
    If the first argument isn't an integer, a 'not_digits' error is raised.
    If the integer is 50, a '50_conf' error is raised.
    If the explanation is too long (over 50 words), a 'long_explanation' error is raised.
    """

    def error_message(self, flags):
        """
        See the abstract method for documentation.
        """
        error_line = ""
        hierarchy = ['first', 'not_digits', '50_conf']
        for error in hierarchy:
            if flags.get(error, False):
                error_line = self.error_lines[error]
                break
        return error_line

    def evaluate_completion(self, content, error_counts=None):
        """
        See the abstract method for documentation.
        """
        error_flags = {}

        splitted = content.split(', ')
        digits = self.get_decision(content, validated=False)['sub_val']
        explanation = ', '.join(splitted[1:])
        # Validate message
        error_flags['not_digits'] = not digits.isdigit()
        if not error_flags['not_digits']:
            error_flags['50_conf'] = int(digits) == 50
        error_flags['long_explanation'] = len(explanation.split(' ')) > 50
        # Count errors
        if error_counts is not None:
            error_counts = self.count_errors(error_flags=error_flags, error_counts=error_counts)
            return error_flags, error_counts
        return error_flags, None

    def get_decision(self, content, validated=True):
        """
        See the abstract method for documentation.
        """
        splitted = content.split(', ')
        confidence = splitted[0]
        if not validated:
            return {'sub_val': confidence}

        confidence = int(confidence)
        decision = {'sub_val': confidence,
                    'binary_decision': int(confidence > 50),
                    'normalized_conf': abs(confidence - 50) * 2}

        return decision


class ProvideTime(IInspector):
    """
    This inspector is used to validate that the content is an integer, then ', ', then a time.
    I.E, in the format of '<integer>, <time>'.
    If the first argument (the confidence) isn't an integer, then a 'not_digits' error is raised.
    If the integer is 50, a '50_conf' error is raised.
    If the time is not an integer, or it's negative ,a 'invalid_time' error is raised.
    """

    def error_message(self, flags):
        """
        See the abstract method for documentation.
        """
        if flags.get('first', False):
            error_line = self.error_lines['first']
        else:
            e1, e2 = "", ""
            hierarchy = ['not_digits', '50_conf']
            for error in hierarchy:
                if flags.get(error, False):
                    e1 = self.error_lines[error] + '\n'
                    break
            if flags.get('invalid_time', False):
                e2 = self.error_lines['invalid_time']
            error_line = f"{e1}{e2}"
        return error_line

    def evaluate_completion(self, content, error_counts=None):
        """
        See the abstract method for documentation.
        """
        error_flags = {}

        decision = self.get_decision(content, validated=False)
        if decision is None:
            error_flags['not_digits'] = True

        else:
            confidence = decision['sub_val']
            time = decision['time']
            # Validate message
            error_flags['not_digits'] = not confidence.isdigit()
            if not error_flags['not_digits']:
                error_flags['50_conf'] = int(confidence) == 50
            if (not is_float(time)) or float(time) < 0:
                error_flags['invalid_time'] = True

        # Count errors
        if error_counts is not None:
            error_counts = self.count_errors(error_flags=error_flags, error_counts=error_counts)
            return error_flags, error_counts
        return error_flags, None

    def get_decision(self, content, validated=True):
        """
        See the abstract method for documentation.
        """

        splitted = content.split(',')
        # Todo - If an answer is not yet validated,
        ## we cannot assume that it has 2 parts to unpack
        try:
            confidence, time = splitted
            confidence, time = confidence.strip(), time.strip()
        except ValueError:
            if validated:
                raise ValueError("The content is validated, and it doesn't have 2 parts to unpack.")
            else:
                return None

        if not validated:
            return {'sub_val': confidence, 'time': time}

        confidence, time = int(confidence), float(time)
        decision = {'sub_val': confidence,
                    'binary_decision': int(confidence > 50),
                    'normalized_conf': abs(confidence - 50) * 2,
                    'time': time}

        return decision


inspectors = {l.INSPECTORS.INTEGER_ONLY: IntegerOnly,
              l.INSPECTORS.SHORT_EXPLANATION: ShortExplanation,
              l.INSPECTORS.PROVIDE_TIME: ProvideTime}
