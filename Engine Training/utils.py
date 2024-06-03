import random


class ListUtils:
    @staticmethod
    def flat_shuffle(lst: list) -> list:
        """
        Get a list of lists, flatten it and shuffle it
        :param lst: list of lists
        :return: flattened and shuffled list
        """
        flat = [item for sublist in lst for item in sublist]
        random.shuffle(flat)
        return flat

    @staticmethod
    def shufflee(lst: list) -> list:
        random.shuffle(lst)
        return lst
