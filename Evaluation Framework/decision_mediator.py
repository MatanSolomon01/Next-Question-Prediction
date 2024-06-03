import ef_literals as l


class DecisionMediator:
    # TODO: We may want to implement this the same way the Inspector does, with classes the inherit an implementation.
    ## Furthermore, we may want to change the implementation according to the recent changes in the facilitator, that
    ## now uses the inspector to return the specific decision parameters according to the chosen logic.

    """
    The DecisionMediator class is responsible for converting the decision from the facilitator to the required format
    of decision features, for the strategy to use (maybe for feeding to the engine).
    """

    def __init__(self, client, decision_mediator_method):
        """
        :param client: The OpenAI client
        :param decision_mediator_method: The method to convert the decision
        """
        self.client = client
        self.decision_mediator_method = decision_mediator_method

    def convert_decision(self, decision, question):
        """
        Convert the decision for a given question to the required format.
        """
        convert = self.methods[self.decision_mediator_method]
        return convert(self, decision, question)

    def embed_explanation(self, decision, **kwargs):
        """
        Embed the explanation of the decision
        """
        explanation = decision['full_answer']

        EmbeddingResponse = self.client.embeddings.create(input=explanation, model='text-embedding-ada-002-Matan')
        embedding = EmbeddingResponse.data[0].embedding
        return embedding

    def human_features(self, decision, question, **kwargs):
        """
         For the ProvideTime inspector, the features are:
         ['userconf', 'user_sub_val', 'user_ans_is_match', 'time', 'token_path', 'term_match', 'word_net']
        """

        conf, time = decision['full_answer'].split(', ')
        conf = int(conf)
        features = [2 * abs(conf - 50),
                    conf,
                    int(conf > 50),
                    float(time),
                    question['token_path'],
                    question['term_match'],
                    question['word_net']]
        return features

    methods = {l.DM.EMBED_EXPLANATION: embed_explanation,
               l.DM.HUMAN_FEATURES: human_features}
