class Trainer:
    """
    Abstract base class for training models. Subclasses should implement the train method.
    """

    def train(self, *args, **kwargs):
        """
        Train the model. This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement the train method.")