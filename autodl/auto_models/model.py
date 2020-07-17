"""Combine all winner solutions in previous challenges (AutoCV, AutoCV2,
AutoNLP and AutoSpeech).
"""

from autodl.utils.log_utils import logger


def meta_domain_2_model(domain):
    if domain in ["image"]:
        from .auto_image.model import Model as AutoImageModel
        return AutoImageModel
    elif domain in ["video"]:
        from .auto_video.model import Model as AutoVideoModel
        return AutoVideoModel
    elif domain in ["text"]:
        from .model_nlp import Model as AutoNlpModel
        return AutoNlpModel
    elif domain in ["speech"]:
        from .at_speech.model import Model as AutoSpeechModel
        return AutoSpeechModel
    else:
        from .auto_tabular.model import Model as TabularModel
        return TabularModel


class Model:
    """A model that combine all winner solutions. Using domain inferring and
  apply winner solution in the corresponding domain."""

    def __init__(self, metadata):
        """
        Args:
          metadata: an AutoDLMetadata object. Its definition can be found in
              AutoDL_ingestion_program/dataset.py
        """
        self.done_training = False
        self.metadata = metadata
        self.domain = infer_domain(metadata)
        DomainModel = meta_domain_2_model(self.domain)
        self.domain_model = DomainModel(self.metadata)
        self.has_exception = False
        self.y_pred_last = None

    def save_model(self):
        """
        :return (model_obj, config_obj)
        """
        logger.info("start to save model")
        return self.domain_model.save_model(modal_type=self.domain)

    def load_model(self, model_obj, config_obj):
        logger.info("start to load model")
        self.domain_model.load_model(model_obj, config_obj)

    def fit(self, dataset, remaining_time_budget=None):
        """Train method of domain-specific model."""
        # Convert training dataset to necessary format and
        # store as self.domain_dataset_train

        try:
            self.domain_model.fit(dataset, remaining_time_budget)
            self.done_training = self.domain_model.done_training

        except Exception as exp:
            logger.exception("exception in fit: {}".format(exp))
            self.has_exception = True
            self.done_training = True

    def predict(self, dataset, remaining_time_budget=None, test=False):
        """Test method of domain-specific model."""
        # Convert test dataset to necessary format and
        # store as self.domain_dataset_test
        # Make predictions

        if self.done_training is True or self.has_exception is True:
            return self.y_pred_last

        try:
            Y_pred = self.domain_model.predict(dataset, remaining_time_budget=remaining_time_budget, test=test)

            self.y_pred_last = Y_pred
            self.done_training = self.domain_model.done_training

        except MemoryError as mem_error:
            logger.exception("exception in predict: {}".format(mem_error))
            self.has_exception = True
            self.done_training = True
        except Exception as exp:
            logger.exception("exception in predict: {}".format(exp))
            self.has_exception = True
            self.done_training = True

        return self.y_pred_last


def infer_domain(metadata):
    """Infer the domain from the shape of the 4-D tensor.

  Args:
    metadata: an AutoDLMetadata object.
  """
    row_count, col_count = metadata.get_matrix_size(0)
    sequence_size = metadata.get_sequence_size()
    channel_to_index_map = metadata.get_channel_to_index_map()
    domain = None
    if sequence_size == 1:
        if row_count == 1 or col_count == 1:
            domain = "tabular"
        else:
            domain = "image"
    else:
        if row_count == 1 and col_count == 1:
            if len(channel_to_index_map) > 0:
                domain = "text"
            else:
                domain = "speech"
        else:
            domain = "video"
    logger.info(f"dataset domain: {domain}")
    return domain
