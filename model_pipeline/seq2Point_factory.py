from model_pipeline.seq2Point_model import *

class Seq2PointFactory:
    """
    Factory class for Seq2Point models.
    """
    model_mappings = {
        'Original Seq2Point': Seq2PointSimple,
        'Reduced Seq2Point': Seq2PointReduced, 
        'CNN/LSTM Seq2Point': Seq2PointCNNLSTM

    }
    @staticmethod
    def createModel(model_type, input_window_length):
        """
        Creates a Seq2Point model of the specified type.
        """
        return Seq2PointFactory.model_mappings[model_type](input_window_length=input_window_length)

    @staticmethod
    def getModelMappings():
        """
        Returns a list of available Seq2Point models.
        """
        return Seq2PointFactory.model_mappings