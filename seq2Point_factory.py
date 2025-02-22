from seq2Point_model import *

class Seq2PointFactory:
    """
    Factory class for Seq2Point models.
    """
    @staticmethod
    def createModel(model_type, input_window_length):
        """
        Creates a Seq2Point model of the specified type.
        """
        if model_type.lower() == 'seq2pointsimple':
            return Seq2PointSimple(input_window_length=input_window_length)
        elif model_type.lower() == 'seq2pointreduced':
            return Seq2PointReduced(input_window_length=input_window_length)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))