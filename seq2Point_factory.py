from seq2Point_model import *

class Seq2PointFactory:
    """
    Factory class for Seq2Point models.
    """
    @staticmethod
    def createModel(model_type, **kwargs):
        """
        Creates a Seq2Point model of the specified type.
        """
        if model_type.lower() == 'seq2pointsimple':
            return Seq2PointSimple(**kwargs)
        elif model_type.lower() == 'seq2pointlstm':
            return Seq2PointLSTM(**kwargs)
        else:
            raise ValueError('Unknown model type: {}'.format(model_type))