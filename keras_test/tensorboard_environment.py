import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

class TensorboardEnvironment(object):
    """description of class"""
    def __init__(self, session_name=''):
        self.session_name = session_name
        self.old_session = KTF.get_session()

    def __enter__(self):
        self.graph = tf.Graph().as_default()
        self.graph.__enter__()
        self.session = tf.Session(self.session_name)
        KTF.set_session(self.session)
        # set learning phase parameter (needed if the model is different in the training phase such as Dropout)
        KTF.set_learning_phase(1)
        
    def __exit__(self, exception_type, exception_value, traceback):
        self.graph.__exit__(exception_type, exception_value, traceback)
        KTF.set_session(self.old_session)
