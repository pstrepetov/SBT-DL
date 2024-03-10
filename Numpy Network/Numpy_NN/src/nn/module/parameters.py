import numpy as np

class Parameters:
    """Здесь будут храниться параметры и их градиенты

    shape : tuple or int
        Определяет размер массива параметров
    """
    def __init__(self, shape):
        self.shape = shape
        self.params = np.zeros(shape)
        self.grads = np.zeros(shape)
        self.m = None
        self.v = None

    def _init_params(self, method='kaiming'):
        """Инициализация параметров

        ---------
        Параметры
        ---------
        method : str (default='kaiming')
            Метод инициализации параметров
            Пока доступен только 'kaiming'
        """
        if method=='kaiming':
            # TODO: Реализовать инициализацию параметров модели методом Kaiming
            inpt_len = self.shape[0] if len(self.shape) == 2 else np.prod(self.shape[1:])
            self.params = np.random.normal(0, np.sqrt(2/inpt_len), self.shape)
