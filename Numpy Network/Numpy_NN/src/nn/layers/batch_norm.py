from nn.module.parameters import Parameters
import numpy as np


class BatchNorm:
    """Реализует Batch norm

    ---------
    Параметры
    ---------
    in_dim : int
        Размерность входного вектора

    eps : float (default=1e-5)
        Параметр модели,
        позволяет избежать деления на 0

    momentum : float (default=0.1)
        Параметр модели
        Используется для обновления статистик
    """

    def __init__(self, in_dim, eps=1e-5, momentum=0.1):
        self.in_dim = in_dim
        self.eps = eps
        self.momentum = 0.1

        self.regime = "Train"

        self.gamma = Parameters((in_dim,))
        self.gamma._init_params()

        self.beta = Parameters(in_dim)

        self.E = np.zeros(in_dim)
        self.D = np.zeros(in_dim)

        self.inpt_hat = None
        self.tmp_D = None

    def forward(self, inpt):
        """Реализует forward-pass

        ---------
        Параметры
        ---------
        inpt : np.ndarray, shape=(M, N_in)
            Входные данные

        ----------
        Возвращает
        ----------
        output : np.ndarray, shape=(M, N_in)
            Выход слоя
        """
        if self.regime == "Eval":
            self.inpt_hat = (inpt - self.E) / np.sqrt(self.D + self.eps)
            out = self.inpt_hat * self.gamma.params + self.beta.params
            return out

        self.tmp_D = np.var(inpt, axis=0)
        self.inpt_hat = (inpt - np.mean(inpt, axis=0)) / np.sqrt(self.tmp_D + self.eps)
        out = self.inpt_hat * self.gamma.params + self.beta.params

        self.D = (1 - self.momentum) * self.D + self.momentum * self.tmp_D
        self.E = (1 - self.momentum) * self.E + self.momentum * np.mean(inpt, axis=0)
        return out

    def __call__(self, *inpt):
        """Аналогично forward"""
        return self.forward(*inpt)

    def parameters(self):
        """Возвращает параметры модели"""
        return (self.gamma, self.beta)

    def _zero_grad(self):
        """Обнуляет градиенты модели"""
        self.gamma.grads = np.zeros(self.gamma.shape)
        self.beta.grads = np.zeros(self.beta.shape)

    def _compute_gradients(self, grads):
        """Считает градиенты модели"""
        if self.regime == "Eval":
            raise RuntimeError("Нельзя посчитать градиенты в режиме оценки")

        batch_size = self.inpt_hat.shape[0]

        self.beta.grads = np.sum(grads, axis=0)
        d_gamma_x_hat = grads

        self.gamma.grads = np.sum(self.inpt_hat * d_gamma_x_hat, axis=0)

        x_mu = self.inpt_hat * np.sqrt(self.tmp_D + self.eps)
        d_var = 0.5 / np.sqrt(self.tmp_D + self.eps) * (-1. / (self.tmp_D + self.eps) * np.sum(
            x_mu * self.gamma.params * d_gamma_x_hat,
            axis=0
        ))

        d_sq_dev = np.ones((batch_size, self.in_dim)) / batch_size * d_var
        d_mu = -np.sum(
            (1. / np.sqrt(self.tmp_D + self.eps) * self.gamma.params * d_gamma_x_hat) + (2 * x_mu * d_sq_dev),
            axis=0
        )
        input_grads = (1. / np.sqrt(self.tmp_D + self.eps) * self.gamma.params * d_gamma_x_hat) + (2 * x_mu * d_sq_dev) + np.ones((batch_size, self.in_dim)) / batch_size * d_mu

        return input_grads

    def _train(self):
        """Переводит модель в режим обучения"""
        self.regime = "Train"

    def _eval(self):
        """Переводит модель в режим оценивания"""
        self.regime = "Eval"

    def __repr__(self):
        return f"BatchNorm(in_dim={self.in_dim}, eps={self.eps})"
