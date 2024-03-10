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
            # TODO: Реализовать batch norm в eval фазе
            self.inpt_hat = (inpt - self.E)/np.sqrt(self.D + self.eps)
            out = self.gamma * self.inpt_hat + self.beta
            return out

        # TODO: Реализовать batch norm в train фазе
        batch_mean = np.mean(inpt, axis=0)
        self.tmp_D = np.var(inpt, axis=0)
        self.inpt_hat = (inpt - batch_mean)/np.sqrt(self.tmp_D + self.eps)
        out = self.gamma * self.inpt_hat + self.beta
        if (self.E is not None and self.D is not None):
            self.E = self.E * self.momentum + batch_mean * (1 - self.momentum)
            self.D = self.D * self.momentum + self.tmp_D * (1 - self.momentum)
        else:
            self.E = batch_mean
            self.D = self.tmp_D

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

        # TODO: Реализовать рассчет градиента в batch norm
        self.beta.grads = np.sum(grads, axis=0)
        self.gamma.grads = np.sum(grads*self.inpt_hat, axis=0)
        
        sqrtvar = np.sqrt(self.tmp_D + self.eps)
        xmu = self.inpt_hat * sqrtvar
        N,D = xmu

        divar = np.sum(grads*xmu, axis=0)
        dxmu1 = grads/sqrtvar

        dsqrtvar = -1. /(sqrtvar**2) * divar

        dvar = 0.5 * 1. /np.sqrt(self.tmp_D + self.eps) * dsqrtvar

        dsq = 1. /N * np.ones((N,D)) * dvar

        dxmu2 = 2 * xmu * dsq

        dx1 = (dxmu1 + dxmu2)
        dmu = -1 * np.sum(dxmu1+dxmu2, axis=0)
        
        dx2 = 1. /N * np.ones((N,D)) * dmu
        input_grads = (dx1 + dx2) * self.gamma

        return input_grads

    def _train(self):
        """Переводит модель в режим обучения"""
        self.regime = "Train"

    def _eval(self):
        """Переводит модель в режим оценивания"""
        self.regime = "Eval"

    def __repr__(self):
        return f"BatchNorm(in_dim={self.in_dim}, eps={self.eps})"
