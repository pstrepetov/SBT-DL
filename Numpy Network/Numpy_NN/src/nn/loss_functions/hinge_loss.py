import numpy as np
from nn.loss_functions.loss import Loss


def hinge_loss(inpt, target):
    """Реализует функцию ошибки hinge loss

    ---------
    Параметры
    ---------
    inpt : Tensor
        Предсказание модели

    target
        Список реальных классов
        Одномерный массив

    ----------
    Возвращает
    ----------
    loss : Loss
        Ошиба
    """

    # Рассчет hinge loss
    correct_labels = (range(len(target)), target)
    correct_class_scores = inpt.array[correct_labels]  # Nx1

    loss_element = inpt.array - correct_class_scores[:, np.newaxis] + 1  # NxC
    correct_classifications = np.where(loss_element <= 0)

    loss_element[correct_classifications] = 0
    loss_element[correct_labels] = 0

    grad = np.ones(loss_element.shape, dtype=np.float16)
    grad[correct_classifications], grad[correct_labels] = 0, 0
    grad[correct_labels] = -1 * grad.sum(axis=-1)
    grad /= inpt.array.shape[0]

    loss = np.sum(loss_element) / inpt.array.shape[0]

    return Loss(loss, grad, inpt.model)
