import time
import numpy as np
from nn.loss_functions.hinge_loss import hinge_loss
from optimization.adam_optimizer import Adam
from optimization.gd_optimizer import GD
from numpy import linalg as LA

def progress_bar(iterable, text='Epoch progress', end=''):
    """Мониториг выполнения эпохи

    ---------
    Параметры
    ---------
    iterable
        Что-то по чему можно итерироваться

    text: str (default='Epoch progress')
        Текст, выводящийся в начале

    end : str (default='')
        Что вывести в конце выполнения
    """
    max_num = len(iterable)
    iterable = iter(iterable)

    start_time = time.time()
    cur_time = 0
    approx_time = 0

    print('\r', end='')

    it = 0
    while it < max_num:
        it += 1
        print(f"{text}: [", end='')

        progress = int((it - 1) / max_num * 50)
        print('=' * progress, end='')
        if progress != 50:
            print('>', end='')
            print(' ' * (50 - progress - 1), end='')
        print('] ', end='')

        print(f'{it - 1}/{max_num}', end='')
        print(' ', end='')

        print(f'{cur_time}s>{approx_time}s', end='')

        yield next(iterable)

        print('\r', end='')
        print(' ' * (60 + len(text) + len(str(max_num)) + len(str(it)) \
                     + len(str(cur_time)) + len(str(approx_time))),
              end='')
        print('\r', end='')

        cur_time = time.time() - start_time

        approx_time = int(cur_time / it * (max_num - it))
        cur_time = int(cur_time)
        print(end, end='')

def gradient_check(
    x,
    y,
    neural_net,
    epsilon=1e-2,
    num_last=3,
    optim_method="GD",
    lr=3e-4,
    alpha1=None,
    alpha2=None,
):
    """

    Проверка градиента методом численной аппроксимации

    """

    params = list(neural_net.parameters())
    norm_chicl = []
    norm_backprop = []
    norm_backprop_vec = np.zeros(len(params))

    if optim_method == "Adam":
        optimizer = Adam(neural_net.parameters(), lr=lr, alpha1=alpha1, alpha2=alpha2)
    elif optim_method == "GD":
        optimizer = GD(neural_net.parameters(), lr=lr, alpha1=alpha1, alpha2=alpha2)

    neural_net.train()
    optimizer.zero_grad()
    pred = neural_net(x)
    train_loss = hinge_loss(pred, y)
    train_loss.backward()

    for i, param in enumerate(params):
        norm_backprop_vec[i] = LA.norm(param.grads)

    for i, param in enumerate(params):
        sh = param.params.shape
        if i >= len(params) - num_last:
            print(sh)
            for j in range(sh[0]):
                if len(sh) == 1:
                    param_copy = param.params[j]
                    param.params[j] = param_copy + epsilon

                    pred = neural_net(x)
                    loss_p = hinge_loss(pred, y)
                    param.params[j] = param_copy - epsilon

                    pred = neural_net(x)
                    loss_m = hinge_loss(pred, y)
                    param.params[j] = param_copy
                    norm_chicl.append((loss_p.loss - loss_m.loss) / (2 * epsilon))
                    norm_backprop.append(param.grads[j])

                elif len(sh) == 2:
                    for k in range(sh[1]):
                        param_copy = param.params[j][k]
                        param.params[j][k] = param_copy + epsilon

                        pred = neural_net(x)
                        loss_p = hinge_loss(pred, y)
                        param.params[j][k] = param_copy - epsilon

                        pred = neural_net(x)
                        loss_m = hinge_loss(pred, y)
                        param.params[j][k] = param_copy
                        norm_chicl.append((loss_p.loss - loss_m.loss) / (2 * epsilon))
                        norm_backprop.append(param.grads.T[j][k])

    norm_chicl = np.array(norm_chicl)
    norm_backprop = np.array(norm_backprop)

    diff = LA.norm(norm_chicl - norm_backprop) / (
        LA.norm(norm_chicl) + LA.norm(norm_backprop)
    )

    print("backprop norm: ", LA.norm(norm_backprop))
    print("grad check norm:", LA.norm(norm_chicl))

    if diff <= epsilon:
        print("Backprop is incorrect!")
    else:
        print("Backprop is correct!")
