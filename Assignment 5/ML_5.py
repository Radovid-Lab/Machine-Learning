import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
import random
import math

state = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 2, 1, 1, 0],
                  [0, 1, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 0, 1, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 1, 0, 0, 1, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0]
                  ])

action = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}  # up down left right

arrow = {0: '↑', 1: '↓', 2: '←', 3: '→'}


def Q_itetation(gamma, state):
    q_matrix = np.zeros((9, 7, 4))
    iteration = 0
    change = 1
    while change == 1:
        change = 0
        iteration = iteration + 1
        tmp_matrix = q_matrix.copy()
        for i in range(0, len(state)):
            for j in range(0, len(state[0])):
                if state[i, j] == 0 or state[i, j] == 2:
                    continue
                else:
                    for a in range(0, 4):
                        q_old = q_matrix[i, j, a]
                        if action[a] == 'up':
                            p = i - 1
                            q = j
                            if p >= 0 and state[p, q] == 0:
                                continue
                        if action[a] == 'down':
                            p = i + 1
                            q = j
                            if p <= 8 and state[p, q] == 0:
                                continue
                        if action[a] == 'left':
                            p = i
                            q = j - 1
                            if q >= 0 and state[p, q] == 0:
                                continue
                        if action[a] == 'right':
                            p = i
                            q = j + 1
                            if q <= 6 and state[p, q] == 0:
                                continue
                        if state[p, q] == 2:
                            q_matrix[i, j, a] = 1 + gamma * max(tmp_matrix[p, q])
                        else:
                            q_matrix[i, j, a] = gamma * max(tmp_matrix[p, q])
                        if q_old != q_matrix[i, j, a]:
                            change = 1
    return iteration, q_matrix


def updateValue(qfunction):
    value = np.zeros((9, 7))
    for i in range(0, len(qfunction)):
        for j in range(0, len(qfunction[0])):
            value[i, j] = max(qfunction[i, j])
    return value


def policy(matrix):
    cmap = colors.ListedColormap(['Black', 'White'])
    plt.figure(figsize=(5, 5))
    plt.pcolor(state[::-1], cmap=cmap, edgecolors='k', linewidths=3)
    plt.plot(0, 1, marker=r'$\downarrow$')
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            if max(matrix[i, j]) == 0:
                continue
            else:
                plt.text(j + 0.5, 9 - i - 0.5, arrow[np.argmax(matrix[i, j])], ha='center', va='center', color='b')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def heatmap(matrix):
    cmap = colors.ListedColormap(['Black', 'White'])
    plt.figure(figsize=(5, 5))
    plt.pcolor(matrix[::-1], cmap='winter', edgecolors='k', linewidths=3)
    plt.plot(0, 1, marker=r'$\downarrow$')
    for i in range(0, len(matrix)):
        for j in range(0, len(matrix[0])):
            if round(matrix[i, j], 2) != 0:
                plt.text(j + 0.5, 9 - i - 0.5, round(matrix[i, j], 2), ha='center', va='center', color='k')
    plt.xticks([])
    plt.yticks([])
    plt.show()


def Q_learning(epsilon, gamma, learningrate):
    q_matrix = np.zeros((9, 7, 4))
    currentState = [7, 5]
    step = 0
    unchangeCount = 0
    while unchangeCount < 1500 and step < 30000:
        if state[currentState[0], currentState[1]] == 2:
            currentState = [7, 5]
            continue
        optimalAction = np.argmax(q_matrix[currentState[0], currentState[1]])
        dice = random.random()
        if dice < epsilon:  # exploration
            a = math.floor(4 * random.random())
        else:
            a = optimalAction
        # apply action
        if action[a] == 'up':
            p = currentState[0] - 1
            q = currentState[1]
            if p >= 0 and state[p, q] == 0:
                step = step + 1
                continue
        if action[a] == 'down':
            p = currentState[0] + 1
            q = currentState[1]
            if p <= 8 and state[p, q] == 0:
                step = step + 1
                continue
        if action[a] == 'left':
            p = currentState[0]
            q = currentState[1] - 1
            if q >= 0 and state[p, q] == 0:
                step = step + 1
                continue
        if action[a] == 'right':
            p = currentState[0]
            q = currentState[1] + 1
            if q <= 6 and state[p, q] == 0:
                step = step + 1
                continue
        if state[p, q] == 2:
            reward = 1
        else:
            reward = 0
        nextBestAction = max(q_matrix[p, q])
        q_old = q_matrix[currentState[0], currentState[1], a]
        q_matrix[currentState[0], currentState[1], a] = q_matrix[currentState[0], currentState[1], a] + learningrate * (
                reward + gamma * nextBestAction - q_matrix[currentState[0], currentState[1], a])
        if q_old == q_matrix[currentState[0], currentState[1], a]:
            unchangeCount = unchangeCount + 1
        currentState = [p, q]
        step = step + 1
    if step >= 30000:
        print('not converge')
    return step, q_matrix


def twoNorm(matrix1, matrix2):
    # two q_matrixs
    result = np.power(np.sum(np.power(matrix1 - matrix2, 2)), 1 / 2)
    return result


def differencePlot(epsilon, gamma, learningrate, matrix, color):
    sumx=np.array([])
    sumy = np.array([])
    for i in range(50):
        q_matrix = np.zeros((9, 7, 4))
        currentState = [7, 5]
        step = 1
        useless, m1 = Q_itetation(gamma, matrix)
        matrix1 = updateValue(m1)
        x = [0]
        y = []
        y.append(twoNorm(matrix1, updateValue(q_matrix)))
        label ='plain + eps= {eps}, alp= {alp} '.format(eps=epsilon, alp=learningrate)
        while step < 30000:
            if state[currentState[0], currentState[1]] == 2:
                currentState = [7, 5]
                continue
            optimalAction = np.argmax(q_matrix[currentState[0], currentState[1]])
            dice = random.random()
            if dice < epsilon:  # exploration
                a = math.floor(4 * random.random())
            else:
                a = optimalAction
            # apply action
            if action[a] == 'up':
                p = currentState[0] - 1
                q = currentState[1]
                if p >= 0 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'down':
                p = currentState[0] + 1
                q = currentState[1]
                if p <= 8 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'left':
                p = currentState[0]
                q = currentState[1] - 1
                if q >= 0 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'right':
                p = currentState[0]
                q = currentState[1] + 1
                if q <= 6 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if state[p, q] == 2:
                reward = 1
            else:
                reward = 0

            nextBestAction = np.argmax(q_matrix[p, q])

            q_matrix[currentState[0], currentState[1], a] = q_matrix[currentState[0], currentState[1], a] + learningrate * (
                    reward + gamma * q_matrix[p,q,nextBestAction] - q_matrix[currentState[0], currentState[1], a])
            currentState = [p, q]
            x.append(step)
            step = step + 1
            y.append(twoNorm(matrix1, updateValue(q_matrix)))
        if len(sumx)==0:
            sumx=x
            sumy=np.array(y)
        else:
            sumx+=np.array(x)
            sumy+=np.array(y)
    plt.plot(sumx/50, sumy/50, label=label, color=color)


def priorKnowledge(epsilon, gamma, learningrate):
    q_matrix = np.zeros((9, 7, 4))
    prior = np.ones((9, 7, 4))
    currentState = [7, 5]
    step = 0
    unchangeCount = 0
    while unchangeCount < 1500 and step < 30000:
        if state[currentState[0], currentState[1]] == 2:
            currentState = [7, 5]
            continue
        optimalAction = np.argmax(q_matrix[currentState[0], currentState[1]])
        dice = random.random()
        if dice < epsilon:  # exploration
            choice,=np.nonzero(prior[currentState[0], currentState[1]])
            tmp = math.floor(len(choice) * random.random())
            a = choice[tmp]
        else:
            a = optimalAction
        # apply action
        if action[a] == 'up':
            p = currentState[0] - 1
            q = currentState[1]
            if p >= 0 and state[p, q] == 0:
                step = step + 1
                prior[currentState[0], currentState[1],a]=0
                continue
        if action[a] == 'down':
            p = currentState[0] + 1
            q = currentState[1]
            if p <= 8 and state[p, q] == 0:
                step = step + 1
                prior[currentState[0], currentState[1], a] = 0
                continue
        if action[a] == 'left':
            p = currentState[0]
            q = currentState[1] - 1
            if q >= 0 and state[p, q] == 0:
                step = step + 1
                prior[currentState[0], currentState[1], a] = 0
                continue
        if action[a] == 'right':
            p = currentState[0]
            q = currentState[1] + 1
            if q <= 6 and state[p, q] == 0:
                step = step + 1
                prior[currentState[0], currentState[1], a] = 0
                continue
        if state[p, q] == 2:
            reward = 1
        else:
            reward = 0
        nextBestAction = max(q_matrix[p, q])
        q_old = q_matrix[currentState[0], currentState[1], a]
        q_matrix[currentState[0], currentState[1], a] = q_matrix[currentState[0], currentState[1], a] + learningrate * (
                reward + gamma * nextBestAction - q_matrix[currentState[0], currentState[1], a])
        if q_old == q_matrix[currentState[0], currentState[1], a]:
            unchangeCount = unchangeCount + 1
        currentState = [p, q]
        step = step + 1
    if step >= 30000:
        print('not converge')
    return step, q_matrix


def differencePlot2(epsilon, gamma, learningrate, matrix, color):
    sumx=np.array([])
    sumy = np.array([])
    for i in range(50):
        q_matrix = np.zeros((9, 7, 4))
        currentState = [7, 5]
        prior = np.ones((9, 7, 4))
        step = 1
        useless, m1 = Q_itetation(gamma, matrix)
        matrix1 = updateValue(m1)
        x = [0]
        y = []
        y.append(twoNorm(matrix1, updateValue(q_matrix)))
        label = 'prior + eps= {eps}, alp= {alp} '.format(eps=epsilon, alp=learningrate)
        while step < 30000:
            if state[currentState[0], currentState[1]] == 2:
                currentState = [7, 5]
                continue
            optimalAction = np.argmax(q_matrix[currentState[0], currentState[1]])
            dice = random.random()
            if dice < epsilon:  # exploration
                choice,=np.nonzero(prior[currentState[0], currentState[1]])
                tmp = math.floor(len(choice) * random.random())
                a = choice[tmp]
            else:
                a = optimalAction
            # apply action
            if action[a] == 'up':
                p = currentState[0] - 1
                q = currentState[1]
                if p >= 0 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'down':
                p = currentState[0] + 1
                q = currentState[1]
                if p <= 8 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'left':
                p = currentState[0]
                q = currentState[1] - 1
                if q >= 0 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'right':
                p = currentState[0]
                q = currentState[1] + 1
                if q <= 6 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if state[p, q] == 2:
                reward = 1
            else:
                reward = 0

            nextBestAction = np.argmax(q_matrix[p, q])

            q_matrix[currentState[0], currentState[1], a] = q_matrix[currentState[0], currentState[1], a] + learningrate * (
                    reward + gamma * q_matrix[p,q,nextBestAction] - q_matrix[currentState[0], currentState[1], a])
            currentState = [p, q]
            x.append(step)
            step = step + 1
            y.append(twoNorm(matrix1, updateValue(q_matrix)))
        if len(sumx)==0:
            sumx=x
            sumy=np.array(y)
        else:
            sumx+=np.array(x)
            sumy+=np.array(y)
    plt.plot(sumx/50, sumy/50, label=label, color=color)



def differencePlot3(epsilon, gamma, learningrate, matrix, color):
    sumx=np.array([])
    sumy = np.array([])
    for i in range(50):
        q_matrix = np.ones((9, 7, 4))*(0.1)
        currentState = [7, 5]
        step = 1
        useless, m1 = Q_itetation(gamma, matrix)
        matrix1 = updateValue(m1)
        x = [0]
        y = []
        y.append(twoNorm(matrix1, updateValue(q_matrix)))
        label ='better initialization + eps= {eps}, alp= {alp} '.format(eps=epsilon, alp=learningrate)
        while step < 30000:
            if state[currentState[0], currentState[1]] == 2:
                currentState = [7, 5]
                continue
            optimalAction = np.argmax(q_matrix[currentState[0], currentState[1]])
            dice = random.random()
            if dice < epsilon:  # exploration
                a = math.floor(4 * random.random())
            else:
                a = optimalAction
            # apply action
            if action[a] == 'up':
                p = currentState[0] - 1
                q = currentState[1]
                if p >= 0 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'down':
                p = currentState[0] + 1
                q = currentState[1]
                if p <= 8 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'left':
                p = currentState[0]
                q = currentState[1] - 1
                if q >= 0 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'right':
                p = currentState[0]
                q = currentState[1] + 1
                if q <= 6 and state[p, q] == 0:
                    step = step + 1
                    x.append(step)
                    y.append(y[-1])
                    continue
            if state[p, q] == 2:
                reward = 1
            else:
                reward = 0

            nextBestAction = np.argmax(q_matrix[p, q])

            q_matrix[currentState[0], currentState[1], a] = q_matrix[currentState[0], currentState[1], a] + learningrate * (
                    reward + gamma * q_matrix[p,q,nextBestAction] - q_matrix[currentState[0], currentState[1], a])
            currentState = [p, q]
            x.append(step)
            step = step + 1
            y.append(twoNorm(matrix1, updateValue(q_matrix)))
        if len(sumx)==0:
            sumx=x
            sumy=np.array(y)
        else:
            sumx+=np.array(x)
            sumy+=np.array(y)
    plt.plot(sumx/50, sumy/50, label=label, color=color)


def differencePlot4(epsilon, gamma, learningrate, matrix, color):
    sumx=np.array([])
    sumy = np.array([])
    for i in range(50):
        q_matrix = np.ones((9, 7, 4))*(0.1)
        currentState = [7, 5]
        prior = np.ones((9, 7, 4))
        step = 1
        useless, m1 = Q_itetation(gamma, matrix)
        matrix1 = updateValue(m1)
        x = [0]
        y = []
        y.append(twoNorm(matrix1, updateValue(q_matrix)))
        label = 'combination + eps= {eps}, alp= {alp} '.format(eps=epsilon, alp=learningrate)
        while step < 30000:
            if state[currentState[0], currentState[1]] == 2:
                currentState = [7, 5]
                continue
            optimalAction = np.argmax(q_matrix[currentState[0], currentState[1]])
            dice = random.random()
            if dice < epsilon:  # exploration
                choice,=np.nonzero(prior[currentState[0], currentState[1]])
                tmp = math.floor(len(choice) * random.random())
                a = choice[tmp]
            else:
                a = optimalAction
            # apply action
            if action[a] == 'up':
                p = currentState[0] - 1
                q = currentState[1]
                if p >= 0 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'down':
                p = currentState[0] + 1
                q = currentState[1]
                if p <= 8 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'left':
                p = currentState[0]
                q = currentState[1] - 1
                if q >= 0 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if action[a] == 'right':
                p = currentState[0]
                q = currentState[1] + 1
                if q <= 6 and state[p, q] == 0:
                    step = step + 1
                    prior[currentState[0], currentState[1], a] = 0
                    x.append(step)
                    y.append(y[-1])
                    continue
            if state[p, q] == 2:
                reward = 1
            else:
                reward = 0

            nextBestAction = np.argmax(q_matrix[p, q])

            q_matrix[currentState[0], currentState[1], a] = q_matrix[currentState[0], currentState[1], a] + learningrate * (
                    reward + gamma * q_matrix[p,q,nextBestAction] - q_matrix[currentState[0], currentState[1], a])
            currentState = [p, q]
            x.append(step)
            step = step + 1
            y.append(twoNorm(matrix1, updateValue(q_matrix)))
        if len(sumx)==0:
            sumx=x
            sumy=np.array(y)
        else:
            sumx+=np.array(x)
            sumy+=np.array(y)
    plt.plot(sumx/50, sumy/50, label=label, color=color)


if __name__ == "__main__":
    # # question 1
    # iteration, q_matrix = Q_itetation(0.9, state)
    # print('number of iteration for gamma=0.9 is : %d' % iteration)
    # value=updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    # # question 2
    # # gamma=0
    # iteration, q_matrix = Q_itetation(0, state)
    # print('number of iteration for gamma=0 is : %d' % iteration)
    # value=updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    # # gamma=0.1
    # iteration, q_matrix = Q_itetation(0.1, state)
    # print('number of iteration for gamma=0.1 is : %d' % iteration)
    # value=updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    # # gamma=0.5
    # iteration, q_matrix = Q_itetation(0.5, state)
    # print('number of iteration for gamma=0.5 is : %d' % iteration)
    # value=updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    # # gamma=1
    # iteration, q_matrix = Q_itetation(1, state)
    # print('number of iteration for gamma=1 is : %d' % iteration)
    # value=updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    # # question 3
    # iteration, q_matrix = Q_learning(0.8,0.9,0.8)
    # print('Q_learning - number of iteration for gamma=0.9 is : %d' % iteration)
    # value=updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    # # l-2 norm graphs
    # differencePlot(1, 0.9, 0.8, state,'c')
    # differencePlot(0.8, 0.9, 0.8, state,'r')
    # differencePlot(0.6, 0.9, 0.8, state,'b')
    # differencePlot(0.4, 0.9, 0.8, state,'y')
    # differencePlot(0.2, 0.9, 0.8, state,'k')
    # plt.xlabel('number of interactions')
    # plt.ylabel('l-2 difference')
    # plt.legend()
    # plt.show()
    # differencePlot(0.8, 0.9, 1, state, 'r')
    # differencePlot(0.8, 0.9, 0.8, state, 'c')
    # differencePlot(0.8, 0.9, 0.6, state, 'k')
    # differencePlot(0.8, 0.9, 0.4, state, 'b')
    # differencePlot(0.8, 0.9, 0.2, state, 'y')
    # plt.xlabel('number of interactions')
    # plt.ylabel('l-2 difference')
    # plt.legend()
    # plt.show()
    #
    #
    # iteration, q_matrix = Q_learning(0.5, 0.8, 0.5)
    # print('Q_learning - number of iteration for gamma=0.8 is : %d' % iteration)
    # value = updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    #
    # differencePlot(0.5, 0.9, 1, state, 'r')
    # differencePlot(0.5, 0.9, 1, state, 'k')
    # plt.xlabel('number of interactions')
    # plt.ylabel('l-2 difference')
    # plt.legend()
    # plt.show()
    #
    # iteration, q_matrix = priorKnowledge(0.8, 0.9, 0.8)
    # print('Q_learning - number of iteration for gamma=0.8 is : %d' % iteration)
    # value = updateValue(q_matrix)
    # policy(q_matrix)
    # heatmap(value)
    #
    # differencePlot2(0.5, 0.9, 0.5, state, 'b')
    # differencePlot(0.5, 0.9, 0.5, state, 'y')
    # plt.xlabel('number of interactions')
    # plt.ylabel('l-2 difference')
    # plt.legend()
    # plt.show()

    differencePlot(0.5, 0.9, 0.5, state, 'b')
    differencePlot2(0.5, 0.9, 0.5, state, 'r')
    differencePlot3(0.5, 0.9, 0.5, state, 'k')
    differencePlot4(0.5, 0.9, 0.5, state, 'y')
    plt.xlabel('number of interactions')
    plt.ylabel('l-2 difference')
    plt.legend()
    plt.show()
