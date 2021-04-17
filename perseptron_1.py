import numpy as np


def perceptron(X, W, theta, result):
    for i in X:
        z = np.dot(W.T, i)
        if z >= theta:
            print("z: %.3f, result: %d" % (z, 1))
            result.append(1)
        else:
            print("z: %.3f, result: %d" % (z, 0))
            result.append(0)
    return result


def update_rule(X, y, prediction, learning_rate, W):
    index = 0
    for label, pred in zip(y, prediction):
        error = label - pred
        update_value = (learning_rate*error)*X[index].reshape(3, 1)
        W += update_value
        index += 1
    return W


# dataset
X = np.random.rand(10, 3)
y = np.random.randint(0, 2, size=(10, 1))

# weight vector
W = np.random.uniform(low=0, high=1, size=(3, 1))

# theta, threshold
theta = 0.5
result = []

prediction = perceptron(X, W, theta, result)

# list => numpy arrayの変換
prediction = np.array(prediction).reshape(len(prediction), 1)

# 学習率
learning_rate = 0.02

print("更新前の重み\n", W)

# 重みの更新
W = update_rule(X, y, prediction, learning_rate, W)
print("重みを一回だけ更新後\n", W)
