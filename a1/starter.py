#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as data :
        Data, Target = data ['images'], data['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget

def MSE(W, b, x, y, reg):
    N = np.shape(y)[0]
    e = np.dot(x, W) + b - y
    return np.sum(e ** 2) / N + (reg / 2) * np.sum(W ** 2)

def gradMSE(W, b, x, y, reg):
    N = np.shape(y)[0]
    e = np.dot(x, W) + b - y
    
    grad_b = (2 / N) * np.sum(e)
    grad_W = (2 / N) * np.dot(e, x) + reg * W

    return grad_W, grad_b

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, validData, validTarget, testData, testTarget):
    trainLossHistory = [MSE(W, b, x, y, reg)]
    validLossHistory = [MSE(W, b, validData, validTarget, reg)]
    testLossHistory = [MSE(W, b, testData, testTarget, reg)]

    for i in range(epochs):
        grad_W, grad_b = gradMSE(W, b, x, y, reg)

        dist_W, W, b = getNewData(W, alpha, grad_W, b, grad_b)
        
        # print(i)
        trainLossHistory.append(MSE(W, b, x, y, reg))
        validLossHistory.append(MSE(W, b, validData, validTarget, reg))
        testLossHistory.append(MSE(W, b, testData, testTarget, reg))
        
        if dist_W < error_tol:
            break

    return W, b, trainLossHistory, validLossHistory, testLossHistory

def getNewData(W, alpha, grad_W, b, grad_b):
    new_W = W - alpha * grad_W
    dist_W = np.linalg.norm(new_W - W)

    return dist_W, new_W, b - alpha * grad_b

# def crossEntropyLoss(W, b, x, y, reg):
#     # Your implementation here

# def gradCE(W, b, x, y, reg):
#     # Your implementation here

def buildGraph(loss, alpha):

    tf.set_random_seed(421)

    if loss == "MSE":
        W, b, trainLossHistory, validLossHistory, testLossHistory = grad_descent(W, b, x, y, alpha, 5000, 0, 1e-7, validData, validTarget, testData, testTarget)
        return W, b, trainLossHistory, validLossHistory, testLossHistory
        
if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    x = trainData.reshape(-1, trainData.shape[1] * trainData.shape[2])
    y = trainTarget.flatten()
    W = np.zeros(x.shape[1])
    b = 0

    validData = validData.reshape(-1, validData.shape[1] * validData.shape[2])
    validTarget = validTarget.flatten()
    testData = testData.reshape(-1, testData.shape[1] * testData.shape[2])
    testTarget = testTarget.flatten()

    W, b, trainLossHistory, validLossHistory, testLossHistory = grad_descent(W, b, x, y, 0.001, 5000, 0, 1e-7, validData, validTarget, testData, testTarget)

    print(trainLossHistory)
    print(validLossHistory)
    print(testLossHistory)
    n = range(len(trainLossHistory))
    plt.plot(n, trainLossHistory)
    plt.plot(n, validLossHistory)
    plt.plot(n, testLossHistory)
    plt.show()