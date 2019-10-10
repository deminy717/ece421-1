# import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf

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

# def buildGraph(loss="MSE"):
#     W = tf.Variable(tf.truncated_normal(shape=[28 * 28, 1], stddev=0.5, name="weights"))
#     b = tf.Variable(0.0, name="biases")
#     x = tf.placeholder(tf.float32, shape=(3500, 28 * 28))
#     y = tf.placeholder(tf.float32, shape=(3500, 1))
#     reg = tf.placeholder(tf.float32, shape=(1))
#     tf.set_random_seed(421)

#     y_pred = tf.matmul(x, W) + b

#     if loss == "MSE":
#         train_loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred) + reg * tf.nn.l2_loss(W)
#     elif loss == "CE":
#         train_loss = tf.losses.sigmoid_cross_entropy(labels=y, predictions=y_pred) + reg * tf.nn.l2_loss(W)

#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
#     train = optimizer.minimize(loss=train_loss)
       
#     return W, b, y_pred, y, train_loss, train

def buildGraph(loss="MSE", beta1=0.9, beta2=0.999, epsilon=1e-08):
    W = tf.Variable(tf.truncated_normal(shape=[28 * 28, 1], stddev=0.5, name="weights"))
    b = tf.Variable(0.0, name="biases")
    x = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    y = tf.placeholder(tf.float32, shape=[None, 1])
    validData = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    validTarget = tf.placeholder(tf.float32, shape=[None, 1])
    testData = tf.placeholder(tf.float32, shape=[None, 28 * 28])
    testTarget = tf.placeholder(tf.float32, shape=[None, 1])
    reg = tf.placeholder(tf.float32)
    tf.set_random_seed(421)

    if loss == "MSE":
        y_pred = tf.matmul(x, W) + b
        valid_pred = tf.matmul(validData, W) + b
        test_pred = tf.matmul(testData, W) + b
        train_loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred) + \
        reg * tf.nn.l2_loss(W)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=beta1, \
                                           beta2=beta2, epsilon=epsilon)
        train = optimizer.minimize(loss=train_loss)

        valid_loss = tf.losses.mean_squared_error(labels=validTarget, \
                                  redictions=valid_pred) + reg * tf.nn.l2_loss(W)
        test_loss = tf.losses.mean_squared_error(labels=testTarget, \
                                  predictions=test_pred) + reg * tf.nn.l2_loss(W)
    elif loss == "CE":
        train_logits = tf.matmul(x, W) + b
        valid_logits = tf.matmul(validData, W) + b
        test_logits = tf.matmul(testData, W) + b

        train_loss = tf.losses.sigmoid_cross_entropy(y, train_logits) + \
                                                        reg * tf.nn.l2_loss(W)

        optimizer = tf.train.AdamOptimizer(learning_rate=0.001, \
                                           beta1=beta1, beta2=beta2, epsilon=epsilon)
        train = optimizer.minimize(loss=train_loss)

        valid_loss = tf.losses.sigmoid_cross_entropy(validTarget, valid_logits) + \
                                                        reg * tf.nn.l2_loss(W)
        test_loss = tf.losses.sigmoid_cross_entropy(testTarget, test_logits) + \
                                                        reg * tf.nn.l2_loss(W)

        y_pred = tf.sigmoid(train_logits)
        valid_pred = tf.sigmoid(valid_logits)
        test_pred = tf.sigmoid(test_logits)

    return W, b, x, y_pred, y, validData, validTarget, testData, testTarget, reg, \
            train_loss, valid_loss, test_loss, train, valid_pred, test_pred

def getAccuracy(predict, target):
    return np.sum((predict >= 0.5) == target) / target.shape[0]

def SGD(loss_type="MSE", batch_size=500, beta1=0.9, beta2=0.999, epsilon=1e-08):
    g1 = tf.Graph()
    with g1.as_default():
        tf.set_random_seed(421)
        W, b, x, y_pred, y, validData, validTarget, testData, testTarget, reg, train_loss, valid_loss, test_loss, train, valid_pred,\
             test_pred = buildGraph(loss_type, beta1, beta2, epsilon)
        reg_param = 0.0
        train_data, valid_data, test_data, train_target, valid_target, test_target = loadData()
        train_data = train_data.reshape(-1, train_data.shape[1] * train_data.shape[2])
        valid_data = valid_data.reshape(-1, valid_data.shape[1] * valid_data.shape[2])
        test_data = test_data.reshape(-1, test_data.shape[1] * test_data.shape[2])
        batch_number = int(train_data.shape[0] / batch_size)

        trainLossHistory = []
        validLossHistory = []
        testLossHistory = []
        trainAccuracyHistory = []
        validAccuracyHistory = []
        testAccuracyHistory = []
        
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for _ in range(700):
                p = np.random.permutation(train_data.shape[0])
                train_data = train_data[p]
                train_target = train_target[p]
                for batch_index in range(batch_number):
                    sess.run([train], feed_dict={x: train_data[batch_index * batch_size:(batch_index + 1) * batch_size], \
                        y: train_target[batch_index * batch_size:(batch_index + 1) * batch_size], reg: reg_param})

                trainLoss, validLoss, testLoss, trainPred, validPred, testPred = sess.run([train_loss, valid_loss, test_loss, y_pred, \
                    valid_pred, test_pred], feed_dict={x: train_data, y: train_target, validData:valid_data, validTarget:valid_target, \
                        testData:test_data, testTarget:test_target, reg: reg_param})
                trainLossHistory.append(trainLoss)
                validLossHistory.append(validLoss)
                testLossHistory.append(testLoss)

                trainAccuracyHistory.append(getAccuracy(trainPred, train_target))
                validAccuracyHistory.append(getAccuracy(validPred, valid_target))
                testAccuracyHistory.append(getAccuracy(testPred, test_target))

    return trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory

def gradient_descent():
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

if __name__ == "__main__":
    tf.disable_v2_behavior()
    n = range(700)

    # 3.2
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE")

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("SGD (alpha = 0.001, lambda = 0, batch_size = 500)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    # #3.3
    #100
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", batch_size=100)

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("SGD Batchsize Investigation (alpha = 0.001, lambda = 0, batch_size = 100)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    #700
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", batch_size=700)

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("SGD Batchsize Investigation (alpha = 0.001, lambda = 0, batch_size = 700)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    #1750
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", batch_size=1750)

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("SGD Batchsize Investigation (alpha = 0.001, lambda = 0, batch_size = 1750)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    plt.show()

    #3.4
    fig, axs = plt.subplots(2, 3, figsize=(10,6), dpi = 100, constrained_layout=True)
    fig.suptitle("SGD Hyperparameter Investigation")
    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", beta1=0.95)
    axs[0, 0].set_title("beta1 = 0.95")
    axs[0, 0].plot(n, trainAccuracyHistory)
    axs[0, 0].plot(n, validAccuracyHistory)
    axs[0, 0].plot(n, testAccuracyHistory)
    axs[0, 0].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[0, 0].set_xlabel('iterations')
    axs[0, 0].set_ylabel('accuracy')
    print("beta1=0.95")
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", beta1=0.99)
    axs[1, 0].set_title("beta1 = 0.99")
    axs[1, 0].plot(n, trainAccuracyHistory)
    axs[1, 0].plot(n, validAccuracyHistory)
    axs[1, 0].plot(n, testAccuracyHistory)
    axs[1, 0].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1, 0].set_xlabel('iterations')
    axs[1, 0].set_ylabel('accuracy')
    print('beta1=0.99')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])
    
    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", beta2=0.99)
    axs[0, 1].set_title("beta2 = 0.99")
    axs[0, 1].plot(n, trainAccuracyHistory)
    axs[0, 1].plot(n, validAccuracyHistory)
    axs[0, 1].plot(n, testAccuracyHistory)
    axs[0, 1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[0, 1].set_xlabel('iterations')
    axs[0, 1].set_ylabel('accuracy')
    print('beta2=0.99')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", beta2=0.9999)
    axs[1, 1].set_title("beta2 = 0.9999")
    axs[1, 1].plot(n, trainAccuracyHistory)
    axs[1, 1].plot(n, validAccuracyHistory)
    axs[1, 1].plot(n, testAccuracyHistory)
    axs[1, 1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1, 1].set_xlabel('iterations')
    axs[1, 1].set_ylabel('accuracy')
    print('beta2=0.9999')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", epsilon=1e-9)
    axs[0, 2].set_title("epsilon = 1e-9")
    axs[0, 2].plot(n, trainAccuracyHistory)
    axs[0, 2].plot(n, validAccuracyHistory)
    axs[0, 2].plot(n, testAccuracyHistory)
    axs[0, 2].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[0, 2].set_xlabel('iterations')
    axs[0, 2].set_ylabel('accuracy')
    print('epsilon=1e-9')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "MSE", epsilon=1e-4)
    axs[1, 2].set_title("epsilon = 1e-4")
    axs[1, 2].plot(n, trainAccuracyHistory)
    axs[1, 2].plot(n, validAccuracyHistory)
    axs[1, 2].plot(n, testAccuracyHistory)
    axs[1, 2].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1, 2].set_xlabel('iterations')
    axs[1, 2].set_ylabel('accuracy')
    print('epsilon=1e-4')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])


    #3.5
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE")

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("SGD with CE (alpha = 0.001, lambda = 0, batch_size = 500)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    # #3.3
    #100
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", batch_size=100)

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("CE Batchsize Investigation (alpha = 0.001, lambda = 0, batch_size = 100)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    #700
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", batch_size=700)

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("CE Batchsize Investigation (alpha = 0.001, lambda = 0, batch_size = 700)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    #1750
    trainLossHistory, validLossHistory, testLossHistory, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", batch_size=1750)

    fig, axs = plt.subplots(1, 2, figsize=(8,4), dpi = 80, constrained_layout=True)

    fig.suptitle("CE Batchsize Investigation (alpha = 0.001, lambda = 0, batch_size = 1750)")
    axs[0].set_title("loss history")
    axs[0].plot(n, trainLossHistory)
    axs[0].plot(n, validLossHistory)
    axs[0].plot(n, testLossHistory)
    axs[0].legend(['train loss', 'valid loss', 'test loss'])
    axs[0].set_xlabel('iterations')
    axs[0].set_ylabel('loss')
  
    axs[1].set_title("accuracy history")
    axs[1].plot(n, trainAccuracyHistory)
    axs[1].plot(n, validAccuracyHistory)
    axs[1].plot(n, testAccuracyHistory)
    axs[1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1].set_xlabel('iterations')
    axs[1].set_ylabel('accuracy')

    plt.show()

    #3.4
    fig, axs = plt.subplots(2, 3, figsize=(10,6), dpi = 100, constrained_layout=True)
    fig.suptitle("CE Hyperparameter Investigation")
    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", beta1=0.95)
    axs[0, 0].set_title("beta1 = 0.95")
    axs[0, 0].plot(n, trainAccuracyHistory)
    axs[0, 0].plot(n, validAccuracyHistory)
    axs[0, 0].plot(n, testAccuracyHistory)
    axs[0, 0].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[0, 0].set_xlabel('iterations')
    axs[0, 0].set_ylabel('accuracy')
    print("beta1=0.95")
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", beta1=0.99)
    axs[1, 0].set_title("beta1 = 0.99")
    axs[1, 0].plot(n, trainAccuracyHistory)
    axs[1, 0].plot(n, validAccuracyHistory)
    axs[1, 0].plot(n, testAccuracyHistory)
    axs[1, 0].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1, 0].set_xlabel('iterations')
    axs[1, 0].set_ylabel('accuracy')
    print('beta1=0.99')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])
    
    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", beta2=0.99)
    axs[0, 1].set_title("beta2 = 0.99")
    axs[0, 1].plot(n, trainAccuracyHistory)
    axs[0, 1].plot(n, validAccuracyHistory)
    axs[0, 1].plot(n, testAccuracyHistory)
    axs[0, 1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[0, 1].set_xlabel('iterations')
    axs[0, 1].set_ylabel('accuracy')
    print('beta2=0.99')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", beta2=0.9999)
    axs[1, 1].set_title("beta2 = 0.9999")
    axs[1, 1].plot(n, trainAccuracyHistory)
    axs[1, 1].plot(n, validAccuracyHistory)
    axs[1, 1].plot(n, testAccuracyHistory)
    axs[1, 1].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1, 1].set_xlabel('iterations')
    axs[1, 1].set_ylabel('accuracy')
    print('beta2=0.9999')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", epsilon=1e-9)
    axs[0, 2].set_title("epsilon = 1e-9")
    axs[0, 2].plot(n, trainAccuracyHistory)
    axs[0, 2].plot(n, validAccuracyHistory)
    axs[0, 2].plot(n, testAccuracyHistory)
    axs[0, 2].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[0, 2].set_xlabel('iterations')
    axs[0, 2].set_ylabel('accuracy')
    print('epsilon=1e-9')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])

    _, _, _, trainAccuracyHistory, validAccuracyHistory, testAccuracyHistory = SGD(loss_type = "CE", epsilon=1e-4)
    axs[1, 2].set_title("epsilon = 1e-4")
    axs[1, 2].plot(n, trainAccuracyHistory)
    axs[1, 2].plot(n, validAccuracyHistory)
    axs[1, 2].plot(n, testAccuracyHistory)
    axs[1, 2].legend(['train accuracy', 'valid accuracy', 'test accuracy'])
    axs[1, 2].set_xlabel('iterations')
    axs[1, 2].set_ylabel('accuracy')
    print('epsilon=1e-4')
    print("train accuracy =", trainAccuracyHistory[-1])
    print("valid accuracy =", validAccuracyHistory[-1])
    print("test accuracy =", testAccuracyHistory[-1])


    plt.show()

