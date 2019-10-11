import tensorflow as tf
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
    N = np.shape(x)[0]
    mse = np.sum(np.square(np.dot(x,W)+b-y))
    weight_loss = reg/2 * np.sum((np.square(W)))
    total_loss = (mse/N + weight_loss)
    return total_loss

def grad_MSE(W, b, x, y, reg):
    N = np.shape(x)[0]
    error = np.dot(x,W) + b - y 
    gradMSE_W = 2/N * np.transpose(np.dot(np.transpose(error),x)) + reg*W
    gradMSE_b = 2/N * np.sum(error)
    return gradMSE_W, gradMSE_b

def grad_descent(W, b, x, y, alpha, epochs, reg, error_tol, lossType, validData, validTarget, testData, testTarget):   
    
    accuracyTrain = [accuracy(W, b, x, y, lossType)]
    accuracyValid = [accuracy(W, b, validData, validTarget,lossType)]
    accuracyTest = [accuracy(W, b, testData, testTarget,lossType)]
    train_loss = [MSE(W, b, x, y, reg)]
    valid_loss = [MSE(W, b, validData, validTarget, reg)]
    test_loss = [MSE(W, b, testData, testTarget, reg)]
    
    W_old = W
    b_old = b
    
    if lossType == "MSE":
        for i in range(epochs):
            gradMSE_W, gradMSE_b = grad_MSE(W_old, b_old, x, y, reg)
            W_new = W_old - alpha*gradMSE_W
            b_new = b_old - alpha*gradMSE_b

            train_loss.append(MSE(W_new, b_new, x, y, reg))
            valid_loss.append(MSE(W_new, b_new, validData, validTarget, reg))
            test_loss.append(MSE(W_new, b_new, testData, testTarget, reg))

            accuracyTrain.append(accuracy(W_new, b_new, x, y,lossType))
            accuracyValid.append(accuracy(W_new, b_new, validData, validTarget,lossType))
            accuracyTest.append(accuracy(W_new, b_new, testData, testTarget,lossType))
            
            if np.linalg.norm(W_new - W_old) < error_tol:
                break
            else:
                W_old = W_new
                b_old = b_new
                
    if lossType == "CE":
        train_loss = [crossEntropyLoss(W, b, x, y, reg)]
        valid_loss = [crossEntropyLoss(W, b, validData, validTarget, reg)]
        test_loss = [crossEntropyLoss(W, b, testData, testTarget, reg)]

        for i in range(epochs):
            gradCE_W, gradCE_b = grad_CE(W_old, b_old, x, y, reg)
            W_new = W_old - alpha*gradCE_W
            b_new = b_old - alpha*gradCE_b

            train_loss.append(crossEntropyLoss(W_new, b_new, x, y, reg))
            valid_loss.append(crossEntropyLoss(W_new, b_new, validData, validTarget, reg))
            test_loss.append(crossEntropyLoss(W_new, b_new, testData, testTarget, reg))

            accuracyTrain.append(accuracy(W_new, b_new, x, y,lossType))
            accuracyValid.append(accuracy(W_new, b_new, validData, validTarget,lossType))
            accuracyTest.append(accuracy(W_new, b_new, testData, testTarget,lossType))
            
            if np.linalg.norm(W_new - W_old) < error_tol:
                break
            else:
                W_old = W_new
                b_old = b_new
        
    return W_new, b_new, train_loss, valid_loss, test_loss, accuracyTrain, accuracyValid, accuracyTest


def crossEntropyLoss(W, b, x, y, reg):
    N = np.shape(x)[0]
     
    WeightDecay_loss = reg/2 * np.sum((np.square(W)))
    
    y_hat = 1/(1+np.exp(-(np.dot(x,W)+b)))
    CE_loss = np.sum(-(y*np.log(y_hat)) - ((1-y)*np.log(1-y_hat)))
    
    total_loss = (CE_loss/N + WeightDecay_loss)
    return total_loss

def grad_CE(W, b, x, y, reg):
    
    N = np.shape(x)[0]
    e_pos = np.exp(np.dot(x,W)+b)
    e_neg = np.exp(-1*(np.dot(x,W)+b))
    y_hat= 1/(1+e_neg)
    
    gradCE_W = 1/N * (np.dot(np.transpose(-1*x),y*y_hat*e_neg) + np.dot(np.transpose(x),(1-y)*(1-y_hat)*e_pos)) + reg*W
    gradCE_b = 1/N * sum(y*y_hat*e_neg*(-1) + (1-y)*(1-y_hat)*e_pos)
    
    return gradCE_W, gradCE_b


def reshapeData (trainData, validData, testData):
    trainData = trainData.reshape(np.shape(trainData)[0], np.shape(trainData)[1]*np.shape(trainData)[2])
    validData = validData.reshape(np.shape(validData)[0], np.shape(validData)[1]*np.shape(validData)[2])
    testData = testData.reshape(np.shape(testData)[0], np.shape(testData)[1]*np.shape(testData)[2])
    return trainData, validData, testData

def accuracy (W, b, x, y, lossType):
    N = np.shape(y)[0]
    
    if lossType == "MSE": 
        y_hat = np.dot(x,W)+b
    if lossType == "CE":
        y_hat = 1/(1+np.exp(-(np.dot(x,W)+b)))
    
    threshold = 0.5
    y_hat[y_hat >= threshold] = 1
    y_hat[y_hat < threshold] = 0
    
    error = np.sum(np.square(y-y_hat))
    accuracy = 1 - error/N
    return accuracy

def normal_MSE (trainData, trainTarget, validData, validTarget, testData, testTarget, reg):
    x = trainData
    y = trainTarget
    x0 = np.ones((np.shape(x)[0],1))
    x = np.hstack((x0,x))  
    xt = np.transpose(x)
    W = np.dot(np.linalg.inv(np.dot(xt,x)),np.dot(xt,y))
    b = W[0].item()
    
    x = np.delete(x, 0, 1)
    W = np.delete(W, 0, 0)
    
    train_loss = [MSE(W, b, trainData, trainTarget, reg)]
    valid_loss = [MSE(W, b, validData, validTarget, reg)]
    test_loss = [MSE(W, b, testData, testTarget, reg)]
    
    accuracyTrain = [accuracy(W, b, trainData, trainTarget,"MSE")]
    accuracyValid = [accuracy(W, b, validData, validTarget,"MSE")]
    accuracyTest = [accuracy(W, b, testData, testTarget,"MSE")]

    
    return W, b, train_loss, valid_loss, test_loss, accuracyTrain, accuracyValid, accuracyTest

#compute normal equation
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = reshapeData (trainData, validData, testData)
W,b, train_loss, valid_loss, test_loss, accuracyTrain, accuracyValid, accuracyTest = normal_MSE (trainData, trainTarget, validData, validTarget, testData, testTarget, 0)
print("Normal Equation Loss: ", train_loss, valid_loss, test_loss, "\n"
      "Normal Equation Accuracy: ", accuracyTrain, accuracyValid, accuracyTest)


##Computation for part 1&2
#initialization
trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
trainData, validData, testData = reshapeData (trainData, validData, testData)

W = np.zeros((28*28,1))
b = 0

alpha1 = 0.005
alpha2 = 0.001
alpha3 = 0.0001

epochs = 5000

reg0 = 0
reg1 = 0.001
reg2 = 0.1
reg3 = 0.5

error_tol = 1e-7

t = np.transpose(np.arange(0,epochs+1,1))

#MSE Loss & Accuracy
lossType = "MSE"
trainLoss_alpha1_reg0, validLoss_alpha1_reg0, testLoss_alpha1_reg0, accTrain_alpha1_reg0, accValid_alpha1_reg0, accTest_alpha1_reg0 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg0, error_tol, lossType, validData, validTarget, testData, testTarget)[2:]
trainLoss_alpha2_reg0, validLoss_alpha2_reg0, testLoss_alpha2_reg0, accTrain_alpha2_reg0, accValid_alpha2_reg0, accTest_alpha2_reg0 = grad_descent(W, b, trainData, trainTarget, alpha2, epochs, reg0, error_tol, lossType, validData, validTarget, testData, testTarget)[2:]
trainLoss_alpha3_reg0, validLoss_alpha3_reg0, testLoss_alpha3_reg0, accTrain_alpha3_reg0, accValid_alpha3_reg0, accTest_alpha3_reg0 = grad_descent(W, b, trainData, trainTarget, alpha3, epochs, reg0, error_tol, lossType, validData, validTarget, testData, testTarget)[2:]
trainLoss_alpha1_reg1, validLoss_alpha1_reg1, testLoss_alpha1_reg1, accTrain_alpha1_reg1, accValid_alpha1_reg1, accTest_alpha1_reg1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg1, error_tol, lossType, validData, validTarget, testData, testTarget)[2:]
trainLoss_alpha1_reg2, validLoss_alpha1_reg2, testLoss_alpha1_reg2, accTrain_alpha1_reg2, accValid_alpha1_reg2, accTest_alpha1_reg2 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg2, error_tol, lossType, validData, validTarget, testData, testTarget)[2:]
trainLoss_alpha1_reg3, validLoss_alpha1_reg3, testLoss_alpha1_reg3, accTrain_alpha1_reg3, accValid_alpha1_reg3, accTest_alpha1_reg3 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg3, error_tol, lossType, validData, validTarget, testData, testTarget)[2:]

print("MSE Loss: \n"
      "alpha1_reg0: ", [trainLoss_alpha1_reg0[-1], validLoss_alpha1_reg0[-1], testLoss_alpha1_reg0[-1]], "\n"
      "alpha2_reg0: ", [trainLoss_alpha2_reg0[-1], validLoss_alpha2_reg0[-1], testLoss_alpha2_reg0[-1]], "\n" 
      "alpha3_reg0: ", [trainLoss_alpha3_reg0[-1], validLoss_alpha3_reg0[-1], testLoss_alpha3_reg0[-1]], "\n" 
      "alpha1_reg1: ", [trainLoss_alpha1_reg1[-1], validLoss_alpha1_reg1[-1], testLoss_alpha1_reg1[-1]], "\n" 
      "alpha1_reg2: ", [trainLoss_alpha1_reg2[-1], validLoss_alpha1_reg2[-1], testLoss_alpha1_reg2[-1]], "\n"
      "alpha1_reg3: ", [trainLoss_alpha1_reg3[-1], validLoss_alpha1_reg3[-1], testLoss_alpha1_reg3[-1]], "\n\n"
      
      "MSE Accuracy: \n"
      "alpha1_reg0: ", [accTrain_alpha1_reg0[-1], accValid_alpha1_reg0[-1], accTest_alpha1_reg0[-1]], "\n"
      "alpha2_reg0: ", [accTrain_alpha2_reg0[-1], accValid_alpha2_reg0[-1], accTest_alpha2_reg0[-1]], "\n" 
      "alpha3_reg0: ", [accTrain_alpha3_reg0[-1], accValid_alpha3_reg0[-1], accTest_alpha3_reg0[-1]], "\n" 
      "alpha1_reg1: ", [accTrain_alpha1_reg1[-1], accValid_alpha1_reg1[-1], accTest_alpha1_reg1[-1]], "\n" 
      "alpha1_reg2: ", [accTrain_alpha1_reg2[-1], accValid_alpha1_reg2[-1], accTest_alpha1_reg2[-1]], "\n"
      "alpha1_reg3: ", [accTrain_alpha1_reg3[-1], accValid_alpha1_reg3[-1], accTest_alpha1_reg3[-1]])

#CE Loss & Accuracy
CEtrainLoss_alpha1_reg2, CEvalidLoss_alpha1_reg2, CEtestLoss_alpha1_reg2, CEaccTrain_alpha1_reg2, CEaccValid_alpha1_reg2, CEaccTest_alpha1_reg2 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg2, error_tol, "CE", validData, validTarget, testData, testTarget)[2:]
CEtrainLoss_alpha1_reg0, CEvalidLoss_alpha1_reg0, CEtestLoss_alpha1_reg0, CEaccTrain_alpha1_reg0, CEaccValid_alpha1_reg0, CEaccTest_alpha1_reg0 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg0, error_tol, "CE", validData, validTarget, testData, testTarget)[2:]

print("CE Loss:\n"
      "alpha1_reg2: ", [CEtrainLoss_alpha1_reg2[-1], CEvalidLoss_alpha1_reg2[-1], CEtestLoss_alpha1_reg2[-1]], "\n"
      "alpha1_reg0: ", [CEtrainLoss_alpha1_reg0[-1], CEvalidLoss_alpha1_reg0[-1], CEtestLoss_alpha1_reg0[-1]], "\n\n"
      
      "CE Accuracy:\n"
      "alpha1_reg0: ", [CEaccTrain_alpha1_reg0[-1], CEaccValid_alpha1_reg0[-1], CEaccTest_alpha1_reg0[-1]], "\n"
      "alpha1_reg2: ", [CEaccTrain_alpha1_reg2[-1], CEaccValid_alpha1_reg2[-1], CEaccTest_alpha1_reg2[-1]])

#Part 1.3, MSE Loss

plt.figure(1)
plt.plot(t,trainLoss_alpha1_reg0,label='α = 0.005')
plt.plot(t,trainLoss_alpha2_reg0,label='α = 0.001')
plt.plot(t,trainLoss_alpha3_reg0,label='α = 0.0001')
plt.legend(loc='upper right')
plt.title("Impact of Learning Rate on Training Data")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(2)
plt.plot(t,validLoss_alpha1_reg0,label='α = 0.005')
plt.plot(t,validLoss_alpha2_reg0,label='α = 0.001')
plt.plot(t,validLoss_alpha3_reg0,label='α = 0.0001')
plt.legend(loc='upper right')
plt.title("Impact of Learning Rate on Validation Data")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(3)
plt.plot(t,testLoss_alpha1_reg0,label='α = 0.005')
plt.plot(t,testLoss_alpha2_reg0,label='α = 0.001')
plt.plot(t,testLoss_alpha3_reg0,label='α = 0.0001')
plt.legend(loc='upper right')
plt.title("Impact of Learning Rate on Test Data")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(4)
plt.plot(t,trainLoss_alpha1_reg0,label='Training Loss')
plt.plot(t,validLoss_alpha1_reg0,label='Validation Loss')
plt.plot(t,testLoss_alpha1_reg0,label='Test Loss')
plt.legend(loc='upper right')
plt.title("MSE Loss for Linear Regression (α = 0.005, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(5)
plt.plot(t,trainLoss_alpha2_reg0,label='Training Loss')
plt.plot(t,validLoss_alpha2_reg0,label='Validation Loss')
plt.plot(t,testLoss_alpha2_reg0,label='Test Loss')
plt.legend(loc='upper right')
plt.title("MSE Loss for Linear Regression (α = 0.001, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(6)
plt.plot(t,trainLoss_alpha3_reg0,label='Training Loss')
plt.plot(t,validLoss_alpha3_reg0,label='Validation Loss')
plt.plot(t,testLoss_alpha3_reg0,label='Test Loss')
plt.legend(loc='upper right')
plt.title("MSE Loss for Linear Regression (α = 0.0001, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.show()

#Part 1.4, MSE Loss
plt.figure(1)
plt.plot(t,trainLoss_alpha1_reg1,label='λ = 0.001')
plt.plot(t,trainLoss_alpha1_reg2,label='λ = 0.1')
plt.plot(t,trainLoss_alpha1_reg3,label='λ = 0.5')
plt.legend(loc='upper right')
plt.title("Impact of Regulation Parameter on Training Data")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(2)
plt.plot(t,validLoss_alpha1_reg1,label='λ = 0.001')
plt.plot(t,validLoss_alpha1_reg2,label='λ = 0.1')
plt.plot(t,validLoss_alpha1_reg3,label='λ = 0.5')
plt.legend(loc='upper right')
plt.title("Impact of Regulation Parameter on Validation Data")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(3)
plt.plot(t,testLoss_alpha1_reg1,label='λ = 0.001')
plt.plot(t,testLoss_alpha1_reg2,label='λ = 0.1')
plt.plot(t,testLoss_alpha1_reg3,label='λ = 0.5')
plt.legend(loc='upper right')
plt.title("Impact of Regulation Parameter on Test Data")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(4)
plt.plot(t,trainLoss_alpha1_reg1,label='Training Loss')
plt.plot(t,validLoss_alpha1_reg1,label='Validation Loss')
plt.plot(t,testLoss_alpha1_reg1,label='Test Loss')
plt.legend(loc='upper right')
plt.title("MSE Loss for Linear Regression (α = 0.005, λ = 0.001)")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(5)
plt.plot(t,trainLoss_alpha1_reg2,label='Training Loss')
plt.plot(t,validLoss_alpha1_reg2,label='Validation Loss')
plt.plot(t,testLoss_alpha1_reg2,label='Test Loss')
plt.legend(loc='upper right')
plt.title("MSE Loss for Linear Regression (α = 0.005, λ = 0.1)")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.figure(6)
plt.plot(t,trainLoss_alpha1_reg3,label='Training Loss')
plt.plot(t,validLoss_alpha1_reg3,label='Validation Loss')
plt.plot(t,testLoss_alpha1_reg3,label='Test Loss')
plt.legend(loc='upper right')
plt.title("MSE Loss for Linear Regression (α = 0.005, λ = 0.5)")
plt.xlabel("Iteration")
plt.ylabel("MSE Loss")

plt.show()

#Part 1.3 & 1.4, MSE Accuracy - Impact of α & λ
plt.figure(1)
plt.plot(t,accTrain_alpha1_reg0,label='α = 0.005')
plt.plot(t,accTrain_alpha2_reg0,label='α = 0.001')
plt.plot(t,accTrain_alpha3_reg0,label='α = 0.0001')
plt.legend(loc='lower right')
plt.title("Impact of Learning Rate on Training Data")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(2)
plt.plot(t,accValid_alpha1_reg0,label='α = 0.005')
plt.plot(t,accValid_alpha2_reg0,label='α = 0.001')
plt.plot(t,accValid_alpha3_reg0,label='α = 0.0001')
plt.legend(loc='lower right')
plt.title("Impact of Learning Rate on Validation Data")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(3)
plt.plot(t,accTest_alpha1_reg0,label='α = 0.005')
plt.plot(t,accTest_alpha2_reg0,label='α = 0.001')
plt.plot(t,accTest_alpha3_reg0,label='α = 0.0001')
plt.legend(loc='lower right')
plt.title("Impact of Learning Rate on Test Data")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(4)
plt.plot(t,accTrain_alpha1_reg1,label='λ = 0.001')
plt.plot(t,accTrain_alpha1_reg2,label='λ = 0.1')
plt.plot(t,accTrain_alpha1_reg3,label='λ = 0.5')
plt.legend(loc='lower right')
plt.title("Impact of Regulation Parameter on Training Data")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(5)
plt.plot(t,accValid_alpha1_reg1,label='λ = 0.001')
plt.plot(t,accValid_alpha1_reg2,label='λ = 0.1')
plt.plot(t,accValid_alpha1_reg3,label='λ = 0.5')
plt.legend(loc='lower right')
plt.title("Impact of Regulation Parameter on Validation Data")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(6)
plt.plot(t,accTest_alpha1_reg1,label='λ = 0.001')
plt.plot(t,accTest_alpha1_reg2,label='λ = 0.1')
plt.plot(t,accTest_alpha1_reg3,label='λ = 0.5')
plt.legend(loc='lower right')
plt.title("Impact of Regulation Parameter on Test Data")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.show()

#Part 1.3 & 1.4, MSE Accuracy for three datasets with changing of α & λ
plt.figure(1)
plt.plot(t,accTrain_alpha1_reg0,label='Training Accuracy')
plt.plot(t,accValid_alpha1_reg0,label='Validation Accuracy')
plt.plot(t,accTest_alpha1_reg0,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Linear Regression (α = 0.005, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(2)
plt.plot(t,accTrain_alpha2_reg0,label='Training Accuracy')
plt.plot(t,accValid_alpha2_reg0,label='Validation Accuracy')
plt.plot(t,accTest_alpha2_reg0,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Linear Regression (α = 0.001, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(3)
plt.plot(t,accTrain_alpha3_reg0,label='Training Accuracy')
plt.plot(t,accValid_alpha3_reg0,label='Validation Accuracy')
plt.plot(t,accTest_alpha3_reg0,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Linear Regression (α = 0.0001, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(4)
plt.plot(t,accTrain_alpha1_reg1,label='Training Accuracy')
plt.plot(t,accValid_alpha1_reg1,label='Validation Accuracy')
plt.plot(t,accTest_alpha1_reg1,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Linear Regression (α = 0.005, λ = 0.001)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(5)
plt.plot(t,accTrain_alpha1_reg2,label='Training Accuracy')
plt.plot(t,accValid_alpha1_reg2,label='Validation Accuracy')
plt.plot(t,accTest_alpha1_reg2,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Linear Regression (α = 0.005, λ = 0.1)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(6)
plt.plot(t,accTrain_alpha1_reg3,label='Training Accuracy')
plt.plot(t,accValid_alpha1_reg3,label='Validation Accuracy')
plt.plot(t,accTest_alpha1_reg3,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Linear Regression (α = 0.005, λ = 0.5)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.show()


#Part 2.2 & 2.3, CE Loss for three datasets
plt.figure(1)
plt.plot(t,CEtrainLoss_alpha1_reg2,label='Training Loss')
plt.plot(t,CEvalidLoss_alpha1_reg2,label='Validation Loss')
plt.plot(t,CEtestLoss_alpha1_reg2,label='Test Loss')
plt.legend(loc='upper right')
plt.title("CE Loss for Logistic Regression (α = 0.005, λ = 0.1)")
plt.xlabel("Iteration")
plt.ylabel("CE Loss")

plt.figure(2)
plt.plot(t,CEtrainLoss_alpha1_reg2,label='Training Loss')
plt.plot(t,CEvalidLoss_alpha1_reg2,label='Validation Loss')
plt.plot(t,CEtestLoss_alpha1_reg2,label='Test Loss')
plt.legend(loc='upper right')
plt.title("CE Loss for Logistic Regression (α = 0.005, λ = 0.1)")
plt.xlabel("Iteration")
plt.ylabel("CE Loss")

plt.show()

#Part 2.2 & 2.3, CE Accuracy for three datasets
plt.figure(1)
plt.plot(t,CEaccTrain_alpha1_reg0,label='Training Accuracy')
plt.plot(t,CEaccValid_alpha1_reg0,label='Validation Accuracy')
plt.plot(t,CEaccTest_alpha1_reg0,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Logistic Regression (α = 0.005, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")

plt.figure(2)
plt.plot(t,CEaccTrain_alpha1_reg2,label='Training Accuracy')
plt.plot(t,CEaccValid_alpha1_reg2,label='Validation Accuracy')
plt.plot(t,CEaccTest_alpha1_reg2,label='Test Accuracy')
plt.legend(loc='lower right')
plt.title("Accuracy of Logistic Regression (α = 0.005, λ = 0.1)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy %")


plt.show()


#Part 2.3, Comparison of Loss and Accuracy
plt.figure(1)
plt.plot(t,CEaccTrain_alpha1_reg0,label='Logistic Regression')
plt.plot(t,accTrain_alpha1_reg0,label='Linear Regression')
plt.legend(loc='lower right')
plt.title("Comparison of Accuracy between \n Logistic Regression and Linear Regression (α = 0.005, λ = 0)")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.show()

plt.figure(2)
plt.plot(t,CEtrainLoss_alpha1_reg0,label='CE Loss')
plt.plot(t,trainLoss_alpha1_reg0,label='MSE Loss')
plt.legend(loc='upper right')
plt.title("Comparison between CE Loss and MSE Loss with α = 0.005, λ = 0")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()



#y_hat distribution
W_CE, b_CE = grad_descent(W, b, trainData, trainTarget, 0.005, epochs, 0, error_tol, "CE", validData, validTarget, testData, testTarget)[:2]
W_MSE, B_MSE = grad_descent(W, b, trainData, trainTarget, 0.005, epochs, 0, error_tol, "MSE", validData, validTarget, testData, testTarget)[:2]

t_dis = np.transpose(np.arange(0,3500,1))
y_hat_CE = 1/(1+np.exp(-(np.dot(trainData,W_CE)+b_CE)))
y_hat_MSE = np.dot(trainData,W_MSE) + B_MSE

plt.figure(1)
plt.plot(t_dis,y_hat_CE,"o")
plt.axis([0, 3500, -0.5, 1.5])
plt.title("Distribution of Predicted Outputs for Logistic Regression")

plt.figure(2)
plt.plot(t_dis,y_hat_MSE,"o")
plt.axis([0, 3500, -0.5, 1.5])
plt.title("Distribution of Predicted Outputs for Linear Regression")

plt.show()


#visulization
w_train_1 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg0, error_tol, "MSE", validData, validTarget, testData, testTarget)[0]
w_train_2 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg1, error_tol, "MSE", validData, validTarget, testData, testTarget)[0]
w_train_3 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg2, error_tol, "MSE", validData, validTarget, testData, testTarget)[0]
w_train_4 = grad_descent(W, b, trainData, trainTarget, alpha1, epochs, reg3, error_tol, "MSE", validData, validTarget, testData, testTarget)[0]



plt.figure(1)
plt.matshow(w_train_1.reshape(28, 28))
plt.title("Weight Matrix with α = 0.005, λ = 0")

plt.figure(2)
plt.matshow(w_train_2.reshape(28, 28))
plt.title("Weight Matrix with α = 0.005, λ = 0.001")

plt.figure(3)
plt.matshow(w_train_3.reshape(28, 28))
plt.title("Weight Matrix with α = 0.005, λ = 0.1")

plt.figure(4)
plt.matshow(w_train_4.reshape(28, 28))
plt.title("Weight Matrix with α = 0.005, λ = 0.5")

plt.show()



#CE vs MSE convergence rate

W0 = np.zeros((28*28,1))
b0 = 0
alpha1 = 0.005
epochs = 5000
reg0 = 0
error_tol = 1e-7

t = np.transpose(np.arange(0,epochs+1,1))

W_MSE,b_MSE = grad_descent(W0, b0, trainData, trainTarget, alpha1, epochs, reg0, error_tol, "MSE", validData, validTarget, testData, testTarget)[2:5]
trainLoss_MSE, validLoss_MSE, testLoss_MSE = grad_descent(W_MSE,b_MSE, trainData, trainTarget, alpha1, epochs, reg0, error_tol, "MSE", validData, validTarget, testData, testTarget)[2:5]

W_CE,b_CE = grad_descent(W0, b0, trainData, trainTarget, alpha1, epochs, reg0, error_tol, "CE", validData, validTarget, testData, testTarget)[0:2]
trainLoss_CE, validLoss_CE, testLoss_CE = grad_descent(W_CE, b_CE, trainData, trainTarget, alpha1, epochs, reg0, error_tol, "CE", validData, validTarget, testData, testTarget)[2:5]

plt.plot(t,trainLoss_CE,label='CE Loss')
plt.plot(t,trainLoss_MSE,label='CE Loss')
plt.legend(loc='upper right')
plt.show()




###Part 3
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

