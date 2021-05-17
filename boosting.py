# from numpy import *  
  
# def loadSimpleData():  
#     datMat = mat([[1., 2.1],  
#                   [2., 1.1],  
#                   [1.3, 1.],  
#                   [1., 1.],  
#                   [2., 1.]])  
#     classLabels = mat([1.0, 1.0, -1.0, -1.0, 1.0])  
#     return datMat, classLabels  
  
# def singleStumpClassipy(dataMat, dim, threshold, thresholdIneq):  
#     classMat = ones((shape(dataMat)[0], 1))  
#     #根据thresholdIneq划分出不同的类，在'-1'和'1'之间切换  
#     if thresholdIneq == 'left':#在threshold左侧的为'-1'  
#         classMat[dataMat[:, dim] <= threshold] = -1.0  
#     else:  
#         classMat[dataMat[:, dim] > threshold] = -1.0  
      
#     return classMat  
  
# def singleStump(dataArr, classLabels, D):  
#     dataMat = mat(dataArr)  
#     labelMat = mat(classLabels).T  
#     m, n = shape(dataMat)  
#     numSteps = 10.0  
#     bestStump = {}  
#     bestClasEst = zeros((m, 1))  
#     minError = inf  
#     for i in range(n):#对每一个特征  
#         #取第i列特征的最小值和最大值，以确定步长  
#         rangeMin = dataMat[:, i].min()  
#         rangeMax = dataMat[:, i].max()  
#         stepSize = (rangeMax - rangeMin) / numSteps  
#         for j in range(-1, int(numSteps) + 1):  
#             #不确定是哪个属于类'-1'，哪个属于类'1'，分两种情况  
#             for inequal in ['left', 'right']:  
#                 threshold = rangeMin + j * stepSize#得到每个划分的阈值  
#                 predictionClass = singleStumpClassipy(dataMat, i, threshold, inequal)  
#                 errorMat = ones((m, 1))  
#                 errorMat[predictionClass == labelMat] = 0  
#                 weightedError = D.T * errorMat#D是每个样本的权重  
#                 if weightedError < minError:  
#                     minError = weightedError  
#                     bestClasEst = predictionClass.copy()  
#                     bestStump['dim'] = i  
#                     bestStump['threshold'] = threshold  
#                     bestStump['inequal'] = inequal  
      
#     return bestStump, minError, bestClasEst  
  
# def adaBoostTrain(dataArr, classLabels, G):  
#     weakClassArr = []  
#     m = shape(dataArr)[0]#样本个数 ，row 
#     m, n = shape(dataArr)  

#     ### Numpy 之 Shape  ###

# # 建立一个4×2的矩阵c
# # >>> c = array([[1,1],[1,2],[1,3],[1,4]])  
# # >>> c.shape  
# # (4, 2)  
# # >>> c.shape[0]  
# # 4  
# # >>> c.shape[1]  
# # 2  
#     ###  ###
#     #初始化D，即每个样本的权重均为1/n

#     D = mat(ones((n, 1)) / m)    
#     ### 
#     # ones（3,3） 可以用来构造（3,3）全一矩阵
#     aggClasEst = mat(zeros((m, 1)))  
      
#     for i in range(G):#G表示的是迭代次数  
#         bestStump, minError, bestClasEst = singleStump(dataArr, classLabels, D)  
#         print('D:', D.T)
#         #计算分类器的权重  
#         alpha = float(0.5 * log((1.0 - minError) / max(minError, 1e-16)))  
#         bestStump['alpha'] = alpha  
#         weakClassArr.append(bestStump)  
#         print('bestClasEst:', bestClasEst.T)
          
#         #重新计算每个样本的权重D  
#         expon = multiply(-1 * alpha * mat(classLabels).T, bestClasEst)  
#         D = multiply(D, exp(expon))  
#         D = D / D.sum()  
          
#         aggClasEst += alpha * bestClasEst  
#         print('aggClasEst:', aggClasEst  )
#         aggErrors = multiply(sign(aggClasEst) != mat(classLabels).T, ones((m, 1)))  
#         errorRate = aggErrors.sum() / m  
#         print('total error:', errorRate  )
#         if errorRate == 0.0:  
#             break  
#     return weakClassArr  
  
# def adaBoostClassify(testData, weakClassify):  
#     dataMat = mat(testData)  
#     m = shape(dataMat)[0]  
#     aggClassEst = mat(zeros((m, 1)))  
#     for i in range(len(weakClassify)):#weakClassify是一个列表  
#         classEst = singleStumpClassipy(dataMat, weakClassify[i]['dim'], weakClassify[i]['threshold'], weakClassify[i]['inequal'])  
#         aggClassEst += weakClassify[i]['alpha'] * classEst  
#         print(aggClassEst  )
#     return sign(aggClassEst)  
              
# if __name__ == '__main__':  
#     datMat, classLabels = loadSimpleData()  
#     weakClassArr = adaBoostTrain(datMat, classLabels, 30)  
#     print("weakClassArr:", weakClassArr  )
#     #test  
#     result = adaBoostClassify([1, 1], weakClassArr)  
#     print(result  )


from numpy import *

#载入数据
def loadSimpData():
    datMat = matrix([[1., 2.1],
                     [2., 1.1],
                     [1.3, 1.],
                     [1., 1.],
                     [2., 1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels

#载入数据
def loadDataSet(fileName):
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat - 1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat

#预测分类
def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt': #比阀值小，就归为-1
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray

#建立单层决策树
def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax - rangeMin) / numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt'] : #less than 和greater than
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal,inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0 #分类错误的标记为1，正确为0
                weightedError = D.T * errArr #增加分类错误的权重
                print( "split: dim %d, thresh %.2f, thresh ineqal: %s, the weighted error is %.3f" \
                       % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst

#训练分类器
def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1)) / m)  #设置一样的初始权重值
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)  #得到“单层”最优决策树
        print("D:",D.T)
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  #计算alpha值
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)  #存储弱分类器
        print("classEst: ",classEst.T)
        expon = multiply(-1 * alpha * mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))  # 更新分类器权重
        D = D / D.sum() #保证权重加和为1
        aggClassEst += alpha * classEst
        print("aggClassEst: ",aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1))) #检查分类出错的类别
        errorRate = aggErrors.sum() / m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst

#用训练出的分类器来作预测
def adaClassify(datToClass, classifierArr):
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], \
                                 classifierArr[i]['thresh'], \
                                 classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha'] * classEst
        print(aggClassEst)
    return sign(aggClassEst)

#绘制ROC曲线
def plotROC(predStrengths, classLabels):
    import matplotlib.pyplot as plt
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1 / float(numPosClas)
    xStep = 1 / float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0] - delX], [cur[1], cur[1] - delY], c='b')
        cur = (cur[0] - delX, cur[1] - delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    print("the Area Under the Curve is: ", ySum * xStep)
    plt.show()

if __name__=='__main__':
    # filename = 'C:\\Users\\Administrator\\Desktop\\data\\horseColicTraining2.txt'
    # dataMat,classLabels = loadDataSet(filename)
    dataMat,classLabels = loadSimpData()
    weakClassArr, aggClassEst = adaBoostTrainDS(dataMat,classLabels,50)
    plotROC(aggClassEst.T, classLabels)
