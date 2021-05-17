from __future__ import absolute_import, division, print_function
import argparse
import numpy as np
from sklearn.svm import SVC
from models.DGP6 import DeepGP
import pandas as pd
# from models.DGP2 import DeepGP2
from sklearn.metrics import f1_score, classification_report, recall_score, precision_score, accuracy_score, r2_score
import sys, time, torch
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

def test(X_test,y_test, gpmodule):
    pred = gpmodule(X_test)
    # print('test() pred::', pred[:5], len(pred))
    # print('y_test::', y_test[:20])
    # print(pred.shape, y_test.shape)
    sum = 0.0
    # pred = np.argmax(pred)
    data = pd.read_csv('dongna/compar/ff-6c.csv') 
    data = np.array(data.loc[:,:]) #去除第一行
    # dat=df.drop(['None'],axis=1) #去除‘id’列
    pred1 = list(map(lambda x:x[1:], data))
    pred = [np.argmax(one_hot)for one_hot in pred1]

    # print('test() pred::', pred[:10], len(pred))
    for i in range(4160):   # 26000 * 0.8 20800
        if (pred[i] == y_test[i]):
            sum = sum + 1
    np.savetxt('dongna/Y_test6.csv', y_test, delimiter=',')
    np.savetxt('dongna/Y_test_pred6.csv', pred, delimiter=',')
    print('saved dongna test3')
    outputfile = open("dongna/compar/6cs.txt", "a")
    sys.stdout = outputfile
    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(sum, 4160, 100. * sum / 4160))
    print("acc:", accuracy_score(y_test, pred))
    print("precision", precision_score(y_test, pred, average='macro'))
    print("recall", recall_score(y_test, pred, average='micro'))
    print("F1", f1_score(y_test, pred, average='macro'))
    print(classification_report(y_test, pred))
    # return np.expand_dims(pred, axis =1)
    return pred1

# D:\vscode\vscodework\zangwen\feature_test.csv
def test2(X_test,y_test, gpmodule):
    pred = gpmodule(X_test)
    print(pred[:20])
    print(y_test[:20])
    print(pred.shape, y_test.shape)

    data = pd.read_csv('dongna/compar/ff-6c.csv') 
    data = np.array(data.loc[:,:]) #去除第一行
    # dat=df.drop(['None'],axis=1) #去除‘id’列
    pred1 = list(map(lambda x:x[1:], data))
    pred = [np.argmax(one_hot)for one_hot in pred1]

    sum = 0.0
    for i in range(5200):   # 26000 * 0.8 20800
        if (pred[i] == y_test[i]):
            sum = sum + 1
    np.savetxt('dongna/Y_test33.csv', y_test, delimiter=',')
    np.savetxt('dongna/Y_test_pred33.csv', pred, delimiter=',')
    print('saved dongna test33')
    outputfile = open("dongna/compar/6cs.txt", "a")
    sys.stdout = outputfile
    print("\nTest set: Accuracy: {}/{} ({:.2f}%)\n".format(sum, 5200, 100. * sum / 5200))
    print("acc:", accuracy_score(y_test, pred))
    print("precision", precision_score(y_test, pred, average='macro'))
    print("recall", recall_score(y_test, pred, average='micro'))
    print("F1", f1_score(y_test, pred, average='macro'))
    print(classification_report(y_test, pred ))
    # return np.expand_dims(pred, axis =1)
    return pred1

def main(args):  # 60cov
    data_train = np.loadtxt(open("D:/vscode/vscodework/zangwen/60covfeature_train.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    data_test = np.loadtxt(open("D:/vscode/vscodework/zangwen/60covfeature_test.csv", encoding='gb18030', errors="ignore"), delimiter=",", skiprows=0)
    X_train, y_train = data_train[:, :-1], data_train[:, -1]
    X2, y2 = data_test[:, :-1], data_test[:, -1]
    X_train = torch.from_numpy(X_train)
    y_train = torch.from_numpy(y_train)
    # print(X_train[:20])
    # print(X[:20])
    # print(y_train[:20])
    # print(y2[:20])
    X_test = torch.from_numpy(X2)
    y_test = torch.from_numpy(y2)
    # print(y_test[:20])
    members = list()
    num_classes=13
    print(X_train.shape, X_test.shape, y_test.shape)
    deep2 = DeepGP(X_train, y_train, num_classes=13)
    # deep3 = DeepGP(X_train, y_train, num_classes=13)
    members.append(deep2)
    # members.append(deep3)
    stack_train = np.zeros((X_train.shape[0], num_classes*len(members)),dtype=np.float32)  # Number of training data x Number of classifiers
    stack_test = np.zeros((X2.shape[0], num_classes*len(members)),dtype=np.float32)  # Number of testing data x Number of classifiers
    n_folds = 5
    a, b = 0, 0
    skf = KFold(n_splits=n_folds, shuffle=True, random_state=1)
    # For each classifier, we train the number of fold times (=len(skf))
    for j, clf in enumerate(members):
        print('Training classifier [%s]' % (j))
        for i, (train_index, cv_index) in enumerate(skf.split(X_train,y_train)):
            print('Fold [%s], j' % (i), j)
            print(train_index, cv_index, len(train_index), len(cv_index))
        # for j,(train_index,test_index) in enumerate(skf.split(X_train,y_train)):
            tr_x = X_train[train_index]
            tr_y = y_train[train_index]
            # reg.fit(tr_x, tr_y)
            # This is the training and validation set
            # X_train = X_dev[train_index]
            # Y_train = Y_dev[train_index]
            # X_cv = X_dev[cv_index]
            deepgp = DeepGP(tr_x, tr_y, num_classes=13)
            
            deepgp.train(args.num_epochs, args.num_iters, args.batch_size, args.learning_rate)
            # X_train = np.concatenate((X_train, ret_x),axis=0)
            # Y_train = np.concatenate((Y_train, ret_y),axis=0)

            # clf.fit(tr_x, tr_y)

            # clf.fit(tr_x, tr_y, verbose=1, batch_size=batch_size, epochs=epochs, validation_data=(X_test, y_test))
            # stack_train[cv_index, j*num_classes:(j+1)*num_classes] = test(X_train[cv_index], y_train[cv_index], clf)
            # stack_test[:, j*num_classes:(j+1)*num_classes] += test(X_test, y_test, clf)
            # train predction
            stack_train[cv_index, 0*num_classes:(0+1)*num_classes] = test(X_train[cv_index], y_train[cv_index], deepgp)
            stack_test[:, j*num_classes:(j+1)*num_classes] += test2(X_test, y_test, deepgp)
            # j = j+1
    stack_test = stack_test / float(n_folds)
    
    labelstack_train = pd.DataFrame(data=stack_train)
    probsstack_test = pd.DataFrame(data=stack_test)
    labelstack_train.to_csv('dongna/compar/train-6c.csv')
    probsstack_test.to_csv('dongna/compar/test-6c.csv')

    print('stack test  -->>  ', stack_test.shape)
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.ensemble import VotingClassifier
    ###第二层模型svc
    clf_second = SVC()
    clf_second.fit(stack_train,y_train)
    pred = clf_second.predict(stack_test)
    print('pred.shape::', pred.shape)
    print('r2_score:',  r2_score(y_test,pred))
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    classify_report = classification_report(y_test, pred, digits=4)
    print('SVC Test classify_report : \n', classify_report)
    print(' Test Accuracy: %.5f  \n' % acc)
    print('svc Test precision:', precision_score(y_test,pred, average='macro'))
    print(' Test f1 score: %.5f  \n' % f1)
    print(' Test recall score: %.5f  \n' % recall)
    # print(' Test precision: %.5f  \n' % precision)

    ###第二层模型LR
    clf_second = LogisticRegression()
    clf_second.fit(stack_train,y_train)
    pred = clf_second.predict(stack_test)
    print('pred.shape::', pred.shape)
    print('r2_score:',  r2_score(y_test,pred))
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    classify_report = classification_report(y_test, pred, digits=4)
    print('LogisticRegression Test classify_report : \n', classify_report)
    print(' Test Accuracy: %.5f  \n' % acc)
    print(' Test precision:', precision_score(y_test,pred, average='macro'))
    print(' Test f1 score: %.5f  \n' % f1)
    print(' Test recall score: %.5f  \n' % recall)

    ###第二层模型GaussianNB
    clf_second = GaussianNB()
    clf_second.fit(stack_train,y_train)
    pred = clf_second.predict(stack_test)
    print('pred.shape::', pred.shape)
    print('r2_score:',  r2_score(y_test,pred))
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    classify_report = classification_report(y_test, pred, digits=4)
    print('GaussianNB Test classify_report : \n', classify_report)
    print(' Test Accuracy: %.5f  \n' % acc)
    print('Test precision:', precision_score(y_test,pred, average='macro'))
    print(' Test f1 score: %.5f  \n' % f1)
    print(' Test recall score: %.5f  \n' % recall)

    ###第二层模型tree
    clf_second = DecisionTreeClassifier()
    clf_second.fit(stack_train,y_train)
    pred = clf_second.predict(stack_test)
    print('pred.shape::', pred.shape)
    print('r2_score:',  r2_score(y_test,pred))
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    classify_report = classification_report(y_test, pred, digits=4)
    print(' Test classify_report : \n', classify_report)
    print(' Test Accuracy: %.5f  \n' % acc)
    print('tree Test precision:', precision_score(y_test,pred, average='macro'))
    print(' Test f1 score: %.5f  \n' % f1)
    print(' Test recall score: %.5f  \n' % recall)

    ###第二层模型Forest
    clf_second = RandomForestClassifier()
    clf_second.fit(stack_train,y_train)
    pred = clf_second.predict(stack_test)
    print('pred.shape::', pred.shape)
    print('r2_score:',  r2_score(y_test,pred))
    
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average='macro')
    recall = recall_score(y_test, pred, average='macro')
    classify_report = classification_report(y_test, pred, digits=4)
    print('Forest Test classify_report : \n', classify_report)
    print(' Test Accuracy: %.5f  \n' % acc)
    print('Forest Test precision:', precision_score(y_test,pred, average='macro'))
    print(' Test f1 score: %.5f  \n' % f1)
    print(' Test recall score: %.5f  \n' % recall)

    # deepgp = DeepGP(X, y, num_classes=13)
    # time_start = time.time()
    # deepgp.train(args.num_epochs, args.num_iters, args.batch_size, args.learning_rate)
    # time_end = time.time()
    # cost = time_end - time_start
    # print('totally cost', time_end - time_start)
    # test(X_test, y_test, deepgp)
    # f1 = open("dongna/compar/60cov20RESULT0001-512-3c.txt", 'a')
    # f1.write(str(cost))

    # model1 = LogisticRegression(random_state=13)
    # model2 = GaussianNB()
    # model3 = RandomForestClassifier(random_state=13)
    # # 组合集成分类器（硬投票）
    # vote = VotingClassifier(estimators=[('lr', model1),
    #                                     ('gnb', model2),
    #                                     ('rfc', model3)],voting='hard')
    # vote.fit(X_train, y_train)
    # print('集成分类器的准确度：',vote.score(X_test, y_test))
    # model1.fit(X_train, y_train)
    # print('逻辑回归分类器的准确度：',model1.score(X_test, y_test))
    # model2.fit(X_train, y_train)
    # print('高斯朴素贝叶斯分类器的准确度：',model2.score(X_test, y_test))
    # model3.fit(X_train, y_train)
    # print('随机森林分类器的准确度：',model3.score(X_test, y_test))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Deep Gaussian Processes")
    parser.add_argument("-n", "--num-epochs", default=20, type=int)
    parser.add_argument("-t", "--num-iters", default=128, type=int)

    parser.add_argument("-b", "--batch-size", default=512, type=int)

    parser.add_argument("-lr", "--learning-rate", default=0.0001, type=float)
    args = parser.parse_args()
    main(args)




