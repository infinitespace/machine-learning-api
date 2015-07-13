import os
import sys
import array
from sklearn import svm
from sklearn import cross_validation
import numpy as np 
from sklearn.metrics.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.externals import joblib

# Set parameters
datafile = 'API_challenge_sample_data.txt'
outputfile = './models/'

# Read data from training datafile
def read_Data(filename):
    try:
        fr = open(filename)
        idx_line = 0
        lines = []
        for line in fr.readlines():
            idx_line = idx_line + 1
            if idx_line > 0:
                lines.append(line)
        X = np.zeros((idx_line - 1, 14))
        Y = np.zeros((idx_line - 1, 1))
        for i in range(idx_line - 1):
            tmp = lines[i+1].split(' ')
            for j in range(14):
                X[i,j] = float(tmp[j+1])
            Y[i,0] = float(tmp[0])
                
    except IOError:
        print "Error: can\'t find file or read data"
    return (X, Y)

# plot the confusion matrix
def plot_confusion_matrix(cm):
    plt.figure()
    plt.imshow(cm, interpolation='nearest')
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, (0,1), rotation=45)
    plt.yticks(tick_marks, (0,1))
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def mkdir(path):
    path=path.strip()
    path=path.rstrip("/")
    isExists=os.path.exists(path)
    if not isExists:
        print path+' success'
        os.makedirs(path)
        return True
    else:
        print path+' already exists'
        return False

# Calculate error rate for this model
def err_rate(cm):
    err_FP =  float(cm[0,1])/(cm[0,0]+cm[0,1])
    err_FN =  float(cm[1,0])/(cm[1,0]+cm[1,1])
    err_ALL = float(cm[0,1]+cm[1,0])/(cm[0,0]+cm[0,1]+cm[1,0]+cm[1,1])
    return err_FP, err_FN, err_ALL
    
def main(argv):
    (X, Y) = read_Data(datafile)
    ### implement cross validation ###
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
    
#     num_class_one = sum(y_train)
    sample_weight_class_one = np.ones(len(y_train))
    for i in range(len(y_train)):
        if y_train[i,0] == 1:
            sample_weight_class_one[i] = 1
    ### generate poly SVM model ###
    # clf = svm.SVC(kernel = 'poly', degree = 5)
    # clf.fit(X_train, y_train[:,0], sample_weight_class_one)
    # mkdir(outputfile)
    # joblib.dump(clf, outputfile + 'svm_poly_4.pkl') # save model to disc

    ### load existing model ###
    clf = joblib.load(outputfile + 'svm_poly_4.pkl') 

    y_pred = clf.predict(X_test)    # calculate prediction result
    cm = confusion_matrix(y_test, y_pred)

# plot_confusion_matrix(cm)
    print '\n', 'Confusion matrix:', '\n', cm

    err_FP, err_FN, err_ALL = err_rate(cm)
    print '\n', 'Over all accuracy: ', clf.score(X_test, y_test)*100, '%'
    print '\n', 'Fasle positive rate: ', err_FP*100, '%'
    print 'Fasle negative rate: ', err_FN*100, '%'
    print 'Over all error rate: ', err_ALL*100, '%'
    return 0

if __name__ == '__main__':
    main(sys.argv)
