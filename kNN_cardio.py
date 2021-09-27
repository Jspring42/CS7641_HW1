import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import learning_curve, validation_curve

def main():
    cv = pd.read_csv('./cardio_train.csv',sep=';')
    
    # # Drop useless columns
    cv = cv.drop(columns = ['id'])
    cv = cv.dropna()
    
    X = cv.iloc[:,:-1].values
    Y = cv.iloc[:,-1:].values
    
    X, x_holdout,Y,y_holdout = train_test_split(X, Y,test_size=0.10)
    x_train, x_test,y_train,y_test = train_test_split(X, Y,test_size=0.20)
    
    np.random.seed(42)
    indices = np.arange(Y.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    
    param_range = np.arange(1,50)
    train_scores, test_scores = validation_curve(KNeighborsClassifier(), 
                                                  X, Y, 
                                                  param_name="n_neighbors",
                                                  param_range=param_range,
                                                  cv=5,
                                                  scoring="accuracy", 
                                                  n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Cardio kNN Validation Curve (neighbors)")
    plt.xlabel("neighbors")
    plt.ylabel("Accuracy")
    # plt.ylim(0.0, 1.1)
    lw = 2
    fig = plt.gcf()
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    fig.savefig('kNN_cardio_neigh.png')
    
    param_range = ['euclidean','manhattan','chebyshev','minkowski','wminkowski','seuclidean','mahalanobis']
    train_scores, test_scores = validation_curve(KNeighborsClassifier(n_neighbors=20), 
                                                  X, Y, 
                                                  param_name="metric",
                                                  param_range=param_range,
                                                  cv=5,
                                                  scoring="accuracy", 
                                                  n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Cardio kNN Validation Curve (metric)")
    plt.xlabel("metric")
    plt.ylabel("Accuracy")
    # plt.ylim(0.0, 1.1)
    lw = 2
    fig = plt.gcf()
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    fig.savefig('kNN_cardio_metric.png')
    
    final_model = KNeighborsClassifier(n_neighbors=30,metric='manhattan')
    train_sizes=np.linspace(.1, 1.0, 5)
    
    train_sizes, train_scores, test_scores = learning_curve(final_model, X, Y, cv=5, n_jobs=-1,train_sizes=train_sizes)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    fig = plt.gcf()
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1,
                  color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                  test_scores_mean + test_scores_std, alpha=0.1,
                  color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
          label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
        label="Cross-validation score")
    plt.legend(loc="best")
    plt.title("Cardio kNN Learning Curve")
    plt.xlabel("Training Size")
    plt.ylabel("Accuracy")
    fig.savefig('kNN_cardio_lc.png')
    
    final_model.fit(x_train, y_train)
    
    y_pred = final_model.predict(x_holdout)
    cm = confusion_matrix(y_holdout, y_pred)
    
    categories = ['No Disease', 'Disease']
    
    sns.heatmap(cm, cmap = 'Blues', fmt = '', annot = True,
                xticklabels = categories, yticklabels = categories)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Cardio kNN Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.savefig('kNN_cardio_holdout_pred.png')