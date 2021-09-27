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
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
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
    
    sc = StandardScaler()
    # X = sc.fit_transform(X)
    X_train = sc.fit_transform(x_train)
    X_test = sc.fit_transform(x_test)
    X_holdout = sc.fit_transform(x_holdout)
    
    # n_estimators tuning
    np.random.seed(42)
    indices = np.arange(Y.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    
    param_range = np.arange(10,4000,50)
    train_scores, test_scores = validation_curve(LGBMClassifier(random_state=42), 
                                                  X, np.ravel(Y), 
                                                  param_name="n_estimators",
                                                  param_range=param_range,
                                                  cv=5,
                                                  scoring="accuracy", 
                                                  n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Cardio Boosted Validation Curve (n_estimators)")
    plt.xlabel("n_estimators")
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
    fig.savefig('BT_cardio_n_estimators.png')
    
    # num_leaves tuning
    param_range = np.arange(1,100,10)
    train_scores, test_scores = validation_curve(LGBMClassifier(random_state=42,
                                                               n_estimators=40), 
                                                  X, np.ravel(Y), 
                                                  param_name="num_leaves",
                                                  param_range=param_range,
                                                  cv=5,
                                                  scoring="accuracy", 
                                                  n_jobs=-1)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    
    plt.title("Cardio Boosted Validation Curve (num_leaves)")
    plt.xlabel("num_leaves")
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
    fig.savefig('BT_cardio_num_leaves.png')
    
    # plot learning curve
    final_model = LGBMClassifier(random_state=42, n_estimators=100, num_leaves=20)
    train_sizes=np.linspace(.1, 1.0, 5)
    
    train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(final_model, 
                                                            X, np.ravel(Y), 
                                                            cv=5, 
                                                            n_jobs=-1,
                                                            train_sizes=train_sizes,
                                                            return_times=True)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)
    
    # Plot learning curve
    _, axes = plt.subplots(2, 1, figsize=(5, 10))
    fig = plt.gcf()
    plt.grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1,
                      color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1,
                      color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                      label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                      label="Cross-validation score")
    axes[0].legend(loc="best")
    axes[0].set_title("Cardio BT Learning Curve/Size")
    axes[0].set_xlabel("Training Size")
    axes[0].set_ylabel("Accuracy")
    
    axes[1].grid()
    axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Cardio BT Learning Curve/Iteration Time")
    fig.savefig('BT_cardio_lc.png')
    
    # plot CM
    final = LGBMClassifier(random_state=42, n_estimators=100, num_leaves=20)
    final.fit(x_train, y_train)
    y_pred = final.predict(x_holdout)
    
    cm = confusion_matrix(y_holdout, y_pred)
    
    categories = ['No Disease', 'Disease']
    
    sns.heatmap(cm, cmap = 'Blues', fmt = '', annot = True,
                xticklabels = categories, yticklabels = categories)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Cardio BT Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.savefig('Boosted_cardio_holdout_pred.png')
    
if __name__ == "__main__":
    
    main()