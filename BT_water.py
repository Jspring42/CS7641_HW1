import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier
from sklearn.utils import resample
from sklearn.model_selection import learning_curve, validation_curve

def main():
    
    wq = pd.read_csv('./water_potability.csv')
    
    # # Drop useless columns
    wq = wq.dropna()
    
    wq_not = wq[wq.Potability==0]
    wq_pot = wq[wq.Potability==1]
    
    wq_pot_up = resample(wq_pot,random_state=42,n_samples=1200)
    
    wq = pd.concat([wq_not,wq_pot_up])
    
    X = wq.iloc[:,:-1].values
    Y = wq.iloc[:,-1:].values
    
    X, x_holdout,Y,y_holdout = train_test_split(X, Y,test_size=0.10)
    x_train, x_test,y_train,y_test = train_test_split(X, Y,test_size=0.20)

    np.random.seed(42)
    indices = np.arange(Y.shape[0])
    np.random.shuffle(indices)
    X, Y = X[indices], Y[indices]
    
    param_range = np.arange(10,150,10)
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
    
    plt.title("Water Boosted Validation Curve (n_estimators)")
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
    plt.show()
    fig.savefig('BT_water_n_estimators.png')

    param_range = np.arange(1,100,10)
    train_scores, test_scores = validation_curve(LGBMClassifier(random_state=42,
                                                               n_estimators=130), 
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
    
    plt.title("Water Boosted Validation Curve (num_leaves)")
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
    plt.show()
    fig.savefig('BT_water_num_leaves.png')
    
    final_model = LGBMClassifier(random_state=42, n_estimators=130, num_leaves=30)
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
    axes[0].set_title("Water BT Learning Curve/Size")
    axes[0].set_xlabel("Training Size")
    axes[0].set_ylabel("Accuracy")
    
    axes[1].grid()
    axes[1].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[1].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1)
    axes[1].set_xlabel("fit_times")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Water BT Learning Curve/Iteration Time")
    fig.savefig('BT_water_lc.png')

    final_model = LGBMClassifier(random_state=42, n_estimators=130, num_leaves=30)
    final_model.fit(x_train, y_train)
    
    y_pred = final_model.predict(x_holdout)
    cm = confusion_matrix(y_holdout, y_pred)
    
    categories = ['Not Potable', 'Potable']
    
    sns.heatmap(cm, cmap = 'Blues', fmt = '', annot = True,
                xticklabels = categories, yticklabels = categories)
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.xlabel("Predicted values", fontdict = {'size':14}, labelpad = 10)
    plt.ylabel("Actual values"   , fontdict = {'size':14}, labelpad = 10)
    plt.title ("Water BT Confusion Matrix", fontdict = {'size':18}, pad = 20)
    plt.savefig('BT_water_holdout_pred.png', figsize=[12,9])