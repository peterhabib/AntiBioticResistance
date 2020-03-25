# set up environment
import joblib
import pandas as pd
import numpy as np
# Libraries
import datetime
import re

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import sys

from matplotlib.colors import LogNorm
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler, Nystroem
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, homogeneity_score, adjusted_rand_score, \
    roc_auc_score, roc_curve, f1_score, auc, average_precision_score, brier_score_loss, precision_score, recall_score, \
    jaccard_score, adjusted_mutual_info_score, completeness_score, fowlkes_mallows_score, normalized_mutual_info_score, \
    explained_variance_score, max_error, mean_squared_log_error, median_absolute_error, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import v_measure_score
from sklearn import metrics, svm, pipeline, mixture
from sklearn.mixture import GaussianMixture
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import Perceptron, Lasso, LassoCV
from sklearn.neighbors import NearestCentroid
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.semi_supervised import LabelSpreading
from sklearn.semi_supervised import LabelPropagation
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble.forest import RandomForestClassifier
from sklearn.ensemble.forest import ExtraTreesClassifier
from sklearn.ensemble.bagging import BaggingClassifier
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier
from sklearn.ensemble.weight_boosting import AdaBoostClassifier
from sklearn.gaussian_process.gpc import GaussianProcessClassifier
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier
from sklearn.linear_model.ridge import RidgeClassifier
from sklearn.linear_model.ridge import RidgeClassifierCV
from sklearn.neighbors.classification import RadiusNeighborsClassifier
from sklearn.neural_network.multilayer_perceptron import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import ExtraTreeClassifier
from sklearn.cluster import KMeans
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,homogeneity_score,adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import v_measure_score
from sklearn import metrics
from sklearn.svm import NuSVC
import time
from sklearn.svm.classes import OneClassSVM
from sklearn.naive_bayes import MultinomialNB
from datetime import datetime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import confusion_matrix
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit

import matplotlib.pyplot as plt
import seaborn as sb
import time


Eval_file = open('Evaluation.csv','w')
Eval_file.write('balanced_accuracy_score, average_precision_score, brier_score_loss, f1_score, precision_score,'
                'recall_score, jaccard_score, roc_auc_score, adjusted_mutual_info_score, adjusted_rand_score, completeness_score,'
                'fowlkes_mallows_score,homogeneity_score, v_measure_score, explained_variance_score, max_error, mean_absolute_error,'
                'mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score, algo,train_time,predict_time,eval_time\n')


# a function for preparing our training and testing data
def prep_data(phenotype):
    pheno = pd.read_csv('./gono-unitigs/metadata.csv', index_col=0)
    pheno = pheno.dropna(subset=[phenotype])  # drop samples that don't have a value for our chosen resistance profile
    pheno = pheno[phenotype]

    # read in unitig data
    X = pd.read_csv('./gono-unitigs/' + phenotype + '_gwas_filtered_unitigs.Rtab', sep=" ", index_col=0,
                    low_memory=False)
    X = X.transpose()
    X = X[X.index.isin(pheno.index)]  # only keep rows with a resistance measure
    pheno = pheno[pheno.index.isin(X.index)]
    return X, pheno





# prepare our data for predicting ciprofloxacin resistance
phenotype = 'cip_sr'
X, pheno = prep_data(phenotype)

# create an array for storing performance metrics
performance = []
method = []
times = []



# look at the length distribution of the unitigs in our dataset
unitigs = X.columns
mylen = np.vectorize(len)
uni_len = mylen(unitigs)
sb.distplot(uni_len)



# function for fitting a model
def fitmodel(X, pheno, estimator, parameters, modelname, method, performance, times):
    score_store = 0

    kfold = KFold(n_splits=5)
    for train_index, test_index in kfold.split(X, pheno):
        # time how long it takes to train each model type
        start = time.process_time()

        # split data into train/test sets
        X_train = X.iloc[train_index]
        y_train = pheno[train_index]
        X_test = X.iloc[test_index]
        y_test = pheno[test_index]

        # perform grid search to identify best hyper-parameters
        train_time = time.time()
        gs_clf =estimator
        gs_clf.fit(X_train, y_train)
        train_end = time.time() - train_time

        # predict resistance in test set
        predict_time = time.time()
        y_pred = gs_clf.predict(X_test)
        y_pred[y_pred < 0.5] = 0
        y_pred[y_pred > 0.5] = 1
        predict_end = time.time() - predict_time


        eval_time = time.time()
        score = balanced_accuracy_score(y_test, y_pred)
        balanced_accuracy_scoreval = balanced_accuracy_score(y_test, y_pred)
        average_precision_scoreval = average_precision_score(y_test, y_pred)
        brier_score_lossval =  brier_score_loss(y_test, y_pred)
        f1_scoreval =  f1_score(y_test, y_pred)
        precision_scoreval = precision_score(y_test, y_pred)
        recall_scoreval = recall_score(y_test, y_pred)
        jaccard_scoreval = jaccard_score(y_test, y_pred)
        roc_auc_scoreval = roc_auc_score(y_test, y_pred)
        adjusted_mutual_info_scoreval = adjusted_mutual_info_score(y_test, y_pred)
        adjusted_rand_scoreval = adjusted_rand_score(y_test, y_pred)
        completeness_scoreval = completeness_score(y_test, y_pred)
        fowlkes_mallows_scoreval = fowlkes_mallows_score(y_test, y_pred)
        homogeneity_scoreval = normalized_mutual_info_score(y_test, y_pred)
        v_measure_scoreval = v_measure_score(y_test, y_pred)
        explained_variance_scoreval = explained_variance_score(y_test, y_pred)
        max_errorval = max_error(y_test, y_pred)
        mean_absolute_errorval = mean_absolute_error(y_test, y_pred)
        mean_squared_errorval = mean_squared_error(y_test, y_pred)
        mean_squared_log_errorval = mean_squared_log_error(y_test, y_pred)
        median_absolute_errorval = median_absolute_error(y_test, y_pred)
        r2_scoreval = r2_score(y_test, y_pred)
        eval_end = time.time() - eval_time


















        # Create classifiers
        gnb = model

        # #############################################################################
        # Plot calibration plots

        plt.figure(figsize=(10, 10))
        ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
        ax2 = plt.subplot2grid((3, 1), (2, 0))

        ax1.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")
        for clf, name in [
                          (gnb, '%s'%algo),

                          ]:
            clf.fit(X_train, y_train)
            if hasattr(clf, "predict_proba"):
                prob_pos = clf.predict_proba(X_test)[:, 1]
            else:  # use decision function
                prob_pos = clf.decision_function(X_test)
                prob_pos = \
                    (prob_pos - prob_pos.min()) / (prob_pos.max() - prob_pos.min())
            fraction_of_positives, mean_predicted_value = \
                calibration_curve(y_test, prob_pos, n_bins=10)

            ax1.plot(mean_predicted_value, fraction_of_positives, "s-",
                     label="%s" % (name,))

            ax2.hist(prob_pos, range=(0, 1), bins=10, label=name,
                     histtype="step", lw=2)

        ax1.set_ylabel("Fraction of positives")
        ax1.set_ylim([-0.05, 1.05])
        ax1.legend(loc="lower right")
        ax1.set_title('Calibration plots  (%s_%s)'% (algo,score))

        ax2.set_xlabel("Mean predicted value")
        ax2.set_ylabel("Count")
        ax2.legend(loc="upper center", ncol=2)

        plt.tight_layout()
        plt.savefig('%s_%s_Calibration_plots.svg' % (algo,score))














        performance = np.append(performance, score)
        method = np.append(method, modelname)
        times = np.append(times, (time.process_time() - start))


        # print("Confusion matrix for this fold")
        # print(algo, 'Accuracy Score:',score*100)
        print('###################################################')



        filename = '%s_Acc: %s.sav' % (algo, score)
        joblib.dump(gs_clf, filename)
        print(filename)

        to_write = "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s`,`%s\n" % (
        balanced_accuracy_scoreval, average_precision_scoreval, brier_score_lossval, f1_scoreval, precision_scoreval,
        recall_scoreval, jaccard_scoreval, roc_auc_scoreval, adjusted_mutual_info_scoreval, adjusted_rand_scoreval, completeness_scoreval,
        fowlkes_mallows_scoreval,
        homogeneity_scoreval, v_measure_scoreval, explained_variance_scoreval, max_errorval, mean_absolute_errorval,
        mean_squared_errorval, mean_squared_log_errorval, median_absolute_errorval, r2_scoreval, algo,train_end,predict_end,eval_end)

        Eval_file.write(str(to_write))





    return gs_clf, method, performance, times






















# function for looking at SVM feature importance
sb.set_context("talk")
def plot_coefficients(model_name, classifier, feature_names, top_features=10):
    coef = classifier.best_estimator_.coef_.ravel()
    top_positive_coefficients = np.argsort(coef)[-top_features:]
    top_negative_coefficients = np.argsort(coef)[:top_features]
    top_coefficients = np.hstack([top_negative_coefficients, top_positive_coefficients])
    # create plot
    plt.figure(figsize=(10,10))
    plt.title("Feature Importances (%s) - Ciprofloxacin Resistance"%algo, y=1.08)
    colors = ['crimson' if c < 0 else 'cornflowerblue' for c in coef[top_coefficients]]
    plt.bar(np.arange(2 * top_features), coef[top_coefficients], color=colors)
    feature_names = np.array(feature_names)
    plt.xticks(np.arange(0, 1 + 2 * top_features), feature_names[top_coefficients], rotation=60, ha='right')
    plt.savefig("%s_FI.svg"%model_name)
    np.asarray(feature_names)[top_positive_coefficients]


def plot_learning_curve(model_name,estimator, title, X, y, ylim=None):
    sb.set_context("talk")
    plt.figure(figsize=(10, 10))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Balanced accuracy")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, scoring="balanced_accuracy")
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid(color='gainsboro')

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="crimson")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="cornflowerblue")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="crimson",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="cornflowerblue",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig("%s_LC.svg"%model_name)
    return plt


algorithms = ['KNeighborsClassifier','GaussianNB','SGDClassifier','ExtraTreeClassifier','DecisionTreeClassifier',
              'MLPClassifier','RidgeClassifierCV','RidgeClassifier','GaussianProcessClassifier',
              'AdaBoostClassifier','GradientBoostingClassifier','BaggingClassifier','ExtraTreesClassifier','RandomForestClassifier',
              'CalibratedClassifierCV','LinearDiscriminantAnalysis','LinearSVC',
              'LogisticRegression','LogisticRegressionCV','Perceptron','QuadraticDiscriminantAnalysis',
              'MultinomialNB']

for algo in algorithms:
    model = eval('%s()'%algo)
    G_params = {}
    G_model, method, performance, times = fitmodel(X, pheno, model, G_params, model, method, performance, times)


    title = "Learning Curve (%s) - Ciprofloxacin resistance"%algo
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)

    estimator = model
    plot_learning_curve(algo, estimator, title, X, pheno, ylim=(0.7, 1.01))




    # plot_coefficients(algo, svm_model, list(X.columns))
    #
    # # if we print the unitigs, we can then look at what genes they relate to
    # coef = svm_model.best_estimator_.coef_.ravel()
    # feature_names = list(X.columns)
    # top_negative_coefficients = np.argsort(coef)[:5]
    # print("Top negative predictors: ", np.asarray(feature_names)[top_negative_coefficients])
    #
    # top_positive_coefficients = np.argsort(coef)[-5:]
    # print("Top positive predictors: ", np.asarray(feature_names)[top_positive_coefficients])



# compare results from the different predictors
sb.set_context("talk")
plt.title("Model Performance - Ciprofloxacin Resistance", y=1.08)
sb.swarmplot(x=method, y=performance, palette="YlGnBu_d", size=10)
sb.despine()
plt.ylabel("Balanced accuracy")
plt.xticks(rotation=30, ha="right")
plt.savefig('Model_performance.svg')
plt.show()
