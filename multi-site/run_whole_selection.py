#!/usr/bin/env python

# Run the multi-site selection procudure on all the dataset

import numpy as np
import pandas as pd
import re

#from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error
#from sklearn.model_selection import KFold
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

import os

# calculate aic for regression
def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic



def plot_ic(q_arr, aic_arr, bic_arr, feature_arr, x_label, y_label, plot_name):
    fig, ax1 = plt.subplots(figsize=(8,5))

    color = 'k'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color=color)
    ax1.plot(q_arr, aic_arr, color='pink', label="AIC")
    ax1.plot(q_arr, bic_arr, color='lightskyblue', label='BIC')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:gray'
    ax2.set_ylabel('Num of features', color=color)
    ax2.plot(q_arr, feature_arr, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.legend()
    fig.tight_layout()

    plt.savefig(plot_name + '.pdf', format = 'pdf')
    plt.close()


def plot(y, predicted_y, name):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y, predicted_y, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(name + '.pdf', format = 'pdf')
    plt.close()

def plot_performance(q_arr, train_arr, test_arr, feature_arr, x_label, y_label, plot_name):
    fig, ax1 = plt.subplots(figsize=(8,5))

    color = 'k'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color=color)
    ax1.plot(q_arr, train_arr, color='r', label="training")
    ax1.plot(q_arr, test_arr, color='b', label='testing')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:gray'
    ax2.set_ylabel('Num of features', color=color)
    ax2.plot(q_arr, feature_arr, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.legend()
    fig.tight_layout()

    plt.savefig(plot_name + '.pdf', format = 'pdf')
    plt.close()

def plot_performance_train(q_arr, train_arr, feature_arr, x_label, y_label, plot_name):
    fig, ax1 = plt.subplots(figsize=(8,5))

    color = 'k'
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y_label, color=color)
    ax1.plot(q_arr, train_arr, color='r', label="whole")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()

    color = 'tab:gray'
    ax2.set_ylabel('Num of features', color=color)
    ax2.plot(q_arr, feature_arr, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax1.legend()
    fig.tight_layout()

    plt.savefig(plot_name + '.pdf', format = 'pdf')
    plt.close()



def selected_performance_lr(X, y, plot_name):
    lr = linear_model.LinearRegression()
    predicted = cross_val_predict(lr, X, y, cv=10)
    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    corr = np.corrcoef(y, predicted)[1,0]
    plot(y, predicted, plot_name + '_train')

    print('After 2nd LR: {0} features were selected.'.format(len(X.columns)))

    num_params = len(X.columns) + 1 # number of features put in lr
    aic = calculate_aic(len(y), mse, num_params)
    bic = calculate_bic(len(y), mse, num_params)

    return r2, mse, corr, len(X.columns), aic, bic



def selected_performance_lasso(X, y, plot_name):
    alphas = np.array([0.001, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
    model = linear_model.Lasso(random_state=random_state, selection='cyclic')
    grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), n_jobs=40, cv=5, refit=True)
    grid = grid.fit(X, y)
    print('Grid search best score: {}'.format(grid.best_score_))
    best_alpha = grid.best_params_['alpha']
    print('Grid search best alpha: {}'.format(best_alpha))

    lasso = linear_model.Lasso(alpha=best_alpha, random_state=random_state, selection='cyclic')
    lasso = lasso.fit(X, y)

    support = lasso.coef_.copy()
    support[support!=0] = 1

    print('After 2nd LASSO: {0} features were selected.'.format(np.sum(support)))


    predicted = lasso.predict(X)

    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    corr = np.corrcoef(y, predicted)[1,0]
    plot(y, predicted, plot_name + '_train')


    num_params = len(support) + 1 # number of features put in lasso
    aic = calculate_aic(len(y), mse, num_params)
    bic = calculate_bic(len(y), mse, num_params)

    return r2, mse, corr, np.sum(support), aic, bic





def load_data(meth_file, sample_file):
    meth = pd.read_csv(meth_file, header = 0, index_col = 0, sep = '\t')
    meth_trans = meth.T
    del meth
    sample = pd.read_csv(sample_file, header = 0, index_col = 0, sep = ',')
    return meth_trans, sample


def data_preprocess_standardize(meth, sample, dep_name,
                    cov_name = ['SEX_new', 'AGE', 'SMOKING_new', 'DMAGE', 'HBA1C', 'SBP', 'DBP',
                                'CD8T', 'CD4T', 'NK', 'Bcell', 'Mono', 'Gran',
                                'sentrix_code', 'sentrix_pos', 'sample_plate', 'sample_well']):
    '''
    meth: df, methylation matrix, each row is a sample, each column is a CpG site.
    sample: df, sample information matrix, each row is a sample, each column is a clinical variable, containing covariates that need to be adjusted and the dependent variable.
    dep_name: str, name of the dependent variable. In this case, it is either 'eGFR_CKDEPI' or 'eGFR_slope'.
    cov_name: list of str, counfounding factors.
'''
    var = sample.loc[:,cov_name]
    ## remove samples with "NA"
    var.dropna(axis = 'index', how = 'any', inplace=True)
    ## remoce CpG sites with "NA"
    meth.dropna(axis = 'columns', how = 'any', inplace=True)

    dataX = meth.join(var, how='inner')

    scaler = StandardScaler()
    dataX_scale = scaler.fit_transform(dataX)
    dataX_scale_df = pd.DataFrame(data=dataX_scale, index=dataX.index, columns=dataX.columns)



    dep_var = sample.loc[:,dep_name]
    ## remove samples with "NA" y
    dep_var.dropna(axis = 'index', how = 'any', inplace=True)

    data = dataX_scale_df.join(dep_var, how='inner')

    n, m = data.shape
    #data.sort_values(by=[dep_name], inplace=True)
    X = data.iloc[:, 0:m-1]
    y = data.iloc[:, m-1]
    n_samples, n_features = X.shape
    print('There are {} samples and {} features.'.format(n_samples, n_features))
    return X, y


def feature_selection(X, y, plot_dir, alphas, n_splits=10, n_repeats=10, n_jobs=40):
    n_samples, n_features = X.shape
    Xvalue = X.values
    yvalue = y.values
    total = n_splits * n_repeats
    supports = np.zeros((total, n_features))

    train_r2 = np.zeros(total)
    train_mse = np.zeros(total)
    train_pcc = np.zeros(total)
    test_r2 = np.zeros(total)
    test_mse = np.zeros(total)
    test_pcc = np.zeros(total)
    no_feature = np.zeros(total)

    #k_fold = KFold(n_fold)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    for k, (train, test) in enumerate(rkf.split(Xvalue, yvalue)):
        xtrain, xtest, ytrain, ytest = Xvalue[train], Xvalue[test], yvalue[train], yvalue[test]

        model = linear_model.Lasso(random_state=random_state, selection='random')
        grid = GridSearchCV(estimator=model, param_grid=dict(alpha=alphas), n_jobs=n_jobs, cv=5, refit=False)
        grid = grid.fit(xtrain, ytrain)
        print('Grid search best score: {}'.format(grid.best_score_))
        best_alpha = grid.best_params_['alpha']
        print('Grid search best alpha: {}'.format(best_alpha))

        mean_test_score = grid.cv_results_['mean_test_score']
        max_ = np.max(mean_test_score)
        std_ = np.std(mean_test_score)
        for i in range(len(alphas)-1, -1, -1):
            if mean_test_score[i] >= max_ - std_:
                break
        print('Grid search select score: {}'.format(mean_test_score[i]))
        select_alpha = alphas[i]
        print('Grid search select alpha: {}'.format(select_alpha))

        ## fit lasso with best alpha on the whole training data
        lasso = linear_model.Lasso(alpha=select_alpha, random_state=random_state, selection='random')
        lasso = lasso.fit(xtrain, ytrain)
        ytrain_pred = lasso.predict(xtrain)
        ytest_pred = lasso.predict(xtest)
        r2_score_lasso_train = r2_score(ytrain, ytrain_pred)
        mse_lasso_train = mean_squared_error(ytrain, ytrain_pred)
        corr_lasso_train = np.corrcoef(ytrain, ytrain_pred)[1,0]
        print('[Fold {0} training] r^2 score: {1:.5f}, mean squared error: {2:.5f}, pearson correlation coefficient: {3: .5f}'.
                format(k, r2_score_lasso_train, mse_lasso_train, corr_lasso_train))
        plot(ytrain, ytrain_pred, plot_dir + 'lasso_cv_training_fold{0}'.format(k))
        train_r2[k] = r2_score_lasso_train
        train_mse[k] = mse_lasso_train
        train_pcc[k] = corr_lasso_train
        r2_score_lasso_test = r2_score(ytest, ytest_pred)
        mse_lasso_test = mean_squared_error(ytest, ytest_pred)
        corr_lasso_test = np.corrcoef(ytest, ytest_pred)[1,0]
        print('[Fold {0} testing] r^2 score: {1:.5f}, mean squared error: {2:.5f}, pearson correlation coefficient: {3: .5f}'.
                format(k, r2_score_lasso_test, mse_lasso_test, corr_lasso_test))
        plot(ytest, ytest_pred, plot_dir + 'lasso_cv_testing_fold{0}'.format(k))
        test_r2[k] = r2_score_lasso_test
        test_mse[k] = mse_lasso_test
        test_pcc[k] = corr_lasso_test
        support = lasso.coef_.copy()
        #support[support!=0] = 1
        support[support!=0] = corr_lasso_test
        supports[k,:] = support

        no_feature[k] = np.count_nonzero(support)
        print('Fold {0}: {1} features were selected.'.format(k, np.count_nonzero(support)))

    print('[{0} fold average training] r^2 score: {1:.5f}, mean squared error: {2:.5f}, pearson correlation coefficient: {3: .5f}'.
        format(total, np.mean(train_r2), np.mean(train_mse), np.mean(train_pcc)))
    print('[{0} fold average testing] r^2 score: {1:.5f}, mean squared error: {2:.5f}, pearson correlation coefficient: {3: .5f}'.
        format(total, np.mean(test_r2), np.mean(test_mse), np.mean(test_pcc)))

    q_arr = np.arange(total)
    plot_performance(q_arr, train_r2, test_r2, no_feature, 'kth model', 'R^2', plot_dir + 'select_lasso_r2')
    plot_performance(q_arr, train_mse, test_mse, no_feature, 'kth model', 'Mean squared error', plot_dir + 'select_lasso_mse')
    plot_performance(q_arr, train_pcc, test_pcc, no_feature, 'kth model', 'Pearson correlation', plot_dir + 'select_lasso_pcc')
    return supports, total



def prediction_with_selected_features_weights(X, y, supports, out_dir, plot_dir, eva_model='lasso', cov=False):
    res = pd.DataFrame(np.transpose(supports), index=X.columns)
    selected = res.loc[(res!=0).any(axis=1)]
    selected_sum = selected.sum(axis=1)
    selected_sum.sort_values(ascending=False, inplace=True)
    selected_sum.to_csv(out_dir + 'selected_CpG.csv', sep=',', header=False, index=True)

    selected_features = selected_sum.index.values

    if cov == False:
        tmp = []
        for i in selected_features:
            if re.match("cg.*", i):
                tmp.append(i)
        selected_features = tmp

    tot_features = len(selected_features)

    r2_arr = np.zeros(tot_features)
    mse_arr = np.zeros(tot_features)
    corr_arr = np.zeros(tot_features)
    feature_arr = np.zeros(tot_features)
    aic_arr = np.zeros(tot_features)
    bic_arr = np.zeros(tot_features)


    for q in range(tot_features):
        sub_features = selected_features[0:(tot_features-q)]

        X_select = X.loc[:, sub_features]
        data_new = X_select.join(y, how='inner')

        data_shuffle = data_new.sample(frac=1)
        n, m = data_shuffle.shape
        X_new = data_shuffle.iloc[:, 0:m-1]
        y_new = data_shuffle.iloc[:, m-1]
        n_samples, n_features = X_new.shape
        print('After selection ({}): there are {} samples and {} features.'.format(q, n_samples, n_features))



        plot_name = plot_dir + eva_model +  '_selected_{}'.format(q)
        if eva_model == 'lr':
            r2, mse, corr, select_num, aic, bic = selected_performance_lr(X_new, y_new, plot_name)
            feature_arr[q] = n_features

        elif eva_model == 'lasso':
            r2, mse, corr, select_num, aic, bic = selected_performance_lasso(X_new, y_new, plot_name)
            feature_arr[q] = select_num


        r2_arr[q] = r2
        mse_arr[q] = mse
        corr_arr[q] = corr

        aic_arr[q] = aic
        bic_arr[q] = bic

    q_arr = np.arange(tot_features)
    plot_performance_train(q_arr, r2_arr, feature_arr, 'feature selection threshold', 'R^2', plot_dir + 'evaluate_' + eva_model + '_r2')
    plot_performance_train(q_arr, mse_arr, feature_arr, 'feature selection threshold',  'Mean squared error', plot_dir + 'evaluate_' + eva_model + '_mse')
    plot_performance_train(q_arr, corr_arr, feature_arr, 'feature selection threshold','Pearson correlation', plot_dir + 'evaluate_' + eva_model + '_pcc')


    plot_ic(q_arr, aic_arr, bic_arr, feature_arr, 'feature selection threshold', 'Information criteria', plot_dir + 'evaluate_' + eva_model + '_ic')

    return r2_arr, mse_arr, corr_arr, feature_arr, aic_arr, bic_arr


if __name__ == "__main__":
    dep_name = sys.argv[1]
    ## e.g., eGFR_CKDEPI or eGFR_slope
    data_dir = sys.argv[2]
    ## e.g., '../example_data/'
    meth_file = sys.argv[3]
    ## e.g., '../example_data/meth_eg.txt'
    sample_file = sys.argv[4]
    ## e.g., '../example_data/sample_eg.csv'


    if dep_name == 'eGFR_CKDEPI':
        alphas = np.array([0.01, 0.1, 1.0, 1.5, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2, 4.4, 4.6, 4.8, 5.0])
    elif dep_name == 'eGFR_slope':
        alphas = np.array([0.001, 0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])


    random_state = np.random.RandomState(0)

    out_dir = './output_whole_' + dep_name + '/'
    plot_dir_select = out_dir + 'plots_select/'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if not os.path.exists(plot_dir_select):
        os.makedirs(plot_dir_select)


    print("=============Loading data=============")
    meth, sample = load_data(meth_file, sample_file)
    print("=============Loading data finished=============")

    print("=============Data preprocessing=============")
    X, y = data_preprocess_standardize(meth, sample, dep_name=dep_name)
    print("=============Data preprocessing finished=============")



    print("=============Feature selection=============")
    ## select features in all the dataset
    supports, total = feature_selection(X, y, plot_dir_select, alphas, n_jobs=50)


    eva_model='lasso'
    cov = False

    if cov == True:
        plot_dir_evaluate = out_dir + 'plots_' + eva_model + '_cov/'
    else:
        plot_dir_evaluate = out_dir + 'plots_' + eva_model + '_cpg/'


    if not os.path.exists(plot_dir_evaluate):
        os.makedirs(plot_dir_evaluate)


    print("=============Prediction using selected features=============")
    ## predict in the whole set using selected features
    r2_arr, mse_arr, corr_arr, feature_arr, aic_arr, bic_arr = prediction_with_selected_features_weights(X, y, supports, out_dir, plot_dir_evaluate, eva_model, cov)

    performance_df = pd.DataFrame([feature_arr, r2_arr, corr_arr, mse_arr, aic_arr, bic_arr], index = ['num_sites', 'r2_train', 'pcc_train',  'mse_train', 'AIC', 'BIC']).T
    performance_df.to_csv(out_dir + 'evaluate_performance.csv', sep=',', header=True, index=False)





    print("=============Determine feature selection threshold=============")
    performance_df = pd.read_csv(out_dir + 'evaluate_performance.csv', sep=',', header=0)
    selected_features = pd.read_csv(out_dir + 'selected_CpG.csv', sep=',', names=['feature', 'weights']).iloc[:,0].values

    tmp = []
    for i in selected_features:
        if re.match("cg.*", i):
            tmp.append(i)
    selected_features = tmp

    bic = performance_df['BIC'].values
    assert len(selected_features) == len(bic), "Length of selected features and length of performance don't match!"

    min_bic = np.min(bic)
    std_bic = np.std(bic)
    for i in range(len(bic)):
        if bic[i] <= min_bic + 0.1*std_bic:
            break

    print('Final select threshold based on BIC is {}'.format(i))
    print('Final number of selected CpG sites is {}'.format(len(bic)-i))
    final_select = selected_features[0:len(bic)-i]

    f_select = open(out_dir + 'final_selected.csv', 'w')
    for cpg in final_select:
        f_select.write(cpg + '\n')
    f_select.close()
