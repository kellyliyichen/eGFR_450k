#!/usr/bin/env python

import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

from sklearn import linear_model

from sklearn.metrics import r2_score, explained_variance_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_predict

from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt

from scipy import stats
import os

# calculate aic for regression
def calculate_aic(n, mse, num_params):
    aic = n * np.log(mse) + 2 * num_params
    return aic


# calculate bic for regression
def calculate_bic(n, mse, num_params):
    bic = n * np.log(mse) + num_params * np.log(n)
    return bic


def plot(y, predicted_y, name):
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(y, predicted_y, edgecolors=(0, 0, 0))
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    plt.savefig(name + '.pdf', format = 'pdf')
    plt.close()



def get_selected_features(selected_feature_file):
    selected_features = []
    with open(selected_feature_file, 'r') as f:
        for line in f:
            cpg = line.strip()
            selected_features.append(cpg)
    return selected_features



def load_data(meth_file, sample_file, selected_features):
    meth = pd.read_csv(meth_file, header = 0, index_col = 0, sep = '\t')
    meth_trans = meth.T
    meth_select = meth_trans.loc[:, selected_features]
    sample = pd.read_csv(sample_file, header = 0, index_col = 0, sep = ',')
    return meth_select, sample


def data_preprocess_standardize(meth, sample, dep_name,
                    cov_name = ['SEX_new', 'AGE', 'SMOKING_new', 'DMAGE', 'HBA1C', 'SBP', 'DBP',
                                'CD8T', 'CD4T', 'NK', 'Bcell', 'Mono', 'Gran',
                                'sentrix_code', 'sentrix_pos', 'sample_plate', 'sample_well']):
    '''
    meth: df, methylation matrix, each row is a sample, each column is a CpG site.
    sample: df, sample information matrix, each row is a sample, each column is a clinical variable, containing covariates that need to be adjusted and the dependent variable.
    dep_name: str, name of the dependent variable. In this case, it is either 'eGFR_BASE' or 'eGFR_slope'.
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


def selected_performance_lasso(X, y, plot_name, coef_file):
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
    #print(support)
    support[support!=0] = 1
    support = support.astype(bool)
    #print(support)

    columns = X.columns[support]
    coefs = lasso.coef_.flatten()[support]
    coefs_df = pd.DataFrame(coefs, index=columns)
    coefs_df.to_csv(coef_file, sep = ',', float_format = '%1.15f', header = False, index = True)

    print('After 2nd LASSO: {0} features were selected.'.format(np.sum(support)))
    print('Intercept of the LASSO model is: {0}.'.format(lasso.intercept_))


    predicted = lasso.predict(X)

    r2 = r2_score(y, predicted)
    mse = mean_squared_error(y, predicted)
    corr = np.corrcoef(y, predicted)[1,0]
    scc, scc_p = stats.spearmanr(y, predicted)
    mae = mean_absolute_error(y, predicted)

    plot(y, predicted, plot_name)


    num_params = len(support) + 1 # number of features put in lasso
    aic = calculate_aic(len(y), mse, num_params)
    bic = calculate_bic(len(y), mse, num_params)

    return r2, mse, mae, corr, scc, scc_p, np.sum(support), aic, bic


if __name__ == "__main__":
    dep_name = sys.argv[1]
    ## e.g., eGFR_CKDEPI or eGFR_slope
    data_dir = sys.argv[2]
    ## e.g., '../example_data/'
    meth_file = sys.argv[3]
    ## e.g., '../example_data/meth_eg.txt'
    sample_file = sys.argv[4]
    ## e.g., '../example_data/sample_eg.csv'

    random_state = np.random.RandomState(0)

    out_dir = './output_whole_' + dep_name + '/'
    final_dir = out_dir + 'final_model/'


    if not os.path.exists(final_dir):
        os.makedirs(final_dir)


    selected_feature_file = out_dir + 'final_selected.csv'


    print("=============Loading data=============")
    selected_features = get_selected_features(selected_feature_file)
    meth, sample = load_data(meth_file, sample_file, selected_features)
    print("=============Loading data finished=============")



    performance_file = open(final_dir + 'performance.csv', 'w')
    performance_file.write(','.join(['covariates', 'num_sites', 'r2', 'pcc', 'scc', 'scc_p', 'mse', 'mae', 'AIC', 'BIC']) + '\n')

    for cov in ['with_cov', 'no_cov']:
        print(cov)
        print("=============Data preprocessing=============")
        if cov == 'with_cov':
            X, y = data_preprocess_standardize(meth, sample, dep_name=dep_name)
        else:
            X, y = data_preprocess_standardize(meth, sample, dep_name=dep_name, cov_name = [])
        print("=============Data preprocessing finished=============")

        plot_name = final_dir + 'lasso_plot_' + cov
        coef_file = final_dir + 'lasso_model_' + cov + '.csv'
        r2, mse, mae, corr, scc, scc_p, num_sites, aic, bic = selected_performance_lasso(X, y, plot_name, coef_file)
        performance_file.write(','.join(list(map(str, [cov, num_sites, r2, corr, scc, scc_p, mse, mae, aic, bic]))) + '\n')

    performance_file.close()
