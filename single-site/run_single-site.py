#!/usr/bin/env python

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

from sklearn.decomposition import PCA

import statsmodels.stats.multitest as mttest

import sys
import os



def load_data(meth_file, sample_file):
    meth = pd.read_csv(meth_file, header = 0, index_col = 0, sep = '\t')
    meth_trans = meth.T
    del meth
    sample = pd.read_csv(sample_file, header = 0, index_col = 0, sep = ',')
    meth_trans.dropna(axis = 'columns', how = 'any', inplace=True)
    return meth_trans, sample


def data_preprocess_standardize(meth, sample, dep_name='eGFR_CKDEPI',
                    cov_name = ['SEX_new', 'AGE', 'SMOKING_new', 'DMAGE', 'HBA1C', 'SBP', 'DBP',
                                'CD8T', 'CD4T', 'NK', 'Bcell', 'Mono', 'Gran',
                                'sentrix_code', 'sentrix_pos', 'sample_plate', 'sample_well']):
    """
    meth: df, methylation matrix, each row is a sample, each column is a CpG site.
    sample: df, sample information matrix, each row is a sample, each column is a clinical variable, containing covariates that need to be adjusted and the dependent variable.
    dep_name: str, name of the dependent variable. In this case, it is either 'eGFR_CKDEPI' or 'eGFR_slope'.
    cov_name: list of str, counfounding factors.
"""
    var = sample.loc[:,cov_name]
    ## remove samples with "NA"
    var.dropna(axis = 'index', how = 'any', inplace=True)
    ## remove CpG sites with "NA"
    meth.dropna(axis = 'columns', how = 'any', inplace=True)

    dataX = var.join(meth, how='inner')

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
    print("There are {} samples and {} features.".format(n_samples, n_features))
    return X, y

def process_single_site(X, i, y):
    feature_index = list(range(17)) + [i+17]
    X_new = X.iloc[:, lambda X: feature_index]

    data = X_new.join(y, how='inner')
    n,m = data.shape
    X = data.iloc[:, 0:m-1]
    y = data.iloc[:, m-1]

    assert X.shape[1] == 18, "No. of feature is not correct!"
    X = sm.add_constant(X) # adding a constant
    model = sm.OLS(y, X).fit()
    #print(model.summary())
    y_prime = y.values - np.dot(X.iloc[:, 0:18].values, model.params[0:18]) ## adjusted y by covariates and constant
    corr = np.corrcoef(X.iloc[:, 18].values, y_prime)[0][1]
    return model.params[-1], model.pvalues[-1], model.rsquared, corr



if __name__ == "__main__":
    dep_name = sys.argv[1]
    ## e.g., eGFR_CKDEPI or eGFR_slope
    data_dir = sys.argv[2]
    ## e.g., '../example_data/'
    meth_file = sys.argv[3]
    ## e.g., '../example_data/meth_eg.txt'
    sample_file = sys.argv[4]
    ## e.g., '../example_data/sample_eg.csv'
    res_dir = sys.argv[5]
    ## e.g., './CpG_pvals/'

    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    print("=============Loading data=============")
    meth, sample = load_data(meth_file, sample_file)
    print("=============Loading data finished=============")

    print("=============Data preprocessing=============")
    X, y = data_preprocess_standardize(meth, sample, dep_name)
    print("=============Data preprocessing finished=============")


    lr_metrics = []
    n_sites = meth.shape[1]
    print("Total number of sites: {}".format(n_sites))
    for i in range(n_sites):
        #if i%10000 == 0:
        #    print(i)
        coef, pval, r2, corr = process_single_site(X, i, y)
        lr_metrics.append([coef, pval, r2, corr])


    lr_metrics_df = pd.DataFrame(data=lr_metrics, index=meth.columns, columns=['coef', 'pval', 'r2', 'pcc'])
    lr_metrics_df.sort_values(by='pval', inplace=True)

    fdr_bh = mttest.multipletests(lr_metrics_df['pval'], method='fdr_bh', is_sorted=True)[1]
    bonferroni = mttest.multipletests(lr_metrics_df['pval'], method='bonferroni', is_sorted=True)[1]

    lr_metrics_df['fdr_bh'] = fdr_bh
    lr_metrics_df['bonferroni'] = bonferroni

    #lr_metrics_df.sort_values(by='fdr_bh', inplace=True)
    lr_metrics_df.sort_values(by='pval', inplace=True)

    lr_metrics_df.to_csv(res_dir + 'CpG_lr_cov_' + dep_name + '.csv', sep=',', index=True, header=True)
