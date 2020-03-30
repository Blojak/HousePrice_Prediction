# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:10:24 2020

@author: benja

Find competition and data under the following link:
    https://www.kaggle.com/c/house-prices-advanced-regression-techniques
"""


# Load all libraries
import seaborn as sb
# sb.set_style('darkgrid')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import norm   #for some statistics
from sklearn.linear_model import LassoCV

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import validation_curve


# =============================================================================
# Load Data
# =============================================================================
def get_data():
    #get train data
    train_data_path ='data/train.csv'
    train = pd.read_csv(train_data_path)
    #get test data
    test_data_path ='data/test.csv'
    test = pd.read_csv(test_data_path)    
    return train , test


def get_combined_data():
  #reading train data
  train , test = get_data()
  target = train.SalePrice # define the variable to predict 
  train.drop(['SalePrice'],axis = 1 , inplace = True)
  combined = train.append(test)
  combined.reset_index(inplace=True)
  combined.drop(['index', 'Id'], inplace=True, axis=1)
  return combined, target

#Load train and test data into pandas DataFrames
train_data, test_data = get_data()

#Combine train and test data to process them together in order check their
# statistical properties over the entire set
combined, target = get_combined_data()


# =============================================================================
# Missing values
# =============================================================================
# Handle categorial variables manually

combined.loc[:, "MasVnrType"]       = combined.loc[:, "MasVnrType"].fillna("None")
combined.loc[:, "GarageQual"]       = combined.loc[:, "GarageQual"].fillna("No")
combined.loc[:, "GarageCond"]       = combined.loc[:, "GarageCond"].fillna("No")
combined.loc[:, "Utilities"]        = combined.loc[:, "Utilities"].fillna("AllPub")
combined.loc[:, "SaleCondition"]    = combined.loc[:, "SaleCondition"].fillna("Normal")
combined.loc[:, "KitchenQual"]      = combined.loc[:, "KitchenQual"].fillna("TA")
combined.loc[:, "BsmtExposure"]     = combined.loc[:, "BsmtExposure"].fillna("No")
combined.BsmtFinSF1.astype('str')
combined.loc[:, 'MSZoning']         = combined.loc[:, 'MSZoning'].fillna('RH')
combined.loc[:, 'SaleType']         = combined.loc[:, 'SaleType'].fillna('Oth')
combined.loc[:, 'Exterior1st']      = combined.loc[:, 'Exterior1st'].fillna('Other')


# Fill the NaNs of metrical variables with zeros
for col in ('LotFrontage', 'MasVnrArea', 'OpenPorchSF','GarageArea',
            'GarageCars', 'HalfBath', 'WoodDeckSF', 'ScreenPorch','KitchenAbvGr',
            'TotalBsmtSF', 'BsmtUnfSF', 'BsmtFinSF1', "BsmtFinSF2"):
    combined[col] = combined[col].fillna(0)
del col   


# Assort according to num and cat
def assort_cols(df):
    num_cols = df.select_dtypes(exclude=['object']).columns
    num_cols = num_cols.tolist()
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_cols = cat_cols.tolist()
    df1 = df[num_cols + cat_cols]
    return num_cols, cat_cols, df1
[num_cols, cat_cols, combined] = assort_cols(combined)


# identify the number of observations in the dependent data set
training_data_len       = len(target)


# =============================================================================
# Group some categories of some variables
# =============================================================================
# If there are too many categories, group some and ecode new
combined = combined.replace({'SaleCondition' : {'AdjLand' : 1, 'Abnorml' : 2, 'Family' :2,
                                                'Alloca' :3, 'Normal':3, 'Partial': 4}})

combined = combined.replace({'MSZoning' : {'C (all)' : 1, 'RM' : 2, 'RH' :2,
                                                'RL' :3, 'FV':3}})
# encode 'Neighborhood' as ordinale scaled var (I choose 4 categories)
combined = combined.replace({'Neighborhood' : {'MeadowV' : 1, 'BrDale' : 1, 'IDOTRR' : 1,
                                                'BrkSide' : 1, 'Edwards' : 1, 'OldTown' : 1, 'Sawyer' : 1,
                                                'Blueste' : 1, 'SWISU' : 1, 'NPkVill' : 1, 'NAmes' : 1, 
                                                'Mitchel' : 2, 'SawyerW' : 2, 'NWAmes' : 2, 'Gilbert' : 2,
                                                'Blmngtn' : 2, 'CollgCr' : 2, 'Crawfor' : 3, 'ClearCr' : 3,
                                                'Somerst' : 3, 'Veenker' : 3, 'Timber' : 3, 'StoneBr': 4, 'NridgHt':4,
                                                'NoRidge' : 4}})

# =============================================================================
# Age of the building
# =============================================================================

combined['Age'] = combined['YearBuilt'].max()+1 - combined['YearBuilt']
combined['Years_sold'] = combined['YrSold'].max()+1 - combined['YearBuilt']
combined['Years_since_remod'] = combined['YrSold'] - combined['YearRemodAdd']
combined.drop(['YearBuilt', 'YearRemodAdd','YrSold'], inplace=True, axis = 1)

# =============================================================================
# Remove obvious Outliers
# =============================================================================
combined[combined.GrLivArea > 4000] = 4000

# =============================================================================
# Drop some vars (which do not show a clear explanatory power)
# =============================================================================
combined.drop(['Alley', 'FireplaceQu', 'PoolQC', 'Fence', 'MiscFeature', 'BsmtFullBath',
               'BsmtHalfBath', 'GarageYrBlt', 'GarageType', 'GarageFinish', 'BsmtQual',
               'BsmtFinType1', 'BsmtFinType2','BsmtCond', 'MoSold'],
              inplace=True, axis=1)


# =============================================================================
# Delete some rows (with single missing values)
# =============================================================================
def where_nan_index(df, col_nam): 
    x = df[col_nam].isnull().values
    x = np.where(x==True)
    x = np.array(x)
    x = x.T
    return x
combined.loc[:, 'Functional']      = combined.loc[:, 'Functional'].fillna('Typ')
combined.loc[:, 'Electrical']      = combined.loc[:, 'Electrical'].fillna('SBrkr')


training_data_len   = target.shape[0]

# del El_nan, Functional, t_nan


# =============================================================================
# Combine Variables (in a suitable manner)
# =============================================================================
combined['Porch']       = combined['OpenPorchSF'] + combined['EnclosedPorch']+ combined['3SsnPorch']+combined['ScreenPorch']
combined.drop(['OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'Street'], inplace = True, axis=1)

combined['BathroomNo']  = combined['FullBath']+combined['HalfBath']

combined.drop(['BsmtFinSF1','BsmtFinSF2', 'LowQualFinSF', 'PoolArea',
               'FullBath', 'HalfBath'], inplace = True, axis=1)

# Assort again
[num_cols, cat_cols, combined] = assort_cols(combined)

# =============================================================================
# One-Hot-Encoder
# =============================================================================
combined_cat        = pd.get_dummies(combined[cat_cols])
combined_cat_col    = combined_cat.columns

combined_num        = combined[num_cols]
combined_num_col    = combined_num.columns

combined_hot        = pd.concat([combined_num, combined_cat], axis = 1)
comb_hot_columns    = combined_hot.columns




# =============================================================================
# Features - corr
# =============================================================================
# Take logs of the dependent variable (due to skewness)
target = np.log(target)

training_data_len = len(target)
# =============================================================================
# Pre-Processing -  Standardization
# =============================================================================
scaler_x                                   = StandardScaler()
scaler_y                                   = StandardScaler()

X                                          = combined_hot.values
# Standardize the metric variables not the dummies
X[:,:np.array(combined_num_col).shape[0]]  = scaler_x.fit_transform(X[:,:np.array(combined_num_col).shape[0]])

X_train   =     X[:training_data_len,:]
target  = np.array(target).reshape(-1,1)
y_train   = scaler_y.fit_transform(target)

# Split standardized '(test)Data' into test and trainings data
# tscv = TimeSeriesSplit()
# for train_index, test_index in tscv.split(X_std):
#     X_train, X_test = X_std[train_index,:], X_std[test_index,:]
#     y_train, y_test = y_std[train_index], y_std[test_index]
    

# =============================================================================
# Lasso Regression -- Feature Selection
# =============================================================================

lasso               = LassoCV(cv=10, random_state=0).fit(X_train, y_train)
lasso_score         = lasso.score(X_train, y_train)

y_pred_lasso_train  = lasso.predict(X_train).reshape(-1,1)
# y_pred_lasso_test   = lasso.predict(X_test).reshape(-1,1)

lasso_resid_train   = y_train-  y_pred_lasso_train
# lasso_resid_test    = y_test -  y_pred_lasso_test

rmse_lasso = np.sqrt(np.mean(y_pred_lasso_train - y_train)**2)

# get coefs
lasso_coefs         = lasso.coef_
lasso_coef_index    = np.array(np.nonzero(lasso_coefs)).T
lasso_coef_names    = comb_hot_columns[lasso_coef_index].values


# =============================================================================
# Select features according to the Lasso F-Selection
# =============================================================================

combined_sel    = combined_hot[lasso_coef_names[:,0]]
col_dummy       = combined_sel.select_dtypes(include=['uint8']).columns
col_nondummy    = combined_sel.select_dtypes(exclude=['uint8']).columns


scaler_x                                   = StandardScaler()
scaler_y                                   = StandardScaler()

X                                          = combined_sel.values
# Standardize the metric regressors
X[:,:np.array(col_nondummy).shape[0]]      = scaler_x.fit_transform(X[:,:np.array(col_nondummy).shape[0]])

X_train             =     X[:training_data_len,:]
target              = np.array(target).reshape(-1,1)
y_train             = scaler_y.fit_transform(target)

# Split standardized '(test)Data' into test and trainings data
# --> Cross Validated
# tscv = TimeSeriesSplit()
# for train_index, test_index in tscv.split(X):
#     X_train, X_test = X[train_index,:], X[test_index,:]
#     y_train, y_test = y[train_index], y[test_index]


# =============================================================================
# Random Forest Regression
# =============================================================================

# Cross Validate the number of trees in the forest and the max_debth
param_est = [np.arange(1,30,2)][0]
forest_debth = 2
forest_train_score, forest_valid_score = validation_curve(
                                RandomForestRegressor(criterion = 'mse',max_depth=forest_debth),
                                X = X_train, y = y_train, 
                                param_name = 'n_estimators', 
                                param_range = param_est, cv = 10)
# ,max_depth=10
forest_train_score_mean = np.mean(forest_train_score, axis = 1)
forest_train_score_std = np.std(forest_train_score, axis = 1)

forest_valid_score_mean = np.mean(forest_valid_score, axis = 1)
forest_valid_score_std = np.std(forest_valid_score, axis = 1)

# Plot the Validation Curve
fig, ax = plt.subplots()
ax.set_title('Validation Curve with RF')
ax.set_label(r'param')
ax.set_ylabel("Score")
ax.set_ylim(0.0, 1.1)
lw = 2
ax.plot(param_est, forest_train_score_mean, label="Training score",
             color="darkorange", lw=lw)
ax.fill_between(param_est, forest_train_score_mean - forest_train_score_std,
                 forest_train_score_mean + forest_train_score_std, alpha=0.2,
                 color="darkorange", lw=lw)
ax.plot(param_est, forest_valid_score_mean, label="Cross-validation score",
             color="navy", lw=lw)
ax.fill_between(param_est, forest_valid_score_mean - forest_valid_score_std ,
                 forest_valid_score_mean+ forest_valid_score_std , alpha=0.2,
                 color="navy", lw=lw)
ax.legend(loc="best")
# ax.set_ylim([0.7,1])
fig.savefig('figures/RF_CV_Validation_Curve.eps')
fig.show()

# =============================================================================
# RF Estimation
# =============================================================================
# Choose the number of trees that maximized the r2 in the CV
max_val = np.argmax(forest_train_score_mean)

# define the model
forest = RandomForestRegressor(n_estimators = max_val,
                               criterion = 'mse',
                               max_depth=forest_debth,
                               random_state = 1, 
                               n_jobs = 1)
# Train the model
forest.fit(X_train , y_train)

forest_score              = forest.score(X_train , y_train)

# y_pred_test_forest        = forest.predict(X_test).reshape(-1,1)

y_pred_train_forest       = forest.predict(X_train).reshape(-1,1)


# lr_resid_test_forest      = y_test -y_pred_test_forest
lr_resid_train_forest     = y_train - y_pred_train_forest


rmse_lr_forest = np.sqrt(np.mean(y_pred_train_forest  - y_train)**2)

'''
Even though the ordinary Lasso regression delivers a better R^2, the additional
Random Forest Regression is useful to obtain a better RMSE. The R^2 metric is 
probably not the best choice for cross-sectional data
'''
train_data, test_data = get_data()


y_prediction_forest      = forest.predict(X[training_data_len:,:]).reshape(-1,1)
y_forest = np.exp(scaler_y.inverse_transform(y_prediction_forest))

predictions = pd.DataFrame(np.concatenate((np.arange(1461,1461+y_forest.shape[0]).reshape(-1,1), y_forest),  axis = 1)
                           , columns = ['Id', 'SalePrice'])
predictions['Id'] = predictions['Id'].astype('int32')

predictions.to_csv('D:/Dropbox/Work/Python Scripts/MyProjects/HousePrice/HousePricesubmission.csv', index = False)






