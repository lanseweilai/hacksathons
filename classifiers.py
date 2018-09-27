import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn import ensemble
from sklearn import datasets,decomposition,manifold
from sklearn.model_selection import GridSearchCV
import gc
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


####################################normalization##############################

def norm_data(X_train_data,X_test_data,norm_flag):
#1. min_max
    if norm_flag == 1 :
        min_max_scaler = preprocessing.MinMaxScaler()
        X_train_data = min_max_scaler.fit_transform(X_train_data)
        X_test_data = min_max_scaler.fit_transform(X_test_data)
#2. normalization
    if norm_flag == 2 :
        scaler = preprocessing.StandardScaler().fit(X_train_data)
        X_train_data = scaler.transform(X_train_data)
        X_test_data = scaler.transform(X_test_data)
    return X_train_data, X_test_data

######################################pca 降维##############################
def dim_reduction(X_train_data,X_test_data,n_components):
    pca=decomposition.PCA(n_components = n_components)
    pca.fit(X_train_data)
    X_train_data = pca.transform(X_train_data)
    X_test_data = pca.transform(X_test_data)
    return X_train_data, X_test_data

##################################LR_classifier##################################

def LR_clf(X_train_data,Y_train_data):
#　use StratifiedKFold
    stratified_folder = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    C = [ 1e-3, 1e-2, 1e-1, 1, 10, 100,1000,10000]
    print('LR_classifier:')
    LR_C_best = 0
    f1_best = 0
    lr_p_t = []
    lr_r_t = []
    lr_f1_t = []
    for C_item in C:
       lr_p = []
       lr_r = []
       lr_f1 = []
       for X_train_index, X_test_index in stratified_folder.split(X_train_data,Y_train_data):
           lr_classifier = LogisticRegression(
                                   penalty='l2', 
                                   dual=False, 
                                   tol=0.0001, 
                                   C=C_item, 
                                   fit_intercept=True, 
                                   intercept_scaling=1, 
                                   class_weight='balanced',
                                   random_state=None, 
                                   solver='liblinear', 
                                   max_iter=100, 
                                   multi_class='ovr', 
                                   verbose=0, 
                                   warm_start=False, 
                                   n_jobs=1)
           lr_classifier.fit(X_train_data[X_train_index], Y_train_data[X_train_index])
           lr_predictions = lr_classifier.predict(X_train_data[X_test_index])
           lr_p_tmp = precision_score(Y_train_data[X_test_index], lr_predictions)
           lr_p.append(lr_p_tmp)
           lr_p_t.append(lr_p_tmp)
           lr_r_tmp = recall_score(Y_train_data[X_test_index], lr_predictions)
           lr_r.append(lr_r_tmp)
           lr_p_t.append(lr_p_tmp)
           lr_f1_tmp = f1_score(Y_train_data[X_test_index], lr_predictions)
           lr_f1.append(lr_f1_tmp)
           lr_p_t.append(lr_p_tmp)
       print("C_item:%4f,lr_p:%4f,lr_r:%4f,lr_f1:%4f" %(C_item,sum(lr_p)/len(lr_p),sum(lr_r)/len(lr_r),sum(lr_f1)/len(lr_f1)))
       if f1_best< sum(lr_f1)/len(lr_f1):
           LR_C_best = C_item
    return lr_p_t, lr_r_t, lr_f1_t, LR_C_best
#
# ###################################SVM classifier##############################
def SVM_clf(X_train_data,Y_train_data):
    stratified_folder = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    C = [ 1e-1,0.5, 1,2, 10, 100,1000]
    print('SVC_classifier:')
    SVM_C_best = 0
    f1_best = 0
    svm_p_t = []
    svm_r_t = []
    svm_f1_t = []
    for C_item in C:
       svm_p = []
       svm_r = []
       svm_f1 = []
       for X_train_index, X_test_index in stratified_folder.split(X_train_data, Y_train_data):
           svm_classifier = SVC(
                               C=C_item, 
                               kernel='rbf', 
                               gamma='auto', 
                               shrinking=True, 
                               probability=False,
                               tol=0.001, 
                               class_weight='balanced', 
                               verbose=False, 
                               max_iter=-1, 
                               decision_function_shape=None, 
                               random_state=None )
           svm_classifier.fit(X_train_data[X_train_index], Y_train_data[X_train_index])
           svm_predictions = svm_classifier.predict(X_train_data[X_test_index])
           svm_p_tmp = precision_score(Y_train_data[X_test_index], svm_predictions)
           svm_p.append(svm_p_tmp)
           svm_p_t.append(svm_p_tmp)
           svm_r_tmp = recall_score(Y_train_data[X_test_index], svm_predictions)
           svm_r.append(svm_r_tmp)
           svm_r_t.append(svm_r_tmp)
           svm_f1_tmp = f1_score(Y_train_data[X_test_index], svm_predictions)
           svm_f1.append(svm_f1_tmp)
           svm_f1_t.append(svm_f1_tmp)
       print("C_item:%f,svm_p:%f,svm_r:%f,svm_f1:%f" % (
       C_item, sum(svm_p) / len(svm_p), sum(svm_r) / len(svm_r), sum(svm_f1) / len(svm_f1)))
       if f1_best< sum(svm_f1)/len(svm_f1):
           SVM_C_best = C_item
    return svm_r_t, svm_p_t, svm_f1_t, SVM_C_best

# ################################gbdt classifier + lr############################
def gbdt_lr_clf(X_train_data,Y_train_data):
    n_estimators = [10,20,30,40,50]
    estimator_best = 0
    f1_best = 0
    gblr_p_t = []
    gblr_r_t = []
    gblr_f1_t = []
    for item in n_estimators:
        grd = ensemble.GradientBoostingClassifier(n_estimators = item)
        stratified_folder = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
        print('gbdt_classifier + LR:')
        gblr_p = []
        gblr_r = []
        gblr_f1 = []
        for X_train_index, X_test_index in stratified_folder.split(X_train_data, Y_train_data):
            X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train_data[X_train_index], Y_train_data[X_train_index], test_size=0.5)
            grd_enc = OneHotEncoder()
            grd_lm = LogisticRegression()
            grd.fit(X_train, y_train)
            grd_enc.fit(grd.apply(X_train)[:, :, 0])
            grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
            y_pred_grd_lm = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_train_data[X_test_index])[:, :, 0]))[:, 1]
            y_pred_grd_lm = (y_pred_grd_lm>=0.5)*1
            gblr_p_tmp = precision_score(Y_train_data[X_test_index], y_pred_grd_lm)
            gblr_p.append(gblr_p_tmp)
            gblr_p_t.append(gblr_p_tmp)
            gblr_r_tmp = recall_score(Y_train_data[X_test_index], y_pred_grd_lm)
            gblr_r.append(gblr_r_tmp)
            gblr_r_t.append(gblr_r_tmp)
            gblr_f1_tmp = f1_score(Y_train_data[X_test_index], y_pred_grd_lm)
            gblr_f1.append(gblr_f1_tmp)
            gblr_f1_t.append(gblr_f1_tmp)
        print("n_estimators:%f,gblr_p:%f,gblr_r:%f,gblr_f1:%f" %
              (item,sum(gblr_p) / len(gblr_p), sum(gblr_r) / len(gblr_r), sum(gblr_f1) / len(gblr_f1)))
        if f1_best < sum(gblr_f1) / len(gblr_f1):
            estimator_best = item
    return gblr_r_t, gblr_p_t, gblr_f1_t,estimator_best
##################################xgboost####################################
# 每一步都手动调节参数比较好，xgboost对电脑性能要求较高

###调节参数n_estimators##########
def xgb_clf(X_train_data,Y_train_data):

    cv_params = {'n_estimators':[50,60,70,80,90,100,110,120]}
    other_params = {
         'booster':'gbtree',
         'objective': 'binary:logistic',
         'learning_rate':0.1,
         'n_estimators':100,
         'max_depth':5,
         'min_child_weight':1,
         'seed':0,
         'subsample':0.8,
         'colsample_bytree':0.8,
         'gama':0,
         'seed':0,
         'nthread':8,
          'silent':1 }
    model = xgb.XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train_data, Y_train_data)
    evalute_result = optimized_GBM.cv_results_
    n_estimators_best = optimized_GBM.best_params_["n_estimators"]
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    del optimized_GBM
    gc.collect()

###调节参数max_depth & min_child_weight######
    cv_params = {'max_depth':[3,4,5,6,7,8,9,10,11],
                 'min_child_weight':[1,2,3,4,5,6,7]}
    other_params = {
      'booster':'gbtree',
      'objective': 'binary:logistic',
      'learning_rate':0.1,
      'n_estimators':n_estimators_best,
      'max_depth':5,
      'min_child_weight':1,
      'seed':0,
      'subsample':0.8,
      'colsample_bytree':0.8,
      'gama':0,
      'seed':0,
      'nthread':8,
       'silent':1 }
    model = xgb.XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train_data, Y_train_data)
    evalute_result = optimized_GBM.cv_results_
    max_depth_best = optimized_GBM.best_params_["max_depth"]
    min_child_weight_best = optimized_GBM.best_params_["min_child_weight"]
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    ##最佳4,3

#########调节参数gamma###############
    cv_params = {'gamma':[0.1,0.2,0.3,0.4,0.5,0.6]}
    other_params = {
         'booster':'gbtree',
         'objective': 'binary:logistic',
         'learning_rate':0.1,
         'n_estimators':n_estimators_best,
         'max_depth':max_depth_best,
         'min_child_weight':min_child_weight_best,
         'seed':0,
         'subsample':0.8,
         'colsample_bytree':0.8,
         'gama':0,
         'seed':0,
         'nthread':8,
          'silent':1 }
    model = xgb.XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train_data, Y_train_data)
    evalute_result = optimized_GBM.cv_results_
    gama_best = optimized_GBM.best_params_["gama"]
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

#########调节参数subsample & colsample_bytree####
    cv_params = {'subsample':[0.6,0.7,0.8,0.9],
                'colsample_bytree':[0.6,0.7,0.8,0.9]}
    other_params = {
         'booster':'gbtree',
         'objective': 'binary:logistic',
         'learning_rate':0.1,
         'n_estimators':n_estimators_best,
         'max_depth':max_depth_best,
         'min_child_weight':min_child_weight_best,
         'seed':0,
         'subsample':0.9,
         'colsample_bytree':0.6,
         'gama':gama_best,
         'seed':0,
         'nthread':8,
          'silent':1 }
    model = xgb.XGBClassifier(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='f1', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(X_train_data, Y_train_data)
    evalute_result = optimized_GBM.cv_results_
    subsample_best = optimized_GBM.best_params_["subsample"]
    colsample_bytree_best = optimized_GBM.best_params_["colsample_bytree"]
    print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))
    #最佳0.9,0.6

################最终结果############
    stratified_folder = StratifiedKFold(n_splits=5, random_state=0, shuffle=False)
    print('xgboost_classifier:')
    clf_p = []
    clf_r = []
    clf_f1 = []

    for X_train_index, X_test_index in stratified_folder.split(X_train_data, Y_train_data):
       dtrain = xgb.DMatrix(X_train_data[X_train_index], Y_train_data[X_train_index])
       dtest = xgb.DMatrix(X_train_data[X_test_index])
       params = {'booster':'gbtree',
         'objective': 'binary:logistic',
         'learning_rate':0.1,
         'n_estimators':n_estimators_best,
         'max_depth':max_depth_best,
         'min_child_weight':min_child_weight_best,
         'seed':0,
         'subsample':subsample_best,
         'colsample_bytree':colsample_bytree_best,
         'gama':0.6,
         'seed':0,
         'nthread':8,
          'silent':1}
       watchlist = [(dtrain,'train')]
       bst = xgb.train(params,dtrain,num_boost_round=100,evals=watchlist)
       clf_predictions = bst.predict(dtest)
       clf_predictions = (clf_predictions >= 0.5) * 1
       clf_p_tmp = precision_score(Y_train_data[X_test_index], clf_predictions)
       clf_p.append(clf_p_tmp)
       clf_r_tmp = recall_score(Y_train_data[X_test_index], clf_predictions)
       clf_r.append(clf_r_tmp)
       clf_f1_tmp = f1_score(Y_train_data[X_test_index], clf_predictions)
       clf_f1.append(clf_f1_tmp)
    print("clf_p:%f,clf_r:%f,clf_f1:%f" % (sum(clf_p) / len(clf_p), sum(clf_r) / len(clf_r), sum(clf_f1) / len(clf_f1))
        )
    return clf_p,clf_r,clf_f1,n_estimators_best,max_depth_best,min_child_weight_best,subsample_best,colsample_bytree_best

def lr_test(X_train,Y_train,X_test,lr_norm,dim_para):
    '''
    :param X_train: train data
    :param Y_train: train label
    :param X_test:  test data 说明卡号在第一类，第二列之后才是数据
    :param lr_norm: 0: 不进行归一化  1：归一化（0,1） 2：标准化
    :param dim_para: PCA 降维 0:不进行降维 其他：降维
    :return: 输出卡号
    '''
    print("LR_model:")
    X_test_org = X_test
    X_test = X_test_org[:,1:]
    if lr_norm == 0:
        pass
    if lr_norm == 1 or 2 :
        X_train, X_test = norm_data(X_train, X_test, lr_norm)

    if dim_para == 0:
        pass
    if dim_para != 0:
        X_train, X_test = dim_reduction(X_train, X_test, dim_para)

    lr_p, lr_r, lr_f1, LR_C_best = LR_clf(X_train, Y_train)
    lr_classifier = LogisticRegression(
        penalty='l2',
        dual=False,
        tol=0.0001,
        C=LR_C_best,
        fit_intercept=True,
        intercept_scaling=1,
        class_weight='balanced',
        random_state=None,
        solver='liblinear',
        max_iter=100,
        multi_class='ovr',
        verbose=0,
        warm_start=False,
        n_jobs=1)
    lr_classifier.fit(X_train, Y_train)
    lr_predictions = lr_classifier.predict(X_test)
    lr_label = (lr_predictions>=0.5) * 1
    print("LR predict positive label number: %d" % (sum(lr_label == 1)))
    card_list = X_test_org[lr_label==1,0]
    return card_list

def SVM_test(X_train,Y_train,X_test,svm_norm,dim_para):
    '''
    :param X_train: train data
    :param Y_train: train label
    :param X_test:  test data 说明卡号在第一类，第二列之后才是数据
    :param svm_norm: 0: 不进行归一化  1：归一化（0,1） 2：标准化
    :param dim_para: PCA 降维 0:不进行降维 其他：降维
    :return: 输出卡号
    '''
    print('SVC_model:')
    X_test_org = X_test
    X_test = X_test_org[:,1:]
    if svm_norm == 0:
        pass
    if svm_norm == 1 or 2 :
        X_train, X_test = norm_data(X_train, X_test, svm_norm)

    if dim_para == 0:
        pass
    if dim_para != 0:
        X_train, X_test = dim_reduction(X_train, X_test, dim_para)

    svm_r, svm_p, svm_f1, SVM_C_best = SVM_clf(X_train, Y_train)
    svm_classifier = SVC(
            C=SVM_C_best,
            kernel='rbf',
            gamma='auto',
            shrinking=True,
            probability=False,
            tol=0.001,
            class_weight='balanced',
            verbose=False,
            max_iter=-1,
            decision_function_shape=None,
            random_state=None)
    svm_classifier.fit(X_train, Y_train)
    svm_predictions = svm_classifier.predict(X_test)
    svm_label = (svm_predictions>=0.5) * 1
    print("svm predict positive label number: %d" % (sum(svm_label == 1)))
    card_list = X_test_org[svm_label==1,0]
    return card_list

def GBDT_LR_test(X_train,Y_train,X_test,glr_norm,dim_para):
    '''
    
    :param X_train: train data
    :param Y_train: train label
    :param X_test:  test data 说明卡号在第一类，第二列之后才是数据
    :param glr_norm: 0: 不进行归一化  1：归一化（0,1） 2：标准化
    :param dim_para: PCA 降维 0:不进行降维 其他：降维
    :return: 输出卡号
    '''

    print("GBDT_LR_model:")
    X_test_org = X_test
    X_test = X_test_org[:, 1:]
    if glr_norm == 0:
        pass
    if glr_norm == 1 or 2:
        X_train, X_test = norm_data(X_train, X_test, glr_norm)

    if dim_para == 0:
        pass
    if dim_para != 0:
        X_train, X_test = dim_reduction(X_train, X_test, dim_para)

    gblr_r, gblr_p, gblr_f1, estimator_best = gbdt_lr_clf(X_train, Y_train)
    grd = ensemble.GradientBoostingClassifier(n_estimators = estimator_best)
    X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train, Y_train, test_size=0.5)
    grd_enc = OneHotEncoder()
    grd_lm = LogisticRegression()
    grd.fit(X_train, y_train)
    grd_enc.fit(grd.apply(X_train)[:, :, 0])
    grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)
    gblr_predictions = grd_lm.predict_proba(grd_enc.transform(grd.apply(X_test)[:, :, 0]))[:, 1]
    gblr_predictions = (gblr_predictions>=0.5)*1
    gblr_label = (gblr_predictions>=0.5) * 1
    print("gbdt_lr predict positive label number: %d" % (sum(gblr_label == 1)))
    card_list = X_test_org[gblr_label == 1,0]
    return card_list

def XGBoost_test(X_train, Y_train, X_test, xgb_norm, dim_para):
    '''

    :param X_train: train data
    :param Y_train: train label
    :param X_test:  test data 说明卡号在第一类，第二列之后才是数据
    :param xgb_norm: 0: 不进行归一化  1：归一化（0,1） 2：标准化
    :param dim_para: PCA 降维 0:不进行降维 其他：降维
    :return: 输出卡号
    '''

    print("XGBoost_model:")
    X_test_org = X_test
    X_test = X_test_org[:, 1:]
    if xgb_norm == 0:
        pass
    if xgb_norm == 1 or 2:
        X_train, X_test = norm_data(X_train, X_test, xgb_norm)

    if dim_para == 0:
        pass
    if dim_para != 0:
        X_train, X_test = dim_reduction(X_train, X_test, dim_para)

    clf_p, clf_r, clf_f1, n_estimators_best, max_depth_best, min_child_weight_best, subsample_best, colsample_bytree_best = \
        xgb_clf(X_train, Y_train)

    dtrain = xgb.DMatrix(X_train, Y_train)
    dtest = xgb.DMatrix(X_test)
    params = {'booster': 'gbtree',
              'objective': 'binary:logistic',
              'learning_rate': 0.1,
              'n_estimators': n_estimators_best,
              'max_depth': max_depth_best,
              'min_child_weight': min_child_weight_best,
              'seed': 0,
              'subsample': subsample_best,
              'colsample_bytree': colsample_bytree_best,
              'gama': 0.6,
              'seed': 0,
              'nthread': 8,
              'silent': 1}
    watchlist = [(dtrain, 'train')]
    bst = xgb.train(params, dtrain, num_boost_round=100, evals=watchlist)
    clf_predictions = bst.predict(dtest)
    clf_predictions = (clf_predictions >= 0.5) * 1
    clf_predictions = (clf_predictions >= 0.5) * 1
    clf_label = (clf_predictions >= 0.5) * 1
    print("XGBoost predict positive label number: %d" % (sum(clf_label == 1)))
    card_list = X_test_org[clf_label == 1, 0]
    return card_list
    

