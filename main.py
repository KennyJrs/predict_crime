import sys

import requests

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model._cd_fast import ConvergenceWarning
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_regression
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, zero_one_loss, precision_score, \
     recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def emb_methods(dataframe):
    X = dataframe.drop(columns=['crm_cd_desc', 'crm_cd'])
    y = dataframe['crm_cd_desc']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)
    X_train.columns = [x.capitalize() for x in X_train.columns]
    X_test.columns = [x.capitalize() for x in X_test.columns]
    X_train, X_test = X_train.drop(columns=['Dr_no', 'Rpt_dist_no', 'Status', 'Crm_cd_1', 'Crm_cd_2',
                                                     'Crm_cd_3']), X_test.drop(columns=['Dr_no',
                                                     'Rpt_dist_no', 'Status', 'Crm_cd_1', 'Crm_cd_2', 'Crm_cd_3'])
    X_train_dt = X_train[:2000]
    X_test_dt = X_test[:2000]
    y_train_dt = y_train[:2000]
    y_test_dt = y_test[:2000]

    dt_fs = SelectFromModel(DecisionTreeClassifier(random_state=5), threshold='median')
    dt_fs.fit(X_train_dt, y_train_dt)
    dt_selected = list(X_train_dt.columns[(dt_fs.get_support())])
    dt_importances = pd.Series(dt_fs.estimator_.feature_importances_)

    rf_fs = SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=5), threshold='median')
    rf_fs.fit(X_train, y_train)
    rf_selected = list(X_train.columns[(rf_fs.get_support())])
    rf_importances = pd.Series(rf_fs.estimator_.feature_importances_)

    print(' Features selected by Decision Tree:', dt_selected)
    print(' Features selected by Random Forest:', rf_selected)

    for_viz = pd.DataFrame()
    for_viz['Features'] = X_train.columns
    for_viz['Dec_tree'] = dt_importances
    for_viz['Ran_forest'] = rf_importances

    plt.figure(figsize=(13, 9))
    bar_width = 0.35
    index = np.arange(len(for_viz['Features']))
    plt.bar(index, for_viz['Dec_tree'], bar_width, label='Dec_tree', color='#96B5DE')
    plt.bar(index + bar_width, for_viz['Ran_forest'], bar_width, label='Ran_forest', color='#F2DA8D')
    plt.xlabel('Features', fontsize=14, labelpad=8)
    plt.ylabel('Importance', fontsize=14, labelpad=5)
    plt.title('Feature importances using Decision Tree and Random Forest', fontsize=16, pad=14)
    plt.xticks(index + bar_width / 2, for_viz['Features'], rotation=45)
    plt.legend()
    plt.show()
    return dt_selected, rf_selected

def flt_methods(dataframe):
    X = dataframe.drop(columns=['crm_cd', 'crm_cd_desc'])
    y = dataframe['crm_cd']
    X.columns = [x.capitalize() for x in X.columns]
    X = X.drop(columns=['Dr_no', 'Rpt_dist_no', 'Status', 'Crm_cd_1', 'Crm_cd_2', 'Crm_cd_3'])

    fs = SelectKBest(score_func=f_regression, k=7)
    x_selected = fs.fit_transform(X, y)
    selected_features = fs.get_support()
    print(' Features selected by Pearson method:', list(X.columns[selected_features]))

    corr_matrix = X.corr(method='pearson')
    f, ax1 = plt.subplots(figsize=(13, 9))
    sns.heatmap(corr_matrix, annot=True, linewidths=.5, ax=ax1)
    ax1.set_title('Correlation matrix by Pearson\'s method', fontsize=16, pad=14)
    plt.show()
    return list(X.columns[selected_features])
def feature_selection(dataframe):
    print('Feature selection:')
    flt_selected = flt_methods(dataframe)
    dt_selected, rf_selected = emb_methods(dataframe)
    features_selected = set(flt_selected) & set(dt_selected) & set(rf_selected)
    print(' Features selected by three methods:', features_selected)
    print(' Amount of features selected by three methods:', len(features_selected), '\n')
    print(' Feature importances and the correlation coefficients are small enough to make a decision based on them.')
    print(' But some of the following columns will be dropped for each classification and regression method: Dr_no, Rpt_dist_no, Status, Crm_cd_2, Crm_cd_3\n')

def knn_classifier(dataframe):
    X = dataframe.drop(columns=['crm_cd', 'crm_cd_desc'])
    y = dataframe['crm_cd']
    X.columns = [x.capitalize() for x in X.columns]
    X = X.drop(columns=['Dr_no', 'Rpt_dist_no', 'Status', 'Crm_cd_2', 'Crm_cd_3'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    knn = KNeighborsClassifier(n_neighbors=15, weights='distance')
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    over_sampler = RandomOverSampler(sampling_strategy='minority')
    under_sampler = RandomUnderSampler(sampling_strategy='majority')
    model_with_sampling = Pipeline([('over_sampler', over_sampler), ('under_sampler', under_sampler), ('lr', knn)])
    model_with_sampling.fit(X_train, y_train)
    y_pred_sampling = model_with_sampling.predict(X_test)

    print('KNN:')
    df_metrics = pd.DataFrame()
    df_metrics['Metrics'] = ['Accuracy', 'Missclassification', 'Precision', 'Recall', 'F1-score']
    df_metrics['No_sampling'] = ['{0:.4f}'.format(accuracy_score(y_test, y_pred)),
                                 '{0:.4f}'.format(zero_one_loss(y_test, y_pred)),
                                 '{0:.4f}'.format(precision_score(y_test, y_pred, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(recall_score(y_test, y_pred, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(f1_score(y_test, y_pred, average='weighted'))]
    df_metrics['Sampling'] = ['{0:.4f}'.format(accuracy_score(y_test, y_pred_sampling)),
                                 '{0:.4f}'.format(zero_one_loss(y_test, y_pred_sampling)),
                                 '{0:.4f}'.format(precision_score(y_test, y_pred_sampling, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(recall_score(y_test, y_pred_sampling, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(f1_score(y_test, y_pred_sampling, average='weighted'))]
    df_metrics.set_index('Metrics', inplace=True)
    print(' ', df_metrics, '\n')

    print('Counting CV scores...')
    k_folds = KFold(n_splits=10)
    scores = cross_val_score(knn, X, y, cv=k_folds)
    print(' Cross Validation Scores: ', list(scores))
    print(' Average CV Score: ', '{0:.4f}'.format(scores.mean()))
    print(' Deviation from average cv score:', '{0:.4f}'.format(accuracy_score(y_test, y_pred) - scores.mean()), '\n')

    print('Calculating perfect k..')
    perfect_k = 0.0
    perfect_k_number = 0
    for k in range(1, 30):
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, X_train, y_train, cv=5)
        if scores.mean() > perfect_k:
            perfect_k = scores.mean()
            perfect_k_number = k
    print(f' The perfect k is {perfect_k_number} with accuracy =', '{0:.4f}'.format(perfect_k), '\n')
    if (float(df_metrics.loc['Accuracy', 'No_sampling']) - float(df_metrics.loc['Accuracy', 'Sampling'])) > 0:
        return [df_metrics.loc['Accuracy', 'No_sampling'], 'No_sampling knn']
    elif (float(df_metrics.loc['Accuracy', 'No_sampling']) - float(df_metrics.loc['Accuracy', 'Sampling'])) < 0:
        return [df_metrics.loc['Accuracy', 'Sampling'], 'Sampling knn']
    else:
        return [df_metrics.loc['Accuracy', 'Sampling'], 'knn']

def rf_classifier(dataframe):
    X = dataframe.drop(columns=['crm_cd', 'crm_cd_desc'])
    y = dataframe['crm_cd_desc']
    X.columns = [x.capitalize() for x in X.columns]
    X = X.drop(columns=['Dr_no'])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)

    rf_model = RandomForestClassifier(n_estimators=50, random_state=5)
    rf_model.fit(X_train, y_train)
    y_pred = rf_model.predict(X_test)
    print('Random forest:')
    c1 = classification_report(y_test, y_pred, zero_division=1)

    over_sampler = RandomOverSampler(sampling_strategy='minority')
    under_sampler = RandomUnderSampler(sampling_strategy='majority')
    model_with_sampling = Pipeline([('over_sampler', over_sampler), ('under_sampler', under_sampler), ('lr', rf_model)])
    model_with_sampling.fit(X_train, y_train)
    y_pred_sampling = model_with_sampling.predict(X_test)
    c2 = classification_report(y_test, y_pred_sampling, zero_division=1)

    c1_lines = c1.split('\n')
    c1_class_names = []
    c1_class_recall = []
    for line in c1_lines[2:-4]:
        values = line.split()
        if len(values) >= 3:
            c1_class_names.append(values[:-4])
            c1_class_recall.append(float(values[-3]))

    c2_lines = c2.split('\n')
    c2_class_names = []
    c2_class_recall = []
    for line in c2_lines[2:-4]:
        values2 = line.split()
        if len(values2) >= 3:
            c2_class_names.append(values2[:-4])
            c2_class_recall.append(float(values2[-3]))

    df_metrics = pd.DataFrame()
    df_metrics['Metrics'] = ['Accuracy', 'Missclassification', 'Precision', 'Recall', 'F1-score']
    df_metrics['No_sampling'] = ['{0:.4f}'.format(accuracy_score(y_test, y_pred)),
                                 '{0:.4f}'.format(zero_one_loss(y_test, y_pred)),
                                 '{0:.4f}'.format(precision_score(y_test, y_pred, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(recall_score(y_test, y_pred, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(f1_score(y_test, y_pred, average='weighted'))]
    df_metrics['Sampling'] = ['{0:.4f}'.format(accuracy_score(y_test, y_pred_sampling)),
                                 '{0:.4f}'.format(zero_one_loss(y_test, y_pred_sampling)),
                                 '{0:.4f}'.format(precision_score(y_test, y_pred_sampling, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(recall_score(y_test, y_pred_sampling, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(f1_score(y_test, y_pred_sampling, average='weighted'))]
    df_metrics.set_index('Metrics', inplace=True)
    print(' ', df_metrics, '\n')
    sum_c1, sum_c2, sum_ = 0.0, 0.0, 0.0
    for s in c1_class_recall:
        sum_c1 += s
    for s in c2_class_recall:
        sum_c2 += s
    sum_ = sum_c1 - sum_c2
    print('No_sampling dataset recall deviation from sampling_dataset:', sum_)
    if sum_ > 0:
        print('Recall in the no_sampling dataset is higher than in the sampling_dataset.\n')
    else:
        print('Recall in the no_sampling dataset is lower than in the sampling_dataset.\n')

    print('Counting CV scores...')
    k_folds = KFold(n_splits=10)
    scores = cross_val_score(rf_model, X, y, cv=k_folds)
    print(' Cross Validation Scores: ', list(scores))
    print(' Average CV Score: ', '{0:.4f}'.format(scores.mean()))
    print(' Deviation from average cv score:', '{0:.4f}'.format(accuracy_score(y_test, y_pred) - scores.mean()), '\n')
    if (float(df_metrics.loc['Accuracy', 'No_sampling']) - float(df_metrics.loc['Accuracy', 'Sampling'])) > 0:
        return [df_metrics.loc['Accuracy', 'No_sampling'], 'No_sampling rf']
    elif (float(df_metrics.loc['Accuracy', 'No_sampling']) - float(df_metrics.loc['Accuracy', 'Sampling'])) < 0:
        return [df_metrics.loc['Accuracy', 'Sampling'], 'Sampling rf']
    else:
        return [df_metrics.loc['Accuracy', 'Sampling'], 'rf']

def classification(dataframe):
    print('Classification:')
    rf_lst = rf_classifier(dataframe)
    knn_lst = knn_classifier(dataframe)
    return rf_lst, knn_lst

def multinomial_regr(dataframe):
    print('Multinomial regression:')
    X = dataframe.drop(columns=['crm_cd', 'crm_cd_desc', 'dr_no', 'rpt_dist_no', 'status', 'crm_cd_2', 'crm_cd_3'])
    y = dataframe['crm_cd']
    X.columns = [x.capitalize() for x in X.columns]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = LogisticRegression(multi_class='multinomial', solver='saga', max_iter=3000, penalty='l2',
                               C=0.5, tol=0.001, random_state=42)
    model2 = LogisticRegression(multi_class='multinomial', solver='saga', tol=0.001)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = model.score(X_test, y_test)

    over_sampler = RandomOverSampler(sampling_strategy='minority')
    under_sampler = RandomUnderSampler(sampling_strategy='majority')
    model_with_sampling = Pipeline([('over_sampler', over_sampler), ('under_sampler', under_sampler), ('lr', model2)])
    model_with_sampling.fit(X_train, y_train)
    y_pred_sampling = model_with_sampling.predict(X_test)

    df_metrics = pd.DataFrame()
    df_metrics['Metrics'] = ['Accuracy', 'Missclassification', 'Precision', 'Recall', 'F1-score']
    df_metrics['No_sampling'] = ['{0:.4f}'.format(accuracy_score(y_test, y_pred)),
                                 '{0:.4f}'.format(zero_one_loss(y_test, y_pred)),
                                 '{0:.4f}'.format(precision_score(y_test, y_pred, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(recall_score(y_test, y_pred, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(f1_score(y_test, y_pred, average='weighted'))]
    df_metrics['Sampling'] = ['{0:.4f}'.format(accuracy_score(y_test, y_pred_sampling)),
                                 '{0:.4f}'.format(zero_one_loss(y_test, y_pred_sampling)),
                                 '{0:.4f}'.format(precision_score(y_test, y_pred_sampling, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(recall_score(y_test, y_pred_sampling, average='weighted', zero_division=1)),
                                 '{0:.4f}'.format(f1_score(y_test, y_pred_sampling, average='weighted'))]
    df_metrics.set_index('Metrics', inplace=True)
    print(' ', df_metrics, '\n')
    if float(df_metrics.loc['Recall', 'No_sampling']) - float(df_metrics.loc['Recall', 'Sampling']):
        print('Recall in the no_sampling dataset is higher than in the sampling_dataset.\n')
    else:
        print('Recall in the no_sampling dataset is lower than in the sampling_dataset.\n')

    print('Counting CV scores...')
    scores = cross_val_score(model, X_train, y_train, cv=3)
    print(f' Scores for each fold: {scores}')
    print(f' Mean score:', '{0:.4f}'.format(scores.mean()))
    print(' Deviation from average cv score:', '{0:.4f}'.format(test_score - scores.mean()), '\n')
    if (float(df_metrics.loc['Accuracy', 'No_sampling']) - float(df_metrics.loc['Accuracy', 'Sampling'])) > 0:
        return [df_metrics.loc['Accuracy', 'No_sampling'], 'No_sampling mlt']
    elif (float(df_metrics.loc['Accuracy', 'No_sampling']) - float(df_metrics.loc['Accuracy', 'Sampling'])) < 0:
        return [df_metrics.loc['Accuracy', 'Sampling'], 'Sampling mlt']
    else:
        return [df_metrics.loc['Accuracy', 'Sampling'], 'mlt']

def regression(dataframe):
    mlt_lst = multinomial_regr(dataframe)
    return mlt_lst
def data_preprocess(dataframe):
    pd.set_option('future.no_silent_downcasting', True)
    pd.set_option('display.max_columns', 28)
    pd.set_option('display.max_rows', 28)
    dataframe['date_occ'] = pd.to_datetime(dataframe['date_occ'])
    dataframe['Year'] = dataframe['date_occ'].dt.year
    dataframe['Month'] = dataframe['date_occ'].dt.month
    dataframe['Day'] = dataframe['date_occ'].dt.day
    dataframe['Weekday'] = dataframe['date_occ'].apply(lambda x: x.weekday())
    dataframe['Time'] = dataframe['time_occ'].astype(int)
    dataframe['Time'] = dataframe['Time'].apply(lambda x: x / 100).astype(int)

    dataframe.drop(columns=['date_occ', 'time_occ', 'date_rptd', 'part_1_2', 'cross_street'], inplace=True)
    print('\nThe unnecessary columns were removed.')
    [dataframe.pop(x) for x in ['area_name', 'mocodes', 'premis_desc', 'weapon_desc', 'status_desc', 'location']]
    dataframe.fillna({'weapon_used_cd': 0, 'crm_cd_2': 0, 'crm_cd_3': 0, 'premis_cd': 0,
                      'vict_sex': 0, 'vict_descent': 0}, inplace=True)
    print('Missing values were filled with 0.')
    dataframe['vict_sex'] = dataframe['vict_sex'].replace({'H': 0, 'X': 0})
    dataframe['vict_descent'] = dataframe['vict_descent'].replace({'X': 0, '-': 0})

    cols_to_conv = ['dr_no', 'area', 'rpt_dist_no', 'crm_cd', 'vict_age', 'premis_cd',
                    'weapon_used_cd', 'crm_cd_1', 'crm_cd_2', 'crm_cd_3']
    dataframe[cols_to_conv] = dataframe[cols_to_conv].astype(int)
    dataframe['lat'] = dataframe['lat'].astype(float)
    dataframe['lon'] = dataframe['lon'].astype(float)
    dataframe['crm_cd_desc'] = dataframe['crm_cd_desc'].astype(str)
    print('Features were converted to int and float.')

    categorical_features = ['vict_sex', 'vict_descent', 'status']
    dataframe[categorical_features] = dataframe[categorical_features].astype(str)
    inv_values = []
    values = []
    dict_lst = {}
    encoded = []
    label_encoder = LabelEncoder()
    for feature in categorical_features:
        inv_values.append(dataframe[feature].value_counts().keys().to_list())
        dataframe[feature] = label_encoder.fit_transform(dataframe[feature])
        values.append(dataframe[feature].value_counts().keys().to_list())
    for i, inv_value_lst in enumerate(inv_values):
        for j, inv_value in enumerate(inv_value_lst):
            dict_lst.update({values[i][j]: inv_value})
        encoded.append(dict_lst.copy())
        dict_lst.clear()
    print('Labels were encoded.')

    for i in range(len(encoded)):
        print(f" Feature {categorical_features[i]}: {encoded[i]}")
    return dataframe
def data_load():
    my_limit = 1000
    my_offset = 47150
    temp_dfs = list()
    i = 0
    while i < 20:
        url = 'https://data.lacity.org/resource/2nrs-mtv8.json?$limit={}&$offset={}'
        endpoint = url.format(my_limit, my_offset)
        response = requests.get(endpoint)
        if response.status_code == 200:
            data = response.json()
            temp_dfs.append(pd.DataFrame(data))
        else:
            print('Error has occurred. Please try again')
            return 0
        i += 1
        my_offset += 47150
    big_df = pd.concat(temp_dfs)
    return big_df

def accuracy_comparison(rf_lst, knn_lst, mlt_lst):
    lst_x, lst_y = [], []
    lst_x.append(rf_lst[1])
    lst_x.append(knn_lst[1])
    lst_x.append(mlt_lst[1])
    lst_y.append(float(rf_lst[0]))
    lst_y.append(float(knn_lst[0]))
    lst_y.append(float(mlt_lst[0]))
    sns.set_theme(style='whitegrid', context='notebook')
    plt.figure(figsize=(13, 9))
    sns.barplot(x=np.array(lst_x), y=np.array(lst_y), color='#96B5DE')
    plt.xlabel('The highest accuracy of each method', fontsize=14, labelpad=8)
    plt.ylabel('Accuracy', fontsize=14, labelpad=5)
    plt.title('Accuracy comparison ', fontsize=16, pad=14)
    plt.show()

def patterns(df):
    print('Some interesting patterns:')
    sns.set_theme(style='whitegrid', context='notebook')
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    sns.barplot(x=np.array(df['Year'].value_counts().keys().to_list()),
                y=np.array(df['Year'].value_counts().values), ax=ax1,
                color='#96B5DE')
    ax1.set_xlabel('Year', fontsize=14, labelpad=8)
    ax1.set_ylabel('Crime count', fontsize=14, labelpad=5)
    ax1.set_title('Crime count by year', fontsize=16, pad=14)

    sns.barplot(x=np.array(df['Month'].value_counts().keys().to_list()),
                y=np.array(df['Month'].value_counts().values), ax=ax2,
                color='#F2DA8D')
    ax2.set_xlabel('Month', fontsize=14, labelpad=8)
    ax2.set_ylabel('Crime count', fontsize=14, labelpad=5)
    ax2.set_title('Crime count by month', fontsize=16, pad=14)

    sns.lineplot(x=np.array(df['Day'].value_counts().keys().to_list()),
                 y=np.array(df['Day'].value_counts().values), ax=ax4,
                 color='#7CDA97')
    ax4.set_xlabel('Day', fontsize=14, labelpad=8)
    ax4.set_ylabel('Crime count', fontsize=14, labelpad=5)
    ax4.set_title('Crime count by day', fontsize=16, pad=14)

    sns.lineplot(x=np.array(df['Time'].value_counts().keys().to_list()),
                 y=np.array(df['Time'].value_counts().values), ax=ax3,
                 color='#EF6A61')
    ax3.set_xlabel('Time', fontsize=14, labelpad=8)
    ax3.set_ylabel('Crime count', fontsize=14, labelpad=5)
    ax3.set_title('Crime count by time', fontsize=16, pad=14)
    plt.show()

    tmp_cd_desc = ['Stol_vehicle', 'Battery(simple_assault)', 'Burglary', 'Identity_theft', 'Burg_from_vehicle',
                   'Vandalism-felony', 'Assault_deadly_weapon', 'Plain_theft', 'Intimate_partner_assault',
                   'Theft_from_motor_vehicle']
    plt.figure(figsize=(13, 9))
    sns.barplot(x=tmp_cd_desc,
                y=np.array(df['crm_cd_desc'].value_counts().values[:10]),
                color='#96B5DE')
    plt.xlabel('Crime category', fontsize=14, labelpad=8)
    plt.ylabel('Crime count', fontsize=14, labelpad=5)
    plt.title('Crime counts by top 10 crime categories', fontsize=16, pad=14)
    plt.xticks(rotation=25)
    plt.show()

    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(13, 9))
    plt.subplots_adjust(hspace=0.4, wspace=0.3)
    sns.barplot(x=np.array(['Mon', 'Tue', 'Wed', 'Th', 'Fri', 'Sat', 'Sun']),
                y=np.array(df['Weekday'].value_counts().values), ax=ax1,
                color='#96B5DE')
    ax1.set_xlabel('Weekday', fontsize=14, labelpad=8)
    ax1.set_ylabel('Crime count', fontsize=14, labelpad=8)
    ax1.set_title('Crime counts by weekdays', fontsize=16, pad=14)

    tmp_vict_age = []
    for age in df['vict_age'].value_counts().keys().to_list()[:10]:
        if age == 0:
            tmp_vict_age.append('Unknown')
        else:
            tmp_vict_age.append(age)

    sns.barplot(x=tmp_vict_age,
                y=np.array(df['vict_age'].value_counts().values[:10]), ax=ax2,
                color='#F2DA8D')
    ax2.set_xlabel('Victim age', fontsize=14, labelpad=8)
    ax2.set_ylabel('Crime count', fontsize=14, labelpad=5)
    ax2.set_title('Crime counts by top 10 victim ages', fontsize=16, pad=14)

    sns.barplot(x=['Male', 'Female', 'Unknown'],
                y=np.array(df['vict_sex'].value_counts().values), ax=ax3,
                color='#EF6A61')
    ax3.set_xlabel('Victim Sex', fontsize=14, labelpad=8)
    ax3.set_ylabel('Crime count', fontsize=14, labelpad=5)
    ax3.set_title('Crime counts by victim sex', fontsize=16, pad=14)

    tmp_vict_descent = ['Hispanic', 'Unknown', 'White', 'Black', 'Other', 'Other Asian',
                        'Korean', 'Filipino', 'Chinese', 'Japanese']
    sns.barplot(x=tmp_vict_descent,
                y=np.array(df['vict_descent'].value_counts().values[:10]), ax=ax4,
                color='#7CDA97')
    ax4.set_xlabel('Victim Descent', fontsize=14, labelpad=8)
    ax4.set_ylabel('Crime count', fontsize=14, labelpad=5)
    ax4.set_title('Crime counts by top 10 victim descents', fontsize=16, pad=14)
    plt.xticks(rotation=25)
    plt.show()


if __name__ == '__main__':
    print('Starting the program...')
    df = data_load()
    if isinstance(df, int):
        sys.exit('Failed to load dataset.')
    elif isinstance(df, pd.DataFrame):
        print('Dataset was loaded successfully.')
        df = data_preprocess(df)
        if isinstance(df, pd.DataFrame):
            print('Dataset was preprocessed successfully.\n')
        else:
            sys.exit('Failed to preprocess dataset.\n')
    feature_selection(df)
    rf_lst, knn_lst = classification(df)
    mlt_lst = regression(df)
    accuracy_comparison(rf_lst, knn_lst, mlt_lst)
    patterns(df)
    print('The program is executed.')
