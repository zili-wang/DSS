# Confidential
# Copyright @2019 Zili Wang
# All rights reserved.
# 31154594 Vraagspecificatie simulatiemodel voorspelling wegdekschade
# Any change, update, or development of the file should be notified to wangzilihit@outlook.com


import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from matplotlib import pyplot as plt
import seaborn as sns

import graphviz
import pydotplus
import io
from scipy import misc
from IPython.display import display

import openpyxl
from typing import Any

import time
import datetime

def DTCfeature(PI):
    if PI == 'Roughness':
        features = ['AGE_IRI', 'SURFACE_COMBID',
                    'SURFACE_DAB', 'SURFACE_EAB', 'SURFACE_OAB', 'SURFACE_SMA', 'SURFACE_ZOAB', 'SURFACE_ZOAB+',
                    'SURFACE_ZOABTW', 'SURFACE_ZOEAB',
                    'IRI_VALUE_0',
                    'I_L1', 'I_L2', 'I_L3', 'T_TEMP_25', 'T_TEMP_0', 'T_TEMP_0_below', 'T_PERCIPITATION']
    else:
        features = ['AGE_RUT', 'SURFACE_COMBID',
                    'SURFACE_DAB', 'SURFACE_EAB', 'SURFACE_OAB', 'SURFACE_SMA', 'SURFACE_ZOAB', 'SURFACE_ZOAB+',
                    'SURFACE_ZOABTW', 'SURFACE_ZOEAB',
                    'RUTTING_VALUE_0',
                    'I_L1', 'I_L2', 'I_L3', 'T_TEMP_25', 'T_TEMP_0', 'T_TEMP_0_below', 'T_PERCIPITATION']
    return features

def DTCtype(PI):
    if PI == 'Roughness':
        types = ["IRI_CLASS"]
    else:
        types = ["RUTTING_CLASS"]
    return types

def DTCtrainsheet(PI):
    if PI == 'Roughness':
        sheet = "Train_IRI"
    else:
        sheet = "Train_RUT"
    return sheet

def DTCtestsheet(chosen_time_str):

    df_test = pd.read_csv("DTC_Test_Raw.csv") # df_test = pd.read_excel('DTCtest.xlsx', sheet_name="Test_Raw")  # Import data
    # Transfer objct in csv to dateFrame
    df_test['DATE_IRI_2'] = pd.to_datetime(df_test['DATE_IRI'])
    del df_test['DATE_IRI']
    df_test['DATE_IRI'] = df_test['DATE_IRI_2']

    df_test['CONSTR_DATE_2'] = pd.to_datetime(df_test['CONSTR_DATE'])
    del df_test['CONSTR_DATE']
    df_test['CONSTR_DATE'] = df_test['CONSTR_DATE_2']


    test = df_test
    # Translate the chosen time string to datetime.panda
    date = datetime.datetime.strptime(chosen_time_str, '%Y-%m-%d')
    # Calculate Day from the measurement to the chosen time
    test['SIMULATION_TIME'] = date
    test['DAY'] = test['SIMULATION_TIME'] - test['DATE_IRI']
    # Calculate AGE_IRI, AGE_RUT
    test['AGE_IRI_time'] = test['SIMULATION_TIME'] - test['CONSTR_DATE']
    test['AGE_RUT_time'] = test['SIMULATION_TIME'] - test['CONSTR_DATE']
    # Convert time to number
    test['AGE_IRI']= test['AGE_IRI_time'].astype('timedelta64[D]').astype(int)
    test['AGE_RUT']= test['AGE_RUT_time'].astype('timedelta64[D]').astype(int)
    # Calculate DAY_I_L1; DAY_I_L2; DAY_I_L3
    I_test = pd.read_csv("DTC_Day_I.csv")  # I_test = pd.read_excel('DTCtest.xlsx', sheet_name="Day_I")  # Import data
    I_test_columns = I_test.columns.values.tolist()


    if chosen_time_str in I_test_columns:
        Num_date = I_test_columns.index(chosen_time_str)
    else:
        print("The chosen simulation time is out of range.")

    measurement_2018 = "2018-05-15"
    Num_date_measurement = I_test_columns.index(measurement_2018)

    I_AL_test = I_test.iloc[0:2830, Num_date_measurement:Num_date]
    I_AL_test['Col_sum'] = I_AL_test.apply(lambda x: x.sum(), axis=1)

    # Calculate the I_L1,I_L2,I_L3 based on the annual statistics data; (coz real-time data is missing)
    I_AL_test['I_L1_ANN'] = test['I_L1_2018'] + test['DAY'].dt.days * test['DAY_L1_2018']
    I_AL_test['I_L2_ANN'] = test['I_L2_2018'] + test['DAY'].dt.days * test['DAY_L2_2018']
    I_AL_test['I_L3_ANN'] = test['I_L3_2018'] + test['DAY'].dt.days * test['DAY_L3_2018']
    # Calculate the I_L1,I_L2,I_L3 based on the real-time data
    I_AL_test['I_L1_REAL'] = test['I_L1_2018'] + I_AL_test['Col_sum'] * I_test['PER_L1_2018']
    I_AL_test['I_L2_REAL'] = test['I_L2_2018'] + I_AL_test['Col_sum'] * I_test['PER_L2_2018']
    I_AL_test['I_L3_REAL'] = test['I_L3_2018'] + I_AL_test['Col_sum'] * I_test['PER_L3_2018']

    # If the realtime data is missing, I_L1 = I_L1_ANN; if realtime data exists I_L1 = I_L1_REAL
    # Firstly, correct ZERO and NON-ZERO series according to Col_sum
    Col_sum_list = list(I_AL_test['Col_sum'])
    Col_sum_arr = np.array(Col_sum_list)

    Col_zero = []
    Col_nonzero = []
    for i in range(len(Col_sum_arr)):
        if Col_sum_arr[i] == 0:
            Col_zero.append(0)
            Col_nonzero.append(1)
        else:
            Col_zero.append(1)
            Col_nonzero.append(0)

    test['I_L1'] = I_AL_test['I_L1_ANN'] * Col_nonzero + I_AL_test['I_L1_REAL'] * Col_zero
    test['I_L2'] = I_AL_test['I_L2_ANN'] * Col_nonzero + I_AL_test['I_L2_REAL'] * Col_zero
    test['I_L3'] = I_AL_test['I_L3_ANN'] * Col_nonzero + I_AL_test['I_L3_REAL'] * Col_zero

    return test
def show_tree(tree,features,path):
    f = io.StringIO()
    export_graphviz(tree,out_file = f, feature_names = features, class_names = '01')
    pydotplus.graph_from_dot_data(f.getvalue()).write_png(path)
    #img = misc.imread(path)
    plt.rcParams["figure.figsize"] = (20,20)
   # plt.imshow(img)

def DTC(features, types, train_sheet, test, chosen_time_str):
    train_sheet_csv_name = 'DTC_'+train_sheet+'.csv'
    df = pd.read_csv(train_sheet_csv_name)  #df = pd.read_excel('DTCtest.xlsx', sheet_name=train_sheet)  # Import data


    train = df

    c = DecisionTreeClassifier(min_samples_split=30, random_state=0)  # Decision tree

    x_train = train[features]
    y_train = train[types]

    x_test = test[features]
    test_without_NaN = test.dropna(axis=0)
    x_test_without_NaN = x_test.dropna(axis=0)
    # y_test = test[types]

    dt = c.fit(x_train, y_train)

    show_tree(dt, features, 'RESULT_Decision Tree.png')

    y_pred = c.predict(x_test_without_NaN)

    # Plot the failure events with all the features
    H_features = ['ROAD', 'DIRECTION', 'FROM_KM', 'AGE_IRI', 'AGE_RUT', 'SURFACE_LAYER',
                  'IRI_VALUE_0', 'RUTTING_VALUE_0', 'I_L1', 'I_L2', 'I_L3',
                  'T_TEMP_25', 'T_TEMP_0', 'T_TEMP_0_below', 'T_PERCIPITATION']
    test_d = test_without_NaN[H_features]#test_d = test[H_features]
    y_pred_n = np.nonzero(y_pred)
    df_test_array = np.array(test_d)
    pro_pred = []

    for i in y_pred_n:
        row = df_test_array[i]
        pro_pred.append(row)

    pro_pred_array = np.array(pro_pred)
    pro_pred_2d_array = pro_pred_array[0]


    test_features = test[H_features]#test_d = test[H_features]
    if train_sheet =="Train_IRI":
        thres = 3.5
        train_thres =train[train.IRI_VALUE >= thres]

    else:
        thres = 18
        train_thres =train[train.RUTTING_VALUE>= thres]

    train_thres_features =train_thres[H_features]
    train_thres_array = np.array(train_thres_features)
    train_thres_2d_array =train_thres_array[0]

    H_Pred = np.array([['ROAD', 'DIRECTION', 'FROM_KM', 'AGE_IRI', 'AGE_RUT', 'SURFACE_LAYER',
                        'IRI_VALUE_0', 'RUTTING_VALUE_0', 'I_L1', 'I_L2', 'I_L3',
                        'T_TEMP_25', 'T_TEMP_0', 'T_TEMP_0_below', 'T_PERCIPITATION']])

    Pred_Data_H = np.concatenate((H_Pred, pro_pred_2d_array,train_thres_array))

    # write the prediction results in excel file
    data = pd.DataFrame(Pred_Data_H)
    writer = pd.ExcelWriter('RESULT_Report.xlsx')
    data.to_excel(writer, chosen_time_str, float_format='%.5f', header=True)
    writer.save()
    writer.close()
    return Pred_Data_H.shape[0]-1

#PI = "Roughness"
#date = "2019-01-01"

#features = DTCfeature(PI)
#types = DTCtype(PI)
#trainsheet_name = DTCtrainsheet(PI)
#test = DTCtestsheet(date)
#a = DTC(features,types,trainsheet_name,test,date)



