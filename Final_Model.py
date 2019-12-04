#!/usr/bin/env python
# coding: utf-8

# In[]:

# libraries to be imported
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import datetime
from statsmodels.tsa.stattools import adfuller
import warnings

warnings.filterwarnings("ignore")

# In[]:

zensors_cols = ['APICaptureDate',
                'sensor_name',
                'data_name',
                'uom_name',
                'uom_description',
                'datavalue']

tsa_cols = ['Airport',
            'Airport Name',
            'Date',
            'Day of Week',
            'Checkpoint',
            'Hour of Day',
            'Metrics',
            'Total Throughput',
            'Total PreCheck',
            'Total MI (w/RTTA)',
            'Pure PreCheck (No MI)',
            'PreCheck Minutes Open',
            'Std Min Open']


def read_files(files, col_names):
    new_list = []
    df = pd.DataFrame()
    for filename in files:
        print("Read ", filename, " successfully !!")
        df = pd.read_excel(filename)
        df.columns = col_names
        df.reset_index(inplace=True, drop=True)
        new_list.append(df)

    if (len(files) > 1):
        data = pd.concat(new_list)
    else:
        data = df
    return data


# In[]:

# Read the data file names
data_files = [file for file in glob.glob('*.xlsx')]
# Zensors files
tsa_files = [file for file in data_files if 'tsa' in file.lower() and not 'forecasted' in file.lower()]
# Zensors files
zensors_files = [file for file in data_files if 'zensors' in file.lower() and not 'forecasted' in file.lower()]

# read data
final_zensors_data = read_files(zensors_files, zensors_cols)
final_tsa_data = read_files(tsa_files, tsa_cols)


# In[]:

# The first thing we should know is where the missing values are
# change the value according to what we want
def remove_null(matrixes):
    for m in matrixes:
        print("new")
        for i in range(len(m)):
            if (str(m[i][5]) == "Nan"):
                missing_date = m[i][0]
                missing_hour = missing_date.hour
                missing_weekday = missing_date.weekday()
                # if the previous or next observation lies in the same day and within the same hour
                if (missing_weekday == m[i - 1][0].weekday() & missing_hour == m[i - 1][0].hour):
                    m[i][5] = m[i - 1][5]
                if (missing_weekday == m[i + 1][0].weekday() & missing_hour == m[i + 1][0].hour):
                    m[i][5] = m[i + 1][5]
                else:  # go back one month
                    values_to_average = []
                    for j in range(13400):
                        temp_date = m[i - 13400 + j][0]
                        if (missing_weekday == temp_date.weekday() & missing_hour == temp_date.hour):
                            values_to_average.append(m[i - 13400 + j][5])
                        if (len(values_to_average) == 0):
                            for k in range(3360):
                                temp_date = m[i - 3360 + k][0]
                                if (missing_hour == temp_date.hour):
                                    values_to_average.append(m[i - 3360 + k][5])
                    m[i][5] = np.mean(values_to_average)


def create_df(matrixes):
    df_list = []
    for m in range(len(matrixes)):
        df_list.append(pd.DataFrame(matrixes[m]))
        df_list[m].rename(columns={0: "APICaptureDate", 1: "sensor_name", 2: "data_name",
                                   3: "uom_name", 4: "uom_description", 5: "datavalue"}, inplace=True)
    return df_list


# Function to convert the data into appropriate time
def modify_date(series):
    for i in range(len(series.APICaptureDate)):
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '00:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '01:00:00':
            # 12 am to 1 am
            series['Time'][i] = '00:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '01:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '02:00:00':
            series['Time'][i] = '01:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '02:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '03:00:00':
            series['Time'][i] = '02:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '03:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '04:00:00':
            series['Time'][i] = '03:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '04:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '05:00:00':
            series['Time'][i] = '04:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '05:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '06:00:00':
            series['Time'][i] = '05:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '06:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '07:00:00':
            series['Time'][i] = '06:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '07:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '08:00:00':
            series['Time'][i] = '07:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '08:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '09:00:00':
            series['Time'][i] = '08:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '09:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '10:00:00':
            series['Time'][i] = '09:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '10:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '11:00:00':
            series['Time'][i] = '10:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '11:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '12:00:00':
            series['Time'][i] = '11:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '12:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '13:00:00':
            series['Time'][i] = '12:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '13:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '14:00:00':
            series['Time'][i] = '13:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '14:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '15:00:00':
            series['Time'][i] = '14:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '15:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '16:00:00':
            series['Time'][i] = '15:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '16:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '17:00:00':
            series['Time'][i] = '16:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '17:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '18:00:00':
            series['Time'][i] = '17:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '18:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '19:00:00':
            series['Time'][i] = '18:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '19:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '20:00:00':
            series['Time'][i] = '19:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '20:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '21:00:00':
            series['Time'][i] = '20:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '21:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '22:00:00':
            series['Time'][i] = '21:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '22:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '23:00:00':
            series['Time'][i] = '22:00:00'
        if series.APICaptureDate[i].strftime('%H:%M:%S') >= '23:00:00' and series.APICaptureDate[i].strftime(
                '%H:%M:%S') < '24:00:00':
            series['Time'][i] = '23:00:00'
    return series


def preprocessing_zensors(dataframe):
    # Considering the wait time/people data only for the Main Checkpoint
    main = dataframe[(dataframe.sensor_name == 'Main Line  (Main)') | (dataframe.sensor_name == 'Main Line (Main)')]
    # resetting index
    main.reset_index(inplace=True, drop=True)
    # getting separate date and time columns
    main['Date'] = [d.date() for d in main.APICaptureDate]
    main['Time'] = [d.time() for d in main.APICaptureDate]
    # resetting index
    main.reset_index(inplace=True, drop=True)
    # modifying the data frame
    modify_date(main)
    # converting to float
    main['datavalue'] = main.datavalue.astype(float)
    # finding mean for every hour
    avg_val_main = main.groupby(['Time', 'Date'], as_index=False)['datavalue'].mean().sort_values(['Date', 'Time'])
    # resetting index
    avg_val_main.reset_index(inplace=True, drop=True)
    return avg_val_main


def preprocessing_tsa(dataframe):
    final_cols = ['Date', 'Time', 'datavalue']

    main = dataframe[dataframe['Checkpoint'] == 'Main Checkpoint'].loc[:,
           ['Date', 'Hour of Day', 'Total Throughput']]
    main.columns = final_cols
    main.Date = [date.date() for date in main.Date]
    # main['Total Throughput'][main['Total PreCheck'] == '-'] = 0
    # main['Total PreCheck'][main['Total PreCheck'] == '-'] = 0

    alt = dataframe[dataframe['Checkpoint'] == 'Alternate Checkpoint'].loc[:,
          ['Date', 'Hour of Day', 'Total Throughput']]
    alt.columns = final_cols
    alt.Date = [date.date() for date in alt.Date]
    # alt['Total Throughput'][alt['Total PreCheck'] == '-'] = 0
    # alt['Total PreCheck'][alt['Total PreCheck'] == '-'] = 0

    return main, alt


# In[]:

# Filtering out people and wait time
all_people = final_zensors_data[final_zensors_data["data_name"] == "People Count"]
all_wait_time = final_zensors_data[final_zensors_data["data_name"] == "Wait Time"]

# Replacing nulls with nan's
lines = [all_people, all_wait_time]
for line in lines:
    line["datavalue"].fillna("Nan", inplace=True)

# convert to np array
matrixes = [tb.as_matrix() for tb in lines]

# get time stamps
for matrix in matrixes:
    for i in range(len(matrix)):
        timestamp = str(matrix[i][0])
        index = timestamp.find(".")
        timestamp = timestamp[:index]
        try:
            matrix[i][0] = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = timestamp + ":00"
            matrix[i][0] = datetime.datetime.strptime(timestamp, '%m/%d/%Y %H:%M:%S')

remove_null(matrixes=matrixes)
zensors_df = create_df(matrixes)

# getting the average wait time and avgerage number of people from Zensors
zensors_people_main = preprocessing_zensors(zensors_df[0])
zensors_waittime_main = preprocessing_zensors(zensors_df[1])

# getting number of people from TSA
tsa_people_main, tsa_people_alt = preprocessing_tsa(final_tsa_data)


# In[]:

# finding if the series is stationery or not (value of d)
def set_d(dataframe):
    result = adfuller(dataframe.datavalue)
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    d = 0.5
    if result[1] <= 0.05:
        d = 0

    return d


def tune_hyperparams(dataframe, d):
    p_values = [5, 10, 15, 20, 25, 30]
    q_values = [5, 10, 15, 20, 25, 30]
    best_score = np.iinfo(np.int32).max
    best_p = 1000
    best_q = 1000
    for p in p_values:
        for q in q_values:
            try:
                print(p, q)
                model = ARIMA(np.asarray(dataframe.datavalue), order=(p, d, q))
                model_fit = model.fit(disp=-1)
                mse = mean_squared_error(dataframe.datavalue, model_fit.fittedvalues)
                if mse < best_score:
                    best_score, best_p, best_q = mse, p, q
                else:
                    continue
            except:
                continue
    return [best_p, best_q, best_score]


def arima_model(dataframe, best_p, best_q, d):
    model = ARIMA(np.asarray(dataframe.datavalue), order=(best_p, d, best_q))
    model_fit = model.fit(disp=-1)
    predictions, se, conf = model_fit.forecast(168, alpha=0.05)
    predictions[predictions < 0] = 0
    return predictions


def run_model(data):
    # finding best parameters for wait times
    d = set_d(data)
    params = tune_hyperparams(data, d)
    print('The best parameters are ', params[0], ' and ', params[1])
    predictions = arima_model(data, params[0], params[1], d)
    return predictions


# In[]:

# getting the last date and time
# adding 7 days to the last date of data passed
date_lst = []
for i in range(1, 8):
    date = zensors_waittime_main.Date.iloc[-1] + datetime.timedelta(days=i)
    date_lst.append(date)

# getting the last time and converting into int
last_time = int(zensors_waittime_main.Time.iloc[-1][0:2])
# Finding the time for the next 24 hours
time_lst = [(last_time + i) - 24 for i in range(1, 25)]
time_lst = [(last_time + i + 1) if time_lst[i] < 0 else time_lst[i] for i in range(len(time_lst))]
time_lst = [str(time_lst[i]) + ':00:00' if len(str(time_lst[i])) == 2 else '0' + str(time_lst[i]) + ':00:00' for i in
            range(len(time_lst))]
# time_lst

date_time_lst = []
for date in date_lst:
    for time in time_lst:
        date_time = str(date) + " " + time
        date_time_lst.append(date_time)
# date_time_lst


# In[]:

# output Zensors predictions

zensors_waittime_pred = run_model(zensors_waittime_main)
zensors_people_pred = run_model(zensors_people_main)

# creating a dataframe for the final forecasted results
zensors_forecasted_data = pd.DataFrame()
zensors_forecasted_data['Timestamp'] = date_time_lst
zensors_forecasted_data['forecasted_waittime'] = zensors_waittime_pred
zensors_forecasted_data['forcasted_people'] = zensors_people_pred
zensors_forecasted_data.to_excel('forecasted_Data.xlsx', index=False)

# In[]:

# output TSA predictions

tsa_people_pred = run_model(tsa_people_main)

tsa_forecasted_data = tsa_people_main.copy()
tsa_forecasted_data['datavalue'] = tsa_people_pred
tsa_forecasted_data.to_excel('forecasted_TSA.xlsx', index=False)
