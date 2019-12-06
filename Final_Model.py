# import the libraries
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


# %%
# I. read data

def read_data(files, data_source):
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

    if data_source == 'zensors':
        cols = zensors_cols
    elif data_source == 'tsa':
        cols = tsa_cols
    else:
        print('data source error!')
        raise

    new_list = []
    df = pd.DataFrame()

    try:
        for filename in files:
            print("Read ", filename, " successfully !!")
            df = pd.read_excel(filename)
            df.columns = cols
            df.reset_index(inplace=True, drop=True)
            new_list.append(df)
    except:
        print('cannot read files!')

    if (len(files) > 1):
        data = pd.concat(new_list)
    else:
        data = df
    return data


# %%:
# II. preprocessing

# get timestamps
def get_timestamp(matrix):
    for i in range(len(matrix)):
        timestamp = str(matrix[i][0])
        index = timestamp.find(".")
        timestamp = timestamp[:index]
        try:
            matrix[i][0] = datetime.datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
        except:
            timestamp = timestamp + ":00"
            matrix[i][0] = datetime.datetime.strptime(timestamp, '%m/%d/%Y %H:%M:%S')

    return matrix


# change the missing values
def remove_null(matrix):
    for i in range(len(matrix)):
        if (str(matrix[i][5]) == "Nan"):
            missing_date = matrix[i][0]
            missing_hour = missing_date.hour
            missing_weekday = missing_date.weekday()
            # if the previous or next observation lies in the same day and within the same hour
            if (missing_weekday == matrix[i - 1][0].weekday() & missing_hour == matrix[i - 1][0].hour):
                matrix[i][5] = matrix[i - 1][5]
            if (missing_weekday == matrix[i + 1][0].weekday() & missing_hour == matrix[i + 1][0].hour):
                matrix[i][5] = matrix[i + 1][5]
            else:  # go back one month
                val_to_avg = []
                for j in range(13400):
                    temp_date = matrix[i - 13400 + j][0]
                    if (missing_weekday == temp_date.weekday() & missing_hour == temp_date.hour):
                        val_to_avg.append(matrix[i - 13400 + j][5])
                    if (len(val_to_avg) == 0):
                        for k in range(3360):
                            temp_date = matrix[i - 3360 + k][0]
                            if (missing_hour == temp_date.hour):
                                val_to_avg.append(matrix[i - 3360 + k][5])
                matrix[i][5] = np.mean(val_to_avg)
    print("remove null")

    return matrix


# convert the data into appropriate time
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


# get the last date and time and
# add 7 days to the last date of data passed
def get_datetime(df):
    date_list = []
    for i in range(1, 8):
        date = df.Date.iloc[-1] + datetime.timedelta(days=i)
        date_list.append(date)

    # get the last time and converting into int
    last_time = int(df.Time.iloc[-1][0:2])
    # find the time for the next 24 hours
    time_list = [(last_time + i) - 24 for i in range(1, 25)]
    time_list = [(last_time + i + 1) if time_list[i] < 0 else time_list[i] for i in range(len(time_list))]
    time_list = [str(time_list[i]) + ':00:00' if len(str(time_list[i])) == 2 else '0' + str(time_list[i]) + ':00:00' for
                 i in range(len(time_list))]

    date_time_list = []
    for date in date_list:
        for time in time_list:
            date_time = str(date) + " " + time
            date_time_list.append(date_time)

    return date_time_list


def get_hourly_avg(df, lane='main'):
    df.columns = ['APICaptureDate',
                  'sensor_name',
                  'data_name',
                  'uom_name',
                  'uom_description',
                  'datavalue']

    # consider the wait time/people data only for the Main Checkpoint
    if lane == 'main':
        data = df[(df.sensor_name == 'Main Line  (Main)') | (df.sensor_name == 'Main Line (Main)')]
    elif lane == 'alt':
        data = df[
            (df.sensor_name == 'Altenative Line  (Altenative)') | (df.sensor_name == 'Altenative Line (Altenative)')]
    else:
        print('lane name error!')
        raise

    # reset index
    data.reset_index(inplace=True, drop=True)
    # get separate date and time columns
    data['Date'] = [d.date() for d in data.APICaptureDate]
    data['Time'] = [d.time() for d in data.APICaptureDate]
    # reset index
    data.reset_index(inplace=True, drop=True)
    # modify the data frame
    modify_date(data)

    # convert to float
    data['datavalue'] = data.datavalue.astype(float)
    # find mean for every hour
    avg_data = data.groupby(['Time', 'Date'], as_index=False)['datavalue'].mean().sort_values(['Date', 'Time'])
    # reset index
    avg_data.reset_index(inplace=True, drop=True)

    return avg_data


def preprocessing_zensors(df, lane='main', data_name='Wait Time'):
    # filter out people and wait time
    result = df[df["data_name"] == data_name]
    # replace nulls with nan's
    result["datavalue"].fillna("Nan", inplace=True)
    # convert to np array
    matrix = remove_null(get_timestamp(result.as_matrix()))
    # get average wait time and average number of people from Zensors
    final = get_hourly_avg(pd.DataFrame(matrix), lane)

    return final


def preprocessing_tsa(df):
    output_cols = ['Date', 'Time', 'datavalue']

    main = df[df['Checkpoint'] == 'Main Checkpoint'].loc[:,
           ['Date', 'Hour of Day', 'Total Throughput']]
    main.columns = output_cols
    main.Date = [date.date() for date in main.Date]
    # main['Total Throughput'][main['Total PreCheck'] == '-'] = 0
    # main['Total PreCheck'][main['Total PreCheck'] == '-'] = 0

    alt = df[df['Checkpoint'] == 'Alternate Checkpoint'].loc[:,
          ['Date', 'Hour of Day', 'Total Throughput']]
    alt.columns = output_cols
    alt.Date = [date.date() for date in alt.Date]
    # alt['Total Throughput'][alt['Total PreCheck'] == '-'] = 0
    # alt['Total PreCheck'][alt['Total PreCheck'] == '-'] = 0

    return main, alt


# %%:
# III. ARIMA Model


# find if the series is stationery or not (value of d)
def stationary_d(df):
    results = adfuller(df.datavalue)
    print('ADF Statistic: %f' % results[0])
    print('p-value: %f' % results[1])

    d = 0.05
    if results[1] <= 0.05:
        d = 0
    return d


# find the best parameters for wait times
def tune_params(df, d):
    p_values, q_values = np.arange(5, 35, 5), np.arange(5, 35, 5)
    best_score, best_p, best_q = np.iinfo(np.int32).max, 1000, 1000

    for p in p_values:
        for q in q_values:
            try:
                print(p, q)
                model = ARIMA(np.asarray(df.datavalue), order=(p, d, q))
                model_fit = model.fit(disp=-1)
                mse = mean_squared_error(df.datavalue, model_fit.fittedvalues)
                if mse < best_score:
                    best_score, best_p, best_q = mse, p, q
                else:
                    continue
            except:
                continue
    return best_p, best_q, best_score


def run_model(df):
    # find the best parameters for wait times
    d = stationary_d(df)
    p, q, s = tune_params(df, d)
    print('The best parameters are ', p, ' and ', q)

    model = ARIMA(np.asarray(df.datavalue), order=(p, d, q))
    model_fit = model.fit(disp=-1)
    predictions, se, conf = model_fit.forecast(168, alpha=0.05)
    predictions[predictions < 0] = 0
    return predictions


# %%:
# IV. Output


# output Zensors predictions
def output_zensors(waittime, people):
    zensors_pred = pd.DataFrame()
    zensors_pred['Timestamp'] = get_datetime(waittime)
    zensors_pred['forecasted_waittime'] = run_model(waittime)
    zensors_pred['forecasted_people'] = run_model(people)
    try:
        zensors_pred.to_excel('forecasted_Data.xlsx', index=False)
    except:
        print('cannot output!')
    return zensors_pred


# output TSA predictions
def output_tsa(people):
    tsa_pred = tsa_main_people.copy()[-168:]
    tsa_pred['Date'] += datetime.timedelta(days=7)
    tsa_pred['datavalue'] = run_model(people)
    try:
        tsa_pred.to_excel('forecasted_TSA.xlsx', index=False)
    except:
        print('cannot output!')
    return tsa_pred


# %%:

if __name__ == '__main__':
    # read file names
    data_files = [file for file in glob.glob('*.xlsx')]
    tsa_files = [file for file in data_files if 'tsa' in file.lower() and not 'forecasted' in file.lower()]
    zensors_files = [file for file in data_files if 'zensors' in file.lower() and not 'forecasted' in file.lower()]

    # read data from files
    zensors_data = read_data(zensors_files, 'zensors')
    tsa_data = read_data(tsa_files, 'tsa')

    # get average wait time and average number of people from Zensors
    zensors_main_waittime = preprocessing_zensors(zensors_data, 'main', 'Wait Time')
    zensors_main_people = preprocessing_zensors(zensors_data, 'main', 'People Count')
    # zensors_alt_waittime = preprocessing_zensors(zensors_data, 'alt', 'Wait Time')
    # zensors_alt_people = preprocessing_zensors(zensors_data, 'alt', 'People Count')

    # get number of people from TSA
    tsa_main_people, tsa_alt_people = preprocessing_tsa(tsa_data)

    # output Zensors predictions
    zensors_main_pred = output_zensors(zensors_main_waittime, zensors_main_people)
    # zensors_alt_pred = output_zensors(zensors_alt_waittime, zensors_alt_waittime)

    # output TSA predictions
    tsa_main_pred = output_tsa(tsa_main_people)
    # tsa_alt_pred = output_tsa(tsa_alt_people)
