import pandas as pd
import numpy as np


# 0. helpers

def findHour(h):
    for i, hr in enumerate(hrs):
        if int(h) >= (hr - 1) * 100 and int(h) < hr * 100:
            return i


def findDay(day):
    for d in days:
        if np.datetime64('2019-08-{:02d}'.format(d)) == day:
            return d - 1


# 1. read files


# 1.1 schedule.xlsx
schedule = pd.ExcelFile('schedule.xlsx')
departures = schedule.parse('Departures', header=1)
departures = departures.fillna(0)
arrivals = schedule.parse('Arrivals', header=1)
arrivals = arrivals.fillna(0)

# 1.2 throughput.xlsx
throughput = pd.ExcelFile('throughput.xlsx').parse(header=1)
throughput = throughput.fillna(0).replace('-', 0)

# 2. idx + cols

hrs, days, daysAug = [], [], []
for i in range(24):
    hrs.append(i + 1)

for i in range(1, 32):
    days.append(i)
    daysAug.append([i for hr in range(len(hrs))])
daysAug = np.array(daysAug).ravel()

airlines = sorted(set(departures.iloc[:, [0]].values.reshape(-1)))
dests = sorted(set(departures.iloc[:, [1]].values.reshape(-1)))

cols = ['hours'] + airlines + dests + ['flights', 'seats', 'bseats']

# 3. OHE

oheDep = np.zeros((len(days) * len(hrs), len(cols)), dtype=int)
oheArr = np.zeros((len(days) * len(hrs), len(cols)), dtype=int)

for day in range(len(days)):
    for h in range(len(hrs)):
        oheDep[day * len(hrs) + h][0] = hrs[h]
        oheArr[day * len(hrs) + h][0] = hrs[h]

for i in range(departures.shape[0]):
    noH = findHour(departures.iloc[[i], [3]].values[0][0])
    noA = airlines.index(departures.iloc[[i], [0]].values[0][0])
    noD = dests.index(departures.iloc[[i], [1]].values[0][0])

    for day in range(len(days)):
        # flights
        oheDep[day * len(hrs) + noH, -3] += departures.iloc[[i], [day * 3 + 4]].values[0][0]
        # seats
        oheDep[day * len(hrs) + noH, -2] += departures.iloc[[i], [day * 3 + 5]].values[0][0]
        # bseats
        oheDep[day * len(hrs) + noH, -1] += departures.iloc[[i], [day * 3 + 6]].values[0][0]

        # seats at airlines
        oheDep[day * len(hrs) + noH, noA + 1] += departures.iloc[[i], [day * 3 + 5]].values[0][0]
        # dests
        oheDep[day * len(hrs) + noH, noD + len(airlines) + 1] = 1

# for i in range(arrivals.shape[0]):
#     noH = findHour(arrivals.iloc[[i], [3]].values[0][0])
#     noA = airlines.index(arrivals.iloc[[i], [0]].values[0][0])
#     noD = dests.index(arrivals.iloc[[i], [1]].values[0][0])
#
#     for day in range(len(days)):
#         # flights
#         oheArr[day * len(hrs) + noH, -3] += arrivals.iloc[[i], [day * 3 + 4]].values[0][0]
#         # seats
#         oheArr[day * len(hrs) + noH, -2] += arrivals.iloc[[i], [day * 3 + 5]].values[0][0]
#         # bseats
#         oheArr[day * len(hrs) + noH, -1] += arrivals.iloc[[i], [day * 3 + 6]].values[0][0]
#
#         # seats at airlines
#         oheArr[day * len(hrs) + noH, noA + 1] += arrivals.iloc[[i], [day * 3 + 5]].values[0][0]
#         # dests
#         oheArr[day * len(hrs) + noH, noD + len(airlines) + 1] = 1

# 4. write to csv

seatsDep = pd.DataFrame(oheDep, index=daysAug, columns=cols)
# seatsArr = pd.DataFrame(oheArr, index=daysAug, columns=cols)

# seatsDep.to_csv('seats_Dep.csv')
# seatsArr.to_csv('seats_Arr.csv')

# 5. EDA

# 5.1 time

lanes = pd.ExcelFile('lanes.xlsx').parse(header=1)

laneMain, laneAlter = [], []
for h in range(len(hrs)):
    for t in range(lanes.shape[0]):
        time = lanes.iloc[[t], [0]].values[0][0]
        if h >= int(time[:2]) and h < int(time[-4:-2]):
            laneMain.append(lanes.iloc[[t], [1]].values[0][0] + lanes.iloc[[t], [2]].values[0][0])
            laneAlter.append(lanes.iloc[[t], [3]].values[0][0] + lanes.iloc[[t], [4]].values[0][0])
            break

# 5.2 Throughput v.s. Seats

checkpoints = list(set(throughput.iloc[:, [4]].values.reshape(-1))) + ['totalCheck']
psg = pd.DataFrame(index=daysAug, columns=['hours', 'seats'] + checkpoints, dtype=int)
psg['hours'] = seatsDep['hours']
psg['seats'] = seatsDep['seats']

for i in range(throughput.shape[0]):
    noDay = findDay(throughput.iloc[[i], [2]].values[0][0])
    noH = findHour(throughput.iloc[[i], [5]].values[0][0][:2] + '00')
    psg[throughput.iloc[[i], [4]].values[0][0]].iloc[noDay * len(hrs) + noH] = throughput.iloc[[i], [6]].values[0][0]

psg = psg.fillna(0)
psg['totalCheck'] = psg[checkpoints[0]] + psg[checkpoints[1]] + psg[checkpoints[2]]

psg['Main Lanes'] = laneMain * len(days)
psg['Alter Lanes'] = laneAlter * len(days)
psg['Total Lanes'] = psg['Main Lanes'] + psg['Alter Lanes']

psg['Main Time'] = 3600 / (psg['Main Checkpoint']+0.001) * psg['Main Lanes']
psg['Alter Time'] = 3600 / (psg['Alternate Checkpoint']+0.001) * psg['Alter Lanes']
psg['Mean Time'] = 3600 / (psg['totalCheck']+0.001) * psg['Total Lanes']

psg.to_csv('psg_total.csv')

# 5.3 avg

avgSeat, avgCp, avgMain, avgAlter, avgMean = [], [], [], [],[]
for h in range(len(hrs)):
    totalSeat, totalCp, totalMain, totalAlter, totalMean = 0, 0, 0, 0,0
    for day in range(len(days)):
        totalSeat += psg['seats'].iloc[day * len(hrs) + h]
        totalCp += psg['totalCheck'].iloc[day * len(hrs) + h]
        totalMain += psg['Main Time'].iloc[day * len(hrs) + h]
        totalAlter += psg['Alter Time'].iloc[day * len(hrs) + h]
        totalMean += psg['Mean Time'].iloc[day * len(hrs) + h]
    avgSeat.append(totalSeat / len(days))
    avgCp.append(totalCp / len(days))
    avgMain.append(totalMain / len(days))
    avgAlter.append(totalAlter / len(days))
    avgMean.append(totalAlter / len(days))

psgAvg = pd.DataFrame(index=hrs)
psgAvg['Avg Seats'] = avgSeat
psgAvg['Avg Checkpoints'] = avgCp
psgAvg['Avg MainTime'] = avgMain
psgAvg['Avg AlterTime'] = avgAlter
psgAvg['Avg MeanTime'] = avgMean

psgAvg.to_csv('psg_avg.csv')

# 5.4 Dest Total Seats

# destSeats = pd.DataFrame(index=days, columns=dests)
# airSeats = pd.DataFrame(index=days, columns=airlines)
#
# for day in range(len(days)):
#     for d in range(len(dests)):
#         destSeats.iloc[[day], [d]] = departures.groupby(by=['Destination Name']).sum().iloc[d][day * 3 + 3]
#     for a in range(len(airlines)):
#         airSeats.iloc[[day], [a]] = departures.groupby(by=['Airline Name']).sum().iloc[a][day * 3 + 4]
#
# destSeats.to_csv('dest_Seats.csv')
# airSeats.to_csv('air_Seats.csv')
