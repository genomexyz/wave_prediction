import numpy as np
import matplotlib.pyplot as plt


#setting
raw_data_file = 'Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv'

raw_data_open = open(raw_data_file)
raw_data = raw_data_open.read()
raw_data_open.close()

raw_data = raw_data.split('\n')[1:] #get rid the headers
if raw_data[-1] == '':
	raw_data = raw_data[:-1]

hs_all = []
hmax_all = []
sst_all = []
waktu = []
for i in range(len(raw_data)):
	try:
		hs = float(raw_data[i].split(',')[1])
		hmax = float(raw_data[i].split(',')[2])
		sst = float(raw_data[i].split(',')[6])
	except ValueError:
		continue
	waktu.append(raw_data[i].split(',')[0])
#	if hs == -99.9 or hmax == -99.9 or sst == -99.9:
#		continue
	hs_all.append(hs)
	hmax_all.append(hmax)
	sst_all.append(sst)

hs_all = np.array(hs_all)
hmax_all = np.array(hmax_all)
sst_all = np.array(sst_all)

hs_all[hs_all == -99.9] = np.nan
#plt.xticks(np.arange(len(hs_all)), waktu)
plt.plot(np.arange(len(hs_all)), hs_all, c="g", label = 'Hs')
plt.legend()
plt.show()
plt.close()
