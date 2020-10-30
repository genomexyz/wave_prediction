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
for i in range(len(raw_data)):
	try:
		hs = float(raw_data[i].split(',')[1])
		hmax = float(raw_data[i].split(',')[2])
		sst = float(raw_data[i].split(',')[6])
	except ValueError:
		continue
	if hs == -99.9 or hmax == -99.9 or sst == -99.9:
		continue
	hs_all.append(hs)
	hmax_all.append(hmax)
	sst_all.append(sst)

hs_all = np.array(hs_all)
hmax_all = np.array(hmax_all)
sst_all = np.array(sst_all)

print('correlation between Hs and Hmax', np.corrcoef(hs_all, hmax_all)[0,1])
print('correlation between Hs and SST', np.corrcoef(hs_all, sst_all)[0,1])

#plot section
plt.scatter(hs_all, hmax_all, c="r", alpha=0.5)
plt.xlabel("Hs")
plt.ylabel("Hmax")
plt.show()
plt.close()

plt.scatter(hs_all, sst_all, c="r", alpha=0.5)
plt.xlabel("Hs")
plt.ylabel("SST")
plt.show()
