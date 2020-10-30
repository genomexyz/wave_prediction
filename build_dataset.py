import numpy as np

#setting
raw_data_file = 'Coastal Data System - Waves (Mooloolaba) 01-2017 to 06 - 2019.csv'
param_seq_len = 6
label_seq_len = 1
flag_no_data = -99.9

raw_data_open = open(raw_data_file)
raw_data = raw_data_open.read()
raw_data_open.close()

raw_data = raw_data.split('\n')[1:] #get rid the headers
if raw_data[-1] == '':
	raw_data = raw_data[:-1]

all_data = []
for i in range(len(raw_data)):
	data = raw_data[i].split(',')[1:]
	data_float = np.array(data)
	data_float = data_float.astype(np.float)
	all_data.append(data_float)

all_data = np.array(all_data)

all_param = []
all_label = []
for i in range(len(all_data)-param_seq_len-label_seq_len):
	candidate_data = all_data[i:i+param_seq_len+label_seq_len, 0]
	if (candidate_data == flag_no_data).any():
		continue
	all_param.append(candidate_data[:param_seq_len])
	all_label.append(candidate_data[param_seq_len:])

#for i in range(len(all_param)):
#	print(all_param[i], all_label[i])

#save model
np.save('all_param.npy', all_param)
np.save('all_label.npy', all_label)
