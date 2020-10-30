import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#setting
testing_data_num = 96
epochs = 1000
batch_size = 100
err_threshold = 10**-10
param_filename = 'all_param.npy'
label_filename = 'all_label.npy'

class LSTM(nn.Module):
	def __init__(self, input_size=1, hidden_layer_size=8, output_size=1):
		super().__init__()
		self.hidden_layer_size = hidden_layer_size

		self.lstm = nn.LSTMCell(input_size, hidden_layer_size)

		self.linear = nn.Linear(hidden_layer_size, output_size)

	def forward(self, input_seq):
		hidden_state = torch.zeros(input_seq.size(0), self.hidden_layer_size)
		cell_state = torch.zeros(input_seq.size(0), self.hidden_layer_size)
		
		for iterasi in range(input_seq.size(1)):
			hidden_state, cell_state = self.lstm(input_seq[:,iterasi,:], (hidden_state, cell_state))
		predictions = self.linear(hidden_state)
		return predictions

all_param = np.load(param_filename)[:3400]
all_label = np.load(label_filename)[:3400]

#separate training and testing
training_param = all_param[:-testing_data_num]
training_label = all_label[:-testing_data_num]

test_param = all_param[-testing_data_num:]
test_label = all_label[-testing_data_num:]

print('jumlah data training', len(training_param))

#normalization
rata_param_train = np.mean(training_param)
rata_label_train = np.mean(training_label)

std_param_train = np.std(training_param)
std_label_train = np.std(training_label)

training_param_norm = ((training_param - rata_param_train) / std_param_train).astype(np.float64)
training_label_norm = ((training_label - rata_label_train) / std_label_train).astype(np.float64)

test_param_norm = ((test_param - rata_param_train) / std_param_train).astype(np.float64)
test_label_norm = ((test_label - rata_label_train) / std_label_train).astype(np.float64)

#convert to torch
training_param_norm = torch.from_numpy(training_param_norm).float()
training_label_norm = torch.from_numpy(training_label_norm).float()
test_param_norm = torch.from_numpy(test_param_norm).float()
test_label_norm = torch.from_numpy(test_label_norm).float()


#cut training data to match batch
training_param_norm = training_param_norm[:(len(training_param_norm)-len(training_param_norm)%batch_size)+1]
training_label_norm = training_label_norm[:(len(training_label_norm)-len(training_label_norm)%batch_size)+1]

#NN section
model = LSTM(1, 5, 1)

loss_function = nn.MSELoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.ASGD(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

error_training = np.zeros(epochs)
error_test = np.zeros(epochs)
for i in range(epochs):
	for j in range(0, len(training_param_norm), batch_size):
		optimizer.zero_grad()
		model.hidden_cell = (torch.zeros(1, len(training_param[0]), model.hidden_layer_size),
						torch.zeros(1, len(training_param[0]), model.hidden_layer_size))
		
		masukan_param_model = torch.reshape(training_param_norm[j:j+batch_size], (-1, len(training_param[0]), 1))
		masukan_label_model = torch.reshape(training_label_norm[j:j+batch_size], (-1, 1))
		
		y_pred = model(masukan_param_model)

		single_loss = loss_function(y_pred, masukan_label_model)
		single_loss.backward()
		optimizer.step()
	
	#determine error for test data
	optimizer.zero_grad()
	y_pred_test = model(torch.reshape(test_param_norm, (-1, len(training_param[0]), 1)))
	loss_test = loss_function(y_pred_test, test_label_norm)
	error_training[i] = single_loss.item()
	error_test[i] = loss_test.item()

	print('epoch: %s, train loss: %s, test loss: %s'%(i, single_loss.item(), loss_test.item()))

#prediction step
result = np.zeros((len(test_param_norm), 2))
norm_pred = model(torch.reshape(test_param_norm, (-1, len(training_param[0]), 1)))
pred = (norm_pred * std_label_train + rata_label_train).detach().numpy()
result[:,0] = np.reshape(pred, (-1))
result[:,1] = np.reshape(test_label, (-1))


print('Data stat:', rata_param_train, rata_label_train, std_param_train, std_label_train)

#save data
np.save('pred_label_lstm.npy', result)
np.save('err_train_lstm.npy', error_training)
np.save('err_test_lstm.npy', error_test)
