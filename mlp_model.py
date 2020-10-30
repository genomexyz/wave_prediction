import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import mean_squared_error

#setting
testing_data_num = 96
epochs = 1000
batch_size = 100
err_threshold = 10**-10
param_filename = 'all_param.npy'
label_filename = 'all_label.npy'

class MLP_wave(nn.Module):
	def __init__(self, input_size=9, hidden_layer_size=10, output_size=1):
		super().__init__()
		self.layer1 = nn.Linear(input_size, hidden_layer_size, bias=True)
		#self.layer2 = nn.Linear(hidden_layer_size, 5, bias=True)
		self.layer_output = nn.Linear(hidden_layer_size, output_size, bias=True)

	def forward(self, masukan):
		x = torch.sigmoid(self.layer1(masukan))
		#x = torch.tanh(self.layer2(x))
		x = self.layer_output(x)
		return x

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
model = MLP_wave(len(training_param[0]), 15, 1)

loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
#optimizer = torch.optim.ASGD(model.parameters())
#optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

error_training = np.zeros(epochs)
error_test = np.zeros(epochs)
for i in range(epochs):
#	for seq, labels in train_inout_seq:
#	for j in range(len(norm_train_param_data)):
	for j in range(0, len(training_param_norm), batch_size):
		optimizer.zero_grad()
		masukan_param_model = training_param_norm[j:j+batch_size]
		#print('check shape masukan', list(masukan_param_model.size()))
		masukan_label_model = torch.reshape(training_label_norm[j:j+batch_size], (-1, 1))
		y_pred = model(masukan_param_model)
		#print('check ini', masukan_label_model.size())

		batch_loss = loss_function(y_pred, masukan_label_model)
		batch_loss.backward()
		optimizer.step()
		
#	if batch_loss.item() < err_threshold: #when model surpass error threshold, we stop training
#		break
	
	#determine error for test data
	optimizer.zero_grad()
	y_pred_test = model(test_param_norm)
	loss_test = loss_function(y_pred_test, test_label_norm)
	error_training[i] = batch_loss.item()
	error_test[i] = loss_test.item()

	print('epoch: %s, train loss: %s, test loss: %s'%(i, batch_loss.item(), loss_test.item()))


#prediction step
result = np.zeros((len(test_param_norm), 2))
norm_pred = model(test_param_norm)
pred = (norm_pred * std_label_train + rata_label_train).detach().numpy()
result[:,0] = np.reshape(pred, (-1))
result[:,1] = np.reshape(test_label, (-1))

print('Data stat:', rata_param_train, rata_label_train, std_param_train, std_label_train)

#save data
np.save('pred_label_mlp.npy', result)
np.save('err_train_mlp.npy', error_training)
np.save('err_test_mlp.npy', error_test)
