import numpy as np
import matplotlib.pyplot as plt


#setting
train_err_data = 'err_train_mlp.npy'
test_err_data = 'err_test_mlp.npy'

train_err_data = 'err_train_lstm.npy'
test_err_data = 'err_test_lstm.npy'

train_err = np.load(train_err_data)
test_err = np.load(test_err_data)

#plot section
plt.plot(np.arange(len(train_err)), train_err , c="r", label = 'Train Error')
plt.plot(np.arange(len(test_err)), test_err, c="g", label = 'Test Error')
plt.legend()
plt.xlabel("Iterasi")
plt.ylabel("Error")
plt.show()
plt.close()
