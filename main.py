import pandas as pd
import torch            # Neural Network
import torch.nn as nn
import torch.nn.functional as F           # Activation Functions
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

#           Data set breakdown:
# https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

# *************** Main ***************
# Labels are at index 0
data = pd.read_csv("diabetes_binary_health_indicators.csv", header = 0)

# DEBUG
# data = pd.read_csv("diabetes_binary_5050split_health_indicators_.csv", header = 0)
# data = pd.read_csv("test.csv", header = 0)

'''
Data set will contain 21 features with a mixture of integers and binary 
along with 253,680 data points.

            *** Data sets ***
* diabetes_binary_5050split: An equal split of people with and without diabetes
0: No diabetes
1: Pre-diabetes or diabetes

* diabetes_binary: Clean dataset where diabetes has 2 classes
0: No diabetes
1: Pre-diabetes or diabetes

* diabetes_012: A clean dataset where diabetes has 3 classes
0: No diabetes
1: Prediabetes
2: Diabetes

'''

'''          DEBUG section, remove later
scaler = MinMaxScaler(feature_range = (0, 1))
data = scaler.fit_transform(data)           # Normalize data
print(scaler.inverse_transform(data))       # Un-normalize data

* Video reference for the code below
https://www.youtube.com/watch?v=JHWqWIoac2I
*** Should change parts of this code ***

* Consideration, will K-Folds be neccessary or will 1
run be sufficient?

* Should there be a user input section where one of the three avalible
data types can be picked?

* Consider adding a readme that can detail the specifics of each dataset
and other relevent information. 

* Should definelty do some normalization and could possible improve accuracy.
********* Current calcuations are done WITHOUT normalization *********
'''

# Create neural network class
class nnModel(nn.Module):

    # Input (21) --> h1 (15) --> h2(8) --> output(2)
    def __init__(self, input = 21, h1 = 15, h2 = 8, output = 2):
        super().__init__()      # Instantiate
        self.dropout = nn.Dropout(0.1)              # Drop 10% of the nodes
        self.node1FC = nn.Linear(input, h1)         # Input layer
        self.node2FC = nn.Linear(h1, h2)            # Hidden layer
        self.out     = nn.Linear(h2, output)        # Output layer

    # Currently using relu activation function, should consider using softmax
    def forward(self, x):
        x = self.dropout(x)                 # Dropout specified in constructor
        # Start at input, run through hidden layers
        x = F.relu(self.node1FC(x))         # Input --> h1
        x = F.relu(self.node2FC(x))         # h1    --> output
        
        # Output
        x = self.out(x)         # output -->
        return x
    pass
# Class END

hard_debug = False           # To see each guess
debug      = True            # Debug var.

# Normalize all data points 
data = preprocessing.normalize(data)
data = pd.DataFrame(data)       # Change back to DataFrame for simplicity

# Get y values
y_actual = data.iloc[:, 0]
y_actual = y_actual.values      # Convert to numpy
# After normalizing, 1 has been turn to 0.___ which will be
# seen as a 0 to the computer. We must change all these values, back to 1
y_actual = y_actual.astype(bool).astype(int)

# Get x values
x_values = data.iloc[:, 1:]
x_values = x_values.values      # Convert to numpy

# Keep things consistent for now
torch.manual_seed(30)       # Not neccesary. Just a random_seed

# Create out model
model = nnModel()

# Split up the data sets
x_train, x_test, y_train, y_test = \
    train_test_split(x_values, y_actual, test_size = 0.2)

# NOTE: It is important that from this point on, the x/y train 
# and test data is of type numpy. torch.Tensor is relying on that

# Train tensors
x_test  = torch.Tensor(x_test)
x_train = torch.Tensor(x_train)

y_test  = torch.LongTensor(y_test)
y_train = torch.LongTensor(y_train)

# Find error
error = nn.CrossEntropyLoss()

# Optimizer, using Adam
opt = torch.optim.Adam(model.parameters(), lr = 0.001)

# 100 iterations for now. For this dataset, realisticly we will need
# a much larger epoch
epoch = 100
for i in range(epoch):
    # Start training model
    y_pred = model.forward(x_train)

    # Measure loss. predicted vs. actual
    loss = error(y_pred, y_train)

    # Print every 10 iterations
    if (i % 10 == 0):
        print("Epoch: %i \t loss: %0.8f" % (i, loss))

    # Tweak some weights and bias
    opt.zero_grad()      # Back propagation
    loss.backward()      
    opt.step()
# For, END

# Evaluate model without back propagation. 
# NOTE: this with is not neccessary, however, it would give
# us insight into how close the neural network to our training
if (debug):
    with torch.no_grad():
        # Evaluate our model using given (x_test) and the
        # predicted values is (y_eval). 
        y_eval = model.forward(x_test)

        # Compare results between actual and predicted
        loss = error(y_eval, y_test)
    print("Loss: " + str(loss))     # How close are we with our original training
# if, END

# Get the size of the testing set
y_size = x_test.shape[0]

correct = 0     # Count correct predictions

# Debug Variables
zero = 0        # How many zeros were predicted
one = 0         # How many ones were predicted
# Find out how many correct it got
with torch.no_grad():
    # Where i is index and value is the data set (all 21 features)
    for i, value in enumerate(x_test):
        # Send data through model
        y_val = model.forward(value)

        # Usefull to see what is actually being guessed
        if (hard_debug and debug):
            print("Actual: %i\t Predicted: %i" % (y_test[i], y_val.argmax().item()))
            pass

        # Our predicted value is the value from the list that is the largest
        # So, find the largest value, then return the index of that value.
        # Effectively, that is our prediction
        if (y_val.argmax().item() == y_test[i]):
            correct += 1

        # Count what was guessed
        if (debug):
            if (y_val.argmax().item() == 0):
                zero += 1
            else: 
                one += 1
    # For, END
print("We got %i correct. Accuracy: %0.2f%%" % (correct, ((correct / y_size) * 100)))

# How many did it guess with and without diabetes
if (debug):
    print("Found %i ones, and %i zeros." % (one, zero))
