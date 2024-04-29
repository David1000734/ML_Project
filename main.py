import pandas as pd
import torch            # Neural Network
import torch.nn as nn
import torch.nn.functional as F           # Activation Functions
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#           Data set breakdown:
# https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators

# *************** Main ***************
# List of avalible data files
# NOTE: Will be able to keep track of the 3 output data set, as long
# as "_012_" is in the file name. No other files should have that
# substring in its name anywhere unless it also has 3 outputs.
fileName = ["diabetes_binary_health_indicators.csv", 
            "diabetes_binary_5050split_health_indicators_.csv",
            "diabetes_012_health_indicators.csv"]

'''          DEBUG section, remove later
* Video reference for the code below
https://www.youtube.com/watch?v=JHWqWIoac2I
*** Should change parts of this code ***

* Consideration, will K-Folds be neccessary or will 1
run be sufficient?

* Consider adding a readme that can detail the specifics of each dataset
and other relevent information. 

* Allow to test (NOT train) model on the other two datasets
'''

# Create neural network class
class nnModel(nn.Module):
    """
    Class for the neural network...
    """

    # Input (21) --> h1 (15) --> h2(8) --> output(2)
    def __init__(self, input = 21, h1 = 15, h2 = 10, h3= 5, output = 1):
        """
        Constructor. NeuralNetwork will have 2 hidden layers and take
        21 features as input and output 2 or 3 depending on the dataset.
        Will be utilizing a dropout of 10% for the provided data and
        linear as the input function
        """
        super().__init__()      # Instantiate
        #self.dropout = nn.Dropout(0)              # Drop 10% of the nodes
        self.node1FC = nn.Linear(input, h1)         # Input layer
        self.node2FC = nn.Linear(h1, h2)            # Hidden layer
        self.node3FC = nn.Linear(h2, h3)             # Hidden layer
        self.out     = nn.Linear(h3, output)        # Output layer

    # Currently using relu activation function, should consider using softmax
    def forward(self, x):
        """
        Function for train/testing. Will be utilizing relu as the 
        activation function as softmax, ELU, and others have 
        worst or similar performance.
        
        param x: The dataset to be trained or tested.

        return: Will output 2 or 3 values as it's prediciton. The one
        with the highest value is chosen. 
        """
        #x = self.dropout(x)                 # Dropout specified in constructor
        # Start at input, run through hidden layers
        #mish is better than relu
        x = F.relu(self.node1FC(x))         # Input --> h1
        x = F.relu(self.node2FC(x))         # h1    --> output
        x = F.relu(self.node3FC(x))         # h1    --> output
        
        # Output
        x = self.out(x)         # output -->
        return x
    pass
# Class END

#Function to calculate accuracy using predictions and the target labels 
def calculate_accuracy(predictions, targets):
    # Convert predictions to class labels (indices of the maximum value)
    predicted_labels = predictions.argmax(dim=1)
    # Compare with target labels 
    correct = (predicted_labels == targets).sum().item()
    # Compute accuracy
    accuracy = correct / len(targets)
    return accuracy

def print_Menu():

    """
    Simple print function, will print the user input menu.
    NOTE: Only prints the menu and does not actually ask for
    user input nor does it do a loop.
    """
    # 0: Diabeties health Indicators
    print("0:\t\t Diabeties Health Data")
    # 1: Diabeties 50/50 split
    print("1:\t\t Diabeties 50/50 Split")
    # 2: Diabeties, 0 no diabeties, 1 pre-diabeties, 2 diabetes
    print("2:\t\t Diabetes, Yes, No, or Pre-Diabetes")

    # Explain dataset
    print("Details:\t Explanation of each data set")

    # Exit
    print("Exit:\t\t End Program")
# printMenu, END

def print_Details():
    """
    Simple print function, will explain each dataset.

    For the sake of time and efficiency, we will be hard coding the
    number of features and data points to avoid reading each file. 
    """
    # Header, for all data sets
    print("\n\nEach dataset contains a variety of information regarding " +
          "the person's lifestyle, medical history, and other " + 
          "sensative information. This information range from gender, " +
          "Smoker, Physical Activity, Age, etc.")
    print("All data recieved is gather via surveys by CDC in 2015. \n\n")
    
    # Diabetes Heath data
    print("\t\t Diabeties Health Data: "\
          "21 features and 253,680 data points")
    print("The target variable will either be a " + 
          "0 (no diabetes) or 1 (pre-diabetes or diabetes). \n")
    
    # 50/50 Diabetes
    print("\t\t Diabetes 50/50 Split: "\
          "21 features and 70,692 data points")
    print("The target variable will either be a " + 
          "0 (no diabetes) or 1 (pre-diabetes or diabetes). \n")
    
    # 0, 1, or 2 Diabetes
    print("\t\t Diabetes, Yes, No, or Pre-Diabetes: "\
          "21 features and 253,680 data points")
    print("The target variable will be 0 (no diabetes) " + 
          "1 (pre-diabetes) or 2 (diabetes). \n")
# printDetails, END

# Initilize data set name
data = None

# How many outputs
out_val = 2

# Input menu for data sets
while (True): 
    bool_exit = False       # Should we exit the program?
    print_Menu()            # print our menu

    # Input section
    menuInput = input("Please select an option: ")

    # Processing section
    try:
        # All inputs should be able to be converted to string
        menuInput = str(menuInput)

        # Lowercase all letters
        menuInput = menuInput.lower()
        
        # Check which menu option was picked
        if (menuInput == "exit"):
            bool_exit = True
        elif (menuInput == "details" or menuInput == "detail"):
            print_Details()
        # IF none of the strings are selected, then it must be a number
        elif (int(menuInput) or int(menuInput) == 0):
            # We found an int
            menuInput = int(menuInput)

            # Valid number is found, read the file
            if ((menuInput < len(fileName)) and (menuInput > -1)):
                # Labels are at index 0.
                data = pd.read_csv(fileName[menuInput], header = 0)

                # Our dataset with '_012_" needs 3 values as output
                if "_012_" in fileName[menuInput]:
                    out_val = 3

                # Exit loop
                break
            else:
                # Number is not within range
                print("\tError: Please pick a number within range.")
        else:
            # Else is not possible?...
            pass

    except:
        # If it is not a number, then whatever was inputed was invalid
        print("\tError: Invalid Option")

    # Exit out here to avoid I/O operation on closed file error
    if (bool_exit):
        exit(0)

    print()         # Formating
# While, END

hard_debug = False           # To see each guess
debug      = True            # Debug var.

data = data.sample(frac=1)          # shuffle data

# Get y values
y_actual = data.iloc[:, 0]
y_actual = y_actual.values      # Convert to numpy
# No point normalizing the y values in the first place

# Get x values, only normalize x values
x_values_before = data.iloc[:, 1:]
#x_values = x_values.values      # Convert to numpy
x_values = preprocessing.normalize(x_values_before)
print("x_val: " + str(x_values.shape))


# Keep things consistent for now
#torch.manual_seed(60)       # Not neccesary. Just a random_seed

# Create out model
model = nnModel(output = out_val)

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
#error = nn.MSELoss()

# Optimizer, using Adam
# Grad Funct = AddmmBackward0
opt = torch.optim.Adam(model.parameters(), lr = .01)
#opt = torch.optim.ASGD(model.parameters(), lr=0.01)

#arrays to keep track of acccuracy values for each iteration 
train_accuracy_values = []
val_accuracy_values = []
train_loss_values = []
val_loss_values = [] 

# 100 iterations for now. For this dataset, realisticly we will need
# a much larger epoch
epoch = 500
for i in range(epoch):
    # Start training model
    y_pred = model.forward(x_train)

    y_val = model.forward(x_test)
     # Compute training and validation accuracy
    train_accuracy = calculate_accuracy(y_pred, y_train)
    val_accuracy = calculate_accuracy(y_val, y_test)
    #add the accuracy values to an array 
    train_accuracy_values.append(train_accuracy)
    val_accuracy_values.append(val_accuracy)

    # Measure loss. predicted vs. actual
    loss = error(y_pred, y_train)
    #validaton loss 
    val_loss = error(y_val, y_test)
    # add loss values to an array
    train_loss_values.append(loss)
    val_loss_values.append(val_loss)


    # Print every 10 iterations
    if (i % 10 == 0):
        print("Epoch: %i \t loss: %0.8f" % (i, loss))

    # Tweak some weights and bias
    opt.zero_grad()      # Clear/ Reset Gradient
    loss.backward()      # Back Prop
    opt.step()           # Updates params/ weights
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
two = 0         # How many twos were predicted
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
            elif (y_val.argmax().item() == 1): 
                one += 1
            else:
                two += 1
    # For, END
print("We got %i correct. Accuracy: %0.2f%%" % (correct, ((correct / y_size) * 100)))

# How many did it guess with and without diabetes
if (debug):
    print("Found %i twos, %i ones, and %i zeros." % (two, one, zero))

#find correlation between features and risk of pre-diabetes and diabetes
#filter data with only pre-diabetes and diabetes variables
data_filter = data[data.iloc[:, 0].isin([1, 2])]
#calculate correlation
correlation = data_filter.corr().iloc[0, 1:]
#graph correlation
plt.figure(figsize=(12, 8))
correlation.plot(kind = 'bar')
plt.title('Correlation Between Features and Pre-Diabetes or Diabetes')
plt.xlabel('Features')
plt.ylabel('Correlation')
plt.show()


#show x values before and after normalization 
plt.figure(figsize=(12, 8))
plt.subplot(2,2,1)
plt.hist(x_values_before.values.flatten(), bins= 10, color = 'blue', alpha = 0.7)
plt.title('Histogram of X values before Normalization')
plt.xlabel('X Values')
plt.ylabel('Frequency')
plt.subplot(2,2,2)
plt.hist(x_values.flatten(), bins= 10, color= 'red', alpha = 0.7)
plt.title('Histogram of X values after Normalization')
plt.xlabel('X Values')
plt.ylabel('Frequency')

# graph the training and validation accuracies for each iteration 
plt.subplot(2,2,3)
plt.plot(range(1, epoch + 1), train_accuracy_values, label='Training Accuracy', linestyle='-')
plt.plot(range(1, epoch + 1), val_accuracy_values, label='Validation Accuracy', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

#remove grad from loss values 
loss_train_values = [tensor.detach().numpy() for tensor in train_loss_values]
loss_val_values= [tensor.detach().numpy() for tensor in val_loss_values]
#plot training and validation loss
plt.subplot(2,2,4)
plt.plot(range(1, epoch + 1), loss_train_values, label='Training Loss', linestyle='-')
plt.plot(range(1, epoch + 1), loss_val_values, label='Validation Loss', linestyle='--')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()


plt.tight_layout()
plt.show()
