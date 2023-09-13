import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


# Get cpu, gpu or mps device for training.
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


##################### 1) Dataset #############################
class LJ_Dataset(Dataset):
    def __init__(self):
        self.n_samples = 10
        self.x_range = np.array([0.9, 2])
        # I set the dtype to float32 because the default for torch weights is also float32
        x_np = np.linspace(self.x_range[0],self.x_range[1],self.n_samples, dtype = np.float32)
        target = torch.from_numpy(self.leonard_jones_toten(x_np))
        # convert these to 2D arrays where the first dimension is the sample # and the second dimension is the descriptors for that sample, in this case just a number
        self.x =torch.reshape( torch.from_numpy(x_np),(len(x_np), 1))                
        self.target = torch.reshape(target,(len(target), 1))

    def leonard_jones_toten(self, x_np):
        """ x is a vector of sampled points in the distribution and it returns a vector of total energies"""
        sigma = 1.0
        epsilon = 1.0
        toten = 4* epsilon * ((sigma/x_np)**12 - (sigma/x_np)**6)
        return toten

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return self.x[idx],  self.target[idx]

    
# note that in this example I use the whole dataset. Never do this! It's just to demonstrate overfitting.
my_dataset = LJ_Dataset()

# Create data loader
batch_size = 10
train_data = my_dataset
train_dataloader = DataLoader(my_dataset, batch_size=batch_size, shuffle=True, drop_last=True )

print(f"Num training samples: {len(train_data)}")
for X, y in train_dataloader:
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")
    break



######################### 2) Model #################################
# note that this model is way too large for this problem. I'm doing this to show overfitting
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        num_basis_functions = 40
        nodes_per_layer = 40
        x_start = 0.9
        x_end = 3
        # gradients will not calculated for mu by default
        self.mu = torch.linspace(x_start,x_end, num_basis_functions) 
        self.sigma = (x_end - x_start)/num_basis_functions
        self.multilayer_perceptron = nn.Sequential(
            nn.Linear(num_basis_functions, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer, nodes_per_layer),
            nn.GELU(),
            nn.Linear(nodes_per_layer,1),
        )

    def gaussian(self,x):
        """ converts a position to a num_basis_function dimensional vector of
        gaussian basis functions. It takes in a n x 1 tensor of batched
        positions, and returns a n x num_basis_functions tensor"""
        return 1/(self.sigma * np.sqrt(2 *np.pi))* torch.exp(-0.5 * (x- self.mu)**2/self.sigma**2)
        
    def forward(self, x):
        # first convert position into a description of gaussian basis function 
        descriptors = self.gaussian(x)
        # then push it through the neural network
        pred = self.multilayer_perceptron(descriptors)
        return pred
        
model = NeuralNetwork().to(device)
print(model)



###################### 3) Loss function ################################
loss_fn = nn.MSELoss()




###################### 4) Minimize loss #################################
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

def train(train_dataloader, model, loss_fn, optimizer):
    """ does a single epoch """
    epoch_loss = 0.
    num_batches = len(train_dataloader.dataset)
    model.train() # turn on training mode
    for batch, (X, y) in enumerate(train_dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)
        epoch_loss += loss.item()
        # calculate the gradients of the loss function using backpropagation
        loss.backward()
        optimizer.step() # change the weights based on this batch of data
        optimizer.zero_grad() # reset all the gradients for the next batch calculation
        return epoch_loss/num_batches

epochs = 100
for epoch in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    if epoch % 5 == 0:
        print(f"Epoch {epoch:3} \t train loss: {train_loss:.4f} ")
# training is done


#################### 5) make a prediction ############################
model.eval() # set the model into evaluation mode
# get the training points
train_x = []
train_y = []
for X, y in train_dataloader:
    train_x.append(X.numpy())
    train_y.append(y.numpy())

# calulate the real LJ curve
x = np.linspace(0.9, 3 , 100, dtype = np.float32)
y_ref = my_dataset.leonard_jones_toten(x)

# calculate the model predictions
with torch.no_grad():
    x = torch.reshape(torch.from_numpy(x),(-1,1))
    y_pred = model(x)    
    
fig, ax = plt.subplots()
ax.plot(x,y_ref, label ="target")
ax.scatter(train_x,train_y, label ="training data")
ax.plot(x,y_pred, label = "prediction")
ax.legend()
plt.savefig("lj_1_results.png")
plt.show()
plt.close()
