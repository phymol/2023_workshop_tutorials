import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math

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
    def __init__(self, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform
        self.n_samples = 16
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
        if self.transform:
            sample_x = self.transform(self.x[idx])
        else:
            sample_x = self.x[idx]
        if self.target_transform:
            sample_target = self.target_transform(self.target[idx])
        else:
            sample_target = self.target[idx]
        return sample_x, sample_target

    
my_dataset = LJ_Dataset()
generator1 = torch.Generator().manual_seed(42)
# 60% of data is used for training, 20% for validation, 20% for testing
train_data, val_data, test_data = torch.utils.data.random_split(my_dataset,[0.6, 0.2, 0.2],  generator=generator1)


# Create data loader
batch_size = len(train_data) # I load all the data in one batch, not advisable
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True )
batch_size = len(val_data) # I load all the data in one batch
val_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=True )

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
        x = self.gaussian(x)
        # then push it through the neural network
        pred = self.multilayer_perceptron(x)
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

def val(dataloader, model, loss_fn):
    """ calculate the validation loss """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval() # turn on evaluation mode
    val_loss = 0.
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            val_loss += loss_fn(pred, y).item()
    return val_loss/ num_batches

    
epochs = 100
skip_report = 5
train_loss_history = []
val_loss_history = []
for epoch in range(epochs):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    if epoch % skip_report == 0:
        val_loss = val(val_dataloader, model, loss_fn)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        print(f"epoch {epoch:3} \t train loss: {train_loss:.4f} \t Val loss: {val_loss:>4f} ")
# training is done


#################### plotting the training progress ############################
    
fig, ax = plt.subplots()
x = np.arange(len(train_loss_history))* skip_report
ax.semilogy(x,train_loss_history, label ="train loss")
ax.semilogy(x,val_loss_history, label ="val loss")
ax.set_xlabel("epoch")
ax.set_ylabel("MSE loss")
ax.legend()
plt.savefig("lj_2_results.png")
plt.show()
plt.close()
