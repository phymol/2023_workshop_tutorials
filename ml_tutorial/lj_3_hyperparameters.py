import os
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
import math
from torch.utils.tensorboard import SummaryWriter
import datetime
# lennard-jones is quite easy to learn, to see interesting phenomenon
# you may need to use very small numbers of nodes and layers

# things that have yet to be implemented in this example:
# normalizing targets (very important!)
# gridsearch of hyperparameters
# more advanced learning rates: reduce on plateau and early stopping
# many more ML advances

# note that this model is way too large for this problem.
# I'm doing this to show overfitting
hparam_dict = {"num_basis_functions":30,
               "nodes_per_layer":100,
               "num_hidden_layers":6,
               "learning_rate":1e-2,
               "L2_regularization":1e-3,
               "num_train_data":40,
               "max_epochs":2000}
# the number of data points or the number of epochs shouldn't be
# tuned like a hyperparameter (more data is almost always better 
# and early stopping is a good practice), 
# but I include it here so that you can see it's effects. Also if these
# calculations take a long time you can reduce the number of epochs and
# training data to make the calculations manageable
metric_dict = {}

experiment_name = (datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                   + f"_basis{hparam_dict['num_basis_functions']}"
                   + f"_nodes{hparam_dict['nodes_per_layer']}"
                   + f"_layers{hparam_dict['num_hidden_layers']}"
                   + f"_lr{hparam_dict['learning_rate']}"
                   + f"_L2reg{hparam_dict['L2_regularization']}"
                   + f"_data{hparam_dict['num_train_data']}")
log_dir = "runs" +os.sep + experiment_name
writer = SummaryWriter(log_dir) # this creates the logs for tensorboard
writer.add_hparams(hparam_dict, metric_dict, 
                   run_name=os.path.dirname(os.path.realpath(__file__))
                   + os.sep + log_dir)
writer.add_custom_scalars({f'{experiment_name}': {
        'Losses': ['MultiLine', ['loss/train', 'loss/val']]
    }})

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
        self.n_samples = 10000
        self.x_range = np.array([0.9, 2])
        x_np = np.linspace(self.x_range[0],self.x_range[1],self.n_samples,
                           dtype = np.float32)
        target = torch.from_numpy(self.leonard_jones_toten(x_np))
        # convert these to 2D arrays where the first dimension is the sample # 
        # and the second dimension is the descriptors for that sample,
        # in this case just a number
        self.x =torch.reshape( torch.from_numpy(x_np),(len(x_np), 1))                
        self.target = torch.reshape(target,(len(target), 1))

    def leonard_jones_toten(self, x_np):
        """ x is a vector of sampled points and it returns
        a vector of total energies"""
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
# this is not the usual way of splitting data, usually it's something
# like 60% training, 20% val, 20% test, but I've done it this
# way so you can test the data efficiency of the model
frac_train = hparam_dict["num_train_data"]/ my_dataset.n_samples
frac_val = 0.02*(1 - frac_train) # 2% of the remaining data is val
frac_test = 0.98*(1 - frac_train) # 98% of the remaining data is test
train_data, val_data, test_data = torch.utils.data.random_split(my_dataset,
                                                                [frac_train, frac_val, frac_test],
                                                                generator=generator1)


# Create data loader
batch_size = len(train_data) # I load all the data in one batch, not advisable
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, 
                              drop_last=True )
batch_size = len(val_data) # I load all the data in one batch
val_dataloader = DataLoader(val_data, batch_size=batch_size, drop_last=True )

print(f"Num training samples: {len(train_data)}")
print(f"Num validation samples: {len(val_data)}")


######################### 2) Model #################################
class NeuralNetwork(nn.Module):
    def __init__(self,num_basis_functions, nodes_per_layer, num_hidden_layers):
        super().__init__()
        x_start = 0.9
        x_end = 3
        # gradients will not calculated for mu by default
        self.mu = torch.linspace(x_start,x_end, num_basis_functions) 
        self.sigma = (x_end - x_start)/num_basis_functions

        modules = []
        modules.append(nn.Linear(10, 10))
        modules.append(nn.Linear(10, 10))
        self.conv = torch.nn.Sequential()
        self.MLP = nn.Sequential()
        self.MLP.add_module("linear_1", torch.nn.Linear(num_basis_functions,
                                                        nodes_per_layer))
        self.MLP.add_module("GELU_1", torch.nn.GELU())
        for i in range(2,num_hidden_layers):
            self.MLP.add_module(f"linear_{i}", torch.nn.Linear(nodes_per_layer,
                                                               nodes_per_layer))
            self.MLP.add_module(f"GELU_{i}", torch.nn.GELU())
        self.MLP.add_module("final_linear", torch.nn.Linear(nodes_per_layer,1))

    def gaussian(self,x):
        """ converts a position to a num_basis_function dimensional vector of
        gaussian basis functions. It takes in a n x 1 tensor of batched
        positions, and returns a n x num_basis_functions tensor"""
        return 1/(self.sigma * np.sqrt(2 *np.pi))* torch.exp(-0.5 * (x- self.mu)**2/self.sigma**2)
        
    def forward(self, x):
        # first convert position into a description of gaussian basis function 
        x = self.gaussian(x)
        # then push it through the neural network
        pred = self.MLP(x)
        return pred

model = NeuralNetwork(hparam_dict['num_basis_functions'], hparam_dict['nodes_per_layer'],
                      hparam_dict['num_hidden_layers']).to(device)
print(model)



###################### 3) Loss function ################################
loss_fn = nn.MSELoss()



###################### 4) Minimize loss #################################
optimizer = torch.optim.AdamW(model.parameters(),
                              lr=hparam_dict["learning_rate"],
                              weight_decay=hparam_dict["L2_regularization"])

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

print("Hyperparameters:")
for k,v in hparam_dict.items():
    print(f"{k}: {v}")


skip_report = 5
for epoch in range(hparam_dict['max_epochs']):
    train_loss = train(train_dataloader, model, loss_fn, optimizer)
    if epoch % skip_report == 0:
        val_loss = val(val_dataloader, model, loss_fn)
        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("loss/val", val_loss, epoch)
        print(f"epoch {epoch:3} \t train loss: {train_loss:.4f} \t Val loss: {val_loss:>4f} ")
writer.flush()
writer.close()
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
plt.savefig("lj_3_results.png")
plt.show()
plt.close()

# run command in terminal:  tensorboard --logdir=runs --reload_multifile True
# copy the link to your browser: http://localhost:6006/
# go to the custom_scalar tab to see a plot of loss/train and loss/val on the
# same plot

