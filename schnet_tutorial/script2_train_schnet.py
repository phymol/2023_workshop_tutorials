import os
from ase import Atoms
import numpy as np
import schnetpack as spk
import schnetpack.transform as trn
from schnetpack.data import ASEAtomsData
import torch
import torchmetrics
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import numpy as np


custom_data = spk.data.AtomsDataModule(
    './ethanol_dataset.db', 
    batch_size=10,
    distance_unit='Ang',
    property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
    num_train=100,
    num_val=100,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=0,
    pin_memory=False, # set to false, when not using a GPU
)
custom_data.prepare_data()
custom_data.setup()

print('Number of reference calculations:', len(custom_data.dataset))
print('Number of train data:', len(custom_data.train_dataset))
print('Number of validation data:', len(custom_data.val_dataset))
print('Number of test data:', len(custom_data.test_dataset))


properties = custom_data.dataset[0]
print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
####### define NN ##########

cutoff = 5.
n_atom_basis = 64

pairwise_distance = spk.atomistic.PairwiseDistances() # calculates pairwise distances between atoms
radial_basis = spk.nn.GaussianRBF(n_rbf=50, cutoff=cutoff)
schnet = spk.representation.SchNet(
    n_atom_basis=n_atom_basis, n_interactions=3,
    radial_basis=radial_basis,
    cutoff_fn=spk.nn.CosineCutoff(cutoff)
)

pred_energy = spk.atomistic.Atomwise(n_in=n_atom_basis, output_key='energy') # this uses the key from the dataset
pred_forces = spk.atomistic.Forces(energy_key='energy', force_key='forces')


# check if addoffsets is working
nnpot = spk.model.NeuralNetworkPotential(
    representation=schnet,
    input_modules=[pairwise_distance],
    output_modules=[pred_energy, pred_forces],
    postprocessors=[trn.CastTo64(), trn.AddOffsets('energy', add_mean=True, add_atomrefs=False)]
)

########## setup training ##############
output_energy = spk.task.ModelOutput(
    name='energy',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.01,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

output_forces = spk.task.ModelOutput(
    name='forces',
    loss_fn=torch.nn.MSELoss(),
    loss_weight=0.99,
    metrics={
        "MAE": torchmetrics.MeanAbsoluteError()
    }
)

task = spk.task.AtomisticTask(
    model=nnpot,
    outputs=[output_energy, output_forces],
    optimizer_cls=torch.optim.AdamW,
    optimizer_args={"lr": 1e-4}
)


logger = pl.loggers.TensorBoardLogger(save_dir='.')
callbacks = [
    spk.train.ModelCheckpoint(
        model_path=os.path.join('./', "best_inference_model"),
        save_top_k=1,
        monitor="val_loss"
    )
]

trainer = pl.Trainer(
    callbacks=callbacks,
    logger=logger,
    default_root_dir='.',
    max_epochs=5, # for testing, we restrict the number of epochs
)
trainer.fit(task, datamodule=custom_data)



# tensorboard --logdir=lightning_logs
