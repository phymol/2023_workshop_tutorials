from ase import Atoms
import ase.io
import schnetpack as spk
import schnetpack.transform as trn
import torch
import os
import numpy as np
import matplotlib.pyplot as plt

# set device
device = torch.device("cpu")

# load model
model_path = os.path.join("./", "best_inference_model")
best_model = torch.load(model_path, map_location=device)

# set up converter
converter = spk.interfaces.AtomsConverter(
    neighbor_list=trn.ASENeighborList(cutoff=5.0), dtype=torch.float32, device=device
)

# create atoms object from dataset
ethanol_data = spk.data.AtomsDataModule(
    './ethanol_dataset.db', 
    batch_size=10,
    distance_unit='Ang',
    property_units={'energy':'kcal/mol', 'forces':'kcal/mol/Ang'},
    num_train=1000,
    num_val=1000,
    transforms=[
        trn.ASENeighborList(cutoff=5.),
        # trn.RemoveOffsets("energy", remove_mean=True, remove_atomrefs=False),
        trn.CastTo32()
    ],
    num_workers=0,
    pin_memory=False, # set to false, when not using a GPU
)
ethanol_data.prepare_data()
ethanol_data.setup()

structure = ethanol_data.test_dataset[0]
atoms = Atoms(
    numbers=structure[spk.properties.Z], positions=structure[spk.properties.R]
)


ase_dir = os.path.join("./", 'ase_calcs')
if not os.path.exists(ase_dir):
    os.mkdir(ase_dir)
molecule_path = os.path.join(ase_dir, 'ethanol.xyz')

ase.io.write(molecule_path, atoms)


ethanol_ase = spk.interfaces.AseInterface(
    molecule_path,
    ase_dir,
    model_file=model_path,
    neighbor_list=trn.ASENeighborList(cutoff=5.0),
    energy_key='energy',
    force_key='forces',
    energy_unit="kcal/mol",
    position_unit="Ang",
    device="cpu",
    dtype=torch.float64,
)



ethanol_ase.init_md(
    'simulation'
)
ethanol_ase.run_md(1000)


# Load logged results
results = np.loadtxt(os.path.join(ase_dir, 'simulation.log'), skiprows=1)

# Determine time axis
time = results[:,0]

# Load energies
energy_tot = results[:,1]
energy_pot = results[:,2]
energy_kin = results[:,3]

# Construct figure
plt.figure(figsize=(14,6))

# Plot energies
plt.subplot(2,1,1)
plt.plot(time, energy_tot, label='Total energy')
plt.plot(time, energy_pot, label='Potential energy')
plt.ylabel('E [eV]')
plt.legend()

plt.subplot(2,1,2)
plt.plot(time, energy_kin, label='Kinetic energy')
plt.ylabel('E [eV]')
plt.xlabel('Time [ps]')
plt.legend()

temperature = results[:,4]
print('Average temperature: {:10.2f} K'.format(np.mean(temperature)))
plt.savefig("results.pdf")
plt.close()

#in anaconda prompt run: ase gui
# go to file -> open -> ase_calcs (in left pane) -> simulation.traj
# go to tools -> movies -> play

# when you run this script you will need to delete the ase_calcs folder
# beforehand if it already exists otherwise the simulation.traj is appended
# and np.loadtxt will throw a bug
