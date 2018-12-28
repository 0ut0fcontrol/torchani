# -*- coding: utf-8 -*-
"""
.. _training-example:

Train Your Own Neural Network Potential
=======================================

This example shows how to use TorchANI train your own neural network potential.
"""

###############################################################################
# To begin with, let's first import the modules we will use:
import torch
import ignite
import torchani
import timeit
import tensorboardX
import os
import ignite.contrib.handlers
from pearson import PearsonMetric


###############################################################################
# Now let's setup training hyperparameters. Note that here for our demo purpose
# , we set both training set and validation set the ``ani_gdb_s01.h5`` in
# TorchANI's repository. This allows this program to finish very quick, because
# that dataset is very small. But this is wrong and should be avoided for any
# serious training. These paths assumes the user run this script under the
# ``examples`` directory of TorchANI's repository. If you download this script,
# you should manually set the path of these files in your system before this
# script can run successfully.

# training and validation set
try:
    path = os.path.dirname(os.path.realpath(__file__))
except NameError:
    path = os.getcwd()

validation_path = "/home/yangjc/tmp/dud/ace/ligonly.train.h5"
training_path = "/home/yangjc/tmp/dud/ace/ligonly.test.h5"

# checkpoint file to save model when validation RMSE improves
model_checkpoint = 'model.pt'

# max epochs to run the training
max_epochs = 100

# Compute training RMSE every this steps. Since the training set is usually
# huge and the loss funcition does not directly gives us RMSE, we need to
# check the training RMSE to see overfitting.
training_rmse_every = 5

# device to run the training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# batch size
batch_size = 64

# log directory for tensorboardX
log = 'runs'


###############################################################################
# Now let's read our constants and self energies from constant files and
# construct AEV computer.
const_file = os.path.join(path, 'dud/rHCNO-5.2R_16-3.5A_a4-8.params')  # noqa: E501
sae_file = os.path.join(path, 'pdbbind/sae_linfit.dat')  # noqa: E501
consts = torchani.neurochem.Constants(const_file)
from torchani.aev import ligand_neighborlist
aev_computer = torchani.AEVComputer(**consts, neighborlist_computer=ligand_neighborlist)
energy_shifter = torchani.neurochem.load_sae(sae_file)


###############################################################################
# Now let's define atomic neural networks. Here in this demo, we use the same
# size of neural network for all atom types, but this is not necessary.
def atomic():
    model = torch.nn.Sequential(
        # torch.nn.Linear(384, 128),
        # torch.nn.Linear(1008, 128),
        torch.nn.Linear(1280, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 128),
        torch.nn.CELU(0.1),
        torch.nn.Linear(128, 64),
        torch.nn.CELU(0.1),
        torch.nn.Linear(64, 1)
    )
    return model


nn = torchani.ANIModel([atomic() for _ in range(7)])
print(nn)

###############################################################################
# If checkpoint from previous training exists, then load it.
if os.path.isfile(model_checkpoint):
    nn.load_state_dict(torch.load(model_checkpoint))
else:
    torch.save(nn.state_dict(), model_checkpoint)

model = torch.nn.Sequential(aev_computer, nn).to(device)

###############################################################################
# Now setup tensorboardX.
writer = tensorboardX.SummaryWriter(log_dir=log)

###############################################################################
# Now load training and validation datasets into memory. Note that we need to
# subtracting energies by the self energies of all atoms for each molecule.
# This makes the range of energies in a reasonable range. The second argument
# defines how to convert species as a list of string to tensor, that is, for
# all supported chemical symbols, which is correspond to ``0``, which
# correspond to ``1``, etc.
training = torchani.data.BatchedANIDataset(
    training_path, consts.species_to_tensor, batch_size, device=device,
    # transform=[energy_shifter.subtract_from_dataset])
    transform=[])

validation = torchani.data.BatchedANIDataset(
    validation_path, consts.species_to_tensor, batch_size, device=device,
    # transform=[energy_shifter.subtract_from_dataset])
    transform=[])

###############################################################################
# When iterating the dataset, we will get pairs of input and output
# ``(species_coordinates, properties)``, where ``species_coordinates`` is the
# input and ``properties`` is the output.
#
# ``species_coordinates`` is a list of species-coordinate pairs, with shape
# ``(N, Na)`` and ``(N, Na, 3)``. The reason for getting this type is, when
# loading the dataset and generating minibatches, the whole dataset are
# shuffled and each minibatch contains structures of molecules with a wide
# range of number of atoms. Molecules of different number of atoms are batched
# into single by padding. The way padding works is: adding ghost atoms, with
# species 'X', and do computations as if they were normal atoms. But when
# computing AEVs, atoms with species `X` would be ignored. To avoid computation
# wasting on padding atoms, minibatches are further splitted into chunks. Each
# chunk contains structures of molecules of similar size, which minimize the
# total number of padding atoms required to add. The input list
# ``species_coordinates`` contains chunks of that minibatch we are getting. The
# batching and chunking happens automatically, so the user does not need to
# worry how to construct chunks, but the user need to compute the energies for
# each chunk and concat them into single tensor.
#
# The output, i.e. ``properties`` is a dictionary holding each property. This
# allows us to extend TorchANI in the future to training forces and properties.
#
# We have tools to deal with these data types at :attr:`torchani.ignite` that
# allow us to easily combine the dataset with pytorch ignite. These tools can
# be used as follows:
container = torchani.ignite.Container({'energies': model})
optimizer = torch.optim.Adam(model.parameters())
trainer = ignite.engine.create_supervised_trainer(
    container, optimizer, torchani.ignite.MSELoss('energies'))
evaluator = ignite.engine.create_supervised_evaluator(container, metrics={
        'RMSE': torchani.ignite.RMSEMetric('energies')
    })

# test_pred = model(next(validation))
# print(test_pred)
###############################################################################
# Let's add a progress bar for the trainer
pbar = ignite.contrib.handlers.ProgressBar()
pbar.attach(trainer)


###############################################################################
# And some event handlers to compute validation and training metrics:
def hartree2kcal(x):
    return 627.509 * x


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def validation_and_checkpoint(trainer):
    def evaluate(dataset, name):
        evaluator = ignite.engine.create_supervised_evaluator(
            container,
            metrics={
                #'RMSE': torchani.ignite.RMSEMetric('energies'),
                'pearson': PearsonMetric('energies')
            }
        )
        evaluator.run(dataset)
        metrics = evaluator.state.metrics
        # rmse = hartree2kcal(metrics['RMSE'])
        # rmse = metrics['RMSE']
        rmse = metrics['pearson']
        writer.add_scalar(name, rmse, trainer.state.epoch)

    # compute validation RMSE
    evaluate(validation, 'validation_pearson_vs_epoch')

    # compute training RMSE
    #if trainer.state.epoch % training_rmse_every == 1:
    #    evaluate(training, 'training_rmse_vs_epoch')
    evaluate(training, 'training_pearson_vs_epoch')

    # checkpoint model
    torch.save(nn.state_dict(), model_checkpoint)


###############################################################################
# Also some to log elapsed time:
start = timeit.default_timer()


@trainer.on(ignite.engine.Events.EPOCH_STARTED)
def log_time(trainer):
    elapsed = round(timeit.default_timer() - start, 2)
    writer.add_scalar('time_vs_epoch', elapsed, trainer.state.epoch)


###############################################################################
# Also log the loss per iteration:
@trainer.on(ignite.engine.Events.ITERATION_COMPLETED)
def log_loss(trainer):
    iteration = trainer.state.iteration
    writer.add_scalar('loss_vs_iteration', trainer.state.output, iteration)


###############################################################################
# And finally, we are ready to run:
trainer.run(training, max_epochs)
