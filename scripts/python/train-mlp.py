import jax, sys, optax
from pathlib import Path

# Run from the repository root, or with the repo root on PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.python.config import DATA_DIR, TRAINED_MODELS_DIR

import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from flax.training import train_state, checkpoints
import flax.linen as nn
from flax import jax_utils
from scripts.python.ml_models import NeuralOdeWrapper, Simple_MLP
import orbax, torch
import flax.linen as nn
import time
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, random_split
from tqdm import tqdm


class RVEI(Dataset):
    def __init__(self, root_dir, size):
        self.root_dir = root_dir
        self.size = size
        self.data = np.load(self.root_dir, mmap_mode='r')

    def __len__(self):
       return self.size

    def __getitem__(self, index):
        rvei = self.data[index,:,:]
        return rvei


def create_train_state(rng, learning_rate):
  """Creates initial `TrainState`."""
  n_hidden = [6,6]
  mlp = Simple_MLP(2, n_hidden, nn.tanh, coupled=True, scaling_factor=1., kernel_init=jax.nn.initializers.xavier_normal())
  mdl = NeuralOdeWrapper(2, 3, dt=.1, coupled=True, dfun=mlp)
  p = mdl.init(rng, (jnp.ones((10,10,5)), jnp.ones((10,10,1)) ))['params']
  tx=optax.chain(optax.clip_by_global_norm(10.), optax.adam(learning_rate))
  return train_state.TrainState.create(
      apply_fn=mdl.apply, params=p, tx=tx)

@jax.jit
def train_step(state, batch):
  batch, i_ext = batch
  def loss_fn(params):
    output = state.apply_fn(
        {'params': params}, (batch, i_ext),
    )
    loss = jax.jit(loss_t)(output, batch)
    return loss
  grads = jax.jit(jax.grad(loss_fn))(state.params)
  return grads

@jax.jit
def update_model(state, grads):
  return state.apply_gradients(grads=grads)

@jax.jit
def loss_t(traj, X):
    X = X[...,:2]
    loss_bias = jnp.max(X[...,0], axis=0)*2
    squared_loss_vec = jnp.sqrt(jnp.square(X[:,:,0] - traj[:,:,0])+jnp.square(X[:,:,1] - traj[:,:,1])).mean(axis=(0))
    return jnp.dot(loss_bias.T,squared_loss_vec)/squared_loss_vec.shape[0]


@jax.jit
def eval_model_ode(state, batch):
    batch, i_ext = batch
    def loss_fn(params):
        output = state.apply_fn(
            {'params': params}, (batch, i_ext),
        )
        loss = jax.jit(loss_t)(output, batch)
        return loss
    loss = loss_fn(state.params)
    return loss



def train_one_epoch(state, dataloader):
    epoch_loss = []
    testing_loss = []
    running_loss = 0.0
    for cnt, batch in tqdm(enumerate(dataloader), total=(np.ceil(len(train_set)/batch_size))):
        grads = train_step(state, (batch[...,:5], batch[...,-1:]))
        state = update_model(state, grads)
        
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_loss = eval_model_ode(state, (train_batch[...,:5], train_batch[...,-1:]))
    test_loss = eval_model_ode(state, (test_batch[...,:5], test_batch[...,-1:]))
    valid_loss = eval_model_ode(state, (valid_batch[...,:5], valid_batch[...,-1:]))
    testing_loss.append(test_loss)
    return state, epoch_loss, testing_loss, valid_loss

# Training / validation trajectories (from mlp_training_data.tar on Zenodo).
TRAIN_DATA = DATA_DIR / 'data' / 'rvepji.npy'
VALID_DATA = DATA_DIR / 'data' / 'rvepji_valid.npy'

torch.random.manual_seed(0)
rvei = np.load(TRAIN_DATA, mmap_mode='r')
dataset = RVEI(str(TRAIN_DATA), rvei.shape[0])
total = dataset.__len__()
print(total)
train_set, validation_set = random_split(dataset,[90*(total//100),10*(total//100)+np.remainder(total, 100)])

@jax.jit
def custom_collate_fn(batch):
    batch = jnp.stack(batch).transpose(1,0,2)
    return batch
  
batch_size = 8*1
num_epochs = 500
train_loader = DataLoader(dataset=train_set, collate_fn=custom_collate_fn,shuffle=False, batch_size=batch_size, drop_last=True)
validation_loader = DataLoader(dataset=validation_set,collate_fn=custom_collate_fn, shuffle=False, batch_size=batch_size, drop_last=True)
test_batch = rvei[validation_loader.dataset.indices][:,:,:].transpose(1,0,2)
train_batch = rvei[train_loader.dataset.indices][:,:,:].transpose(1,0,2)
valid_batch = np.load(VALID_DATA, mmap_mode='r').transpose(1,0,2)[::10]
# train_set_loss = rvei[train_loader.dataset.indices].transpose(1,0,2)

rng = jax.random.PRNGKey(0)
rng, init_rng = jax.random.split(rng)
state = create_train_state(rng ,0.00001)
del init_rng


from itertools import product
train_loss = []
test_loss = []
valid_loss = []

# Checkpoints are written to data/trained_models/tmp/checkpoint_<epoch>, the same
# location the bifurcation notebook restores from.
ckpt_dir = str(TRAINED_MODELS_DIR)
start = time.time()
# pars, opt_state = state.params, state.opt_state
start = 0
for epoch in range(start, num_epochs):
    state, epoch_loss, testing_loss, valid_loss_i = train_one_epoch(state, train_loader)
    print(f"Epoch: {epoch + 1}, train loss: {np.mean(epoch_loss):.4f}, test loss: {testing_loss[0]:.4f}, valid loss: {valid_loss_i:.4f}", flush=True)
    train_loss.append(np.mean(epoch_loss))
    test_loss.append(testing_loss[0])
    valid_loss.append(np.array(valid_loss_i))
    pars = state.params
    checkpoints.save_checkpoint(ckpt_dir=ckpt_dir+'/tmp',
                    target=pars,
                    step=epoch,
                    overwrite=True,
                    keep=0,
                    keep_every_n_steps=1)
    np.savez(f'{ckpt_dir}/loss.npz', train=np.array(train_loss), test_loss=np.array(test_loss), valid_loss=np.array(valid_loss))

print("Total time: ", time.time() - start, "seconds")

