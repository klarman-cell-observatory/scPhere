import numpy as np

from matplotlib import pyplot as plt

from scphere.util.util import read_mtx
from scphere.util.trainer import Trainer
from scphere.model.vae import SCPHERE
from scphere.util.plot import plot_trace

# Preparing a sparse matrix and using ~2000 variable genes for efficiency. 
# Data can be downloaded from single cell portal (login with a Google account):
# https://singlecell.broadinstitute.org/single_cell/study/SCP551/scphere#study-download
data_dir = './example/data/'
mtx = data_dir + 'cd14_monocyte_erythroid.mtx'
x = read_mtx(mtx)
x = x.transpose().todense()

# The batch vector should be 0-based
# For the cases there are no batch vectors, we can set n_batch=0,
# and create an artificial batch vector (just for running scPhere,
# the batch vector will not influence the results), e.g.,
batch = np.zeros(x.shape[0]) * -1

# build the model
# n_gene: the number of genes
# n_batch: the number of batches for each component of the batch vector.
#          For this case, we set it to 0 as there is no need to correct for batch effects. 
# z_dim: the number of latent dimensions, setting to 2 for visualizations
# latent_dist: 'vmf' for spherical latent spaces, and 'wn' for hyperbolic latent spaces
# observation_dist: the gene expression distribution, 'nb' for negative binomial
# seed: seed used for reproducibility
# batch_invariant: batch_invariant=True to train batch-invarant scPhere.
#                  To train batch-invariant scPhere, i.e.,
#                  a scPhere model taking gene expression vectors only as inputs and
#                  embedding them to a latent space. The trained model can be used to map new data,
#                  e.g., from new patients that have not been seen during training scPhere
#                  (assuming patient is the major batch vector)
model = SCPHERE(n_gene=x.shape[1], n_batch=0, batch_invariant=False,
                z_dim=2, latent_dist='vmf',
                observation_dist='nb', seed=0)

# training
# model: the built model above
# x: the UMI count matrix
# max_epoch: the number of epochs used to train scPhere
# mb_size: the number of cells used in minibatch training
# learning_rate: the learning rate of the gradient descent optimization algorithm
trainer = Trainer(model=model, x=x, batch_id=batch, max_epoch=250,
                  mb_size=128, learning_rate=0.001)

trainer.train()

# save the trained model
save_path = './example/demo-out/'

model_name = save_path + 'cd14_mono_eryth_model_250epoch'
model.save_sess(model_name)

# embedding all the data
z_mean = model.encode(x, batch)
np.savetxt(save_path +
           'cd14_mono_eryth_latent_250epoch.tsv',
           z_mean)

# the log-likelihoods
ll = model.get_log_likelihood(x, batch)
np.savetxt(save_path +
           'cd14_mono_eryth_ll_250epoch.tsv',
           z_mean)

# Plotting log-likelihood and kl-divergence at each iteration
plot_trace([np.arange(len(trainer.status['kl_divergence']))*50] * 2,
           [trainer.status['log_likelihood'], trainer.status['kl_divergence']],
           ['log_likelihood', 'kl_divergence'])
# plt.show()

plt.savefig(save_path +
            'cd14_mono_eryth_train.png')

