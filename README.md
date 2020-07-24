---
output:
  html_document: default
  pdf_document: default
---

Deep generative model embedding single-cell RNA-Seq profiles on hyperspheres or hyperbolic spaces
====================

## System requirements
 scPhere has been tested on Python 3.6.7 on Linux (Red Hat Enterprise Linux Server release 7.5 (Maipo)) and macOS (Sierra Version 10.12.6, High Sierra Version 10.13.6). 

No special hardware is required.


## Installation 

Dependencies

* Python 3.6 or above 
* numpy >= 1.16.4
* scipy >= 1.3.0
* pandas >= 0.21.0
* matplotlib >= 3.1.0
* We used Tensorflow v1.14.0 for all analyses
* We used tensorflow-probability v0.7.0 for all analyses

We can use pip (in terminal) to install these packages, e.g., 

* pip install tensorflow==1.14
* pip install -U tensorflow-probability==0.7.0

Next, after cloning scPhere to a folder (in the terminal, using command `git clone https://github.com/klarman-cell-observatory/scPhere`), we can use `python setup.py install` for installations (we need to change current directory to the scPhere folder). The whole process typically takes < 5 minutes. We will simplify the installation using conda packaging.


## Running scPhere

A demo script (`example/script/demo.py`) shows how to run scPhere using data in the folder `example/data`. 
It takes about 2 minute to run this script. The script will produce four major files in the folder `example/demo-out/`: 

* cd14_mono_eryth_latent_250epoch.tsv (the embeddings in a sphere, which can be visulzied by the PlotSphere R function, below in this document)
* cd14_mono_eryth_ll_250epoch.tsv (the log-likelihoods of cells)
* cd14_mono_eryth_train.png (the log-likelihood and KL-divergence at each iteration)
* cd14_mono_eryth_model_250epoch* (the trained scPhere model that can be used later)


Below we used large example data with complex batch effects from human colon mucosa epithelial cells to show how to run scPhere. 
The data can be downloaded from Single Cell Portal (https://singlecell.broadinstitute.org/single_cell/study/SCP551/scphere#study-download). 
You need a Google account to login to Google cloud for data downloading. 

```python
import pandas as pd
import numpy as np

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from scphere.util.util import read_mtx
from scphere.util.trainer import Trainer
from scphere.model.vae import SCPHERE
from scphere.util.plot import plot_trace

# Preparing a sparse matrix and using ~2000 variable genes for efficiency. 
# Data can be downloaded from single cell portal (login with a Google account):
# https://singlecell.broadinstitute.org/single_cell/study/SCP551/scphere#study-download
data_dir = '/Users/jding/work/scdata/'
mtx = data_dir + 'uc_epi.mtx'
x = read_mtx(mtx)
x = x.transpose().todense()

# The batch vector should be 0-based
batch_p = pd.read_csv(data_dir + 'uc_epi_batch_patient.tsv', header=None)
batch_h = pd.read_csv(data_dir + 'uc_epi_batch_health.tsv', header=None)
batch_l = pd.read_csv(data_dir + 'uc_epi_batch_location.tsv', header=None)

batch = pd.concat([batch_p.iloc[:, 0],
                   batch_h.iloc[:, 0],
                   batch_l.iloc[:, 0]], axis=1).values

# build the model
# n_gene: the number of genes
# n_batch: the number of batches for each component of the batch vector.
#          For this case, there are three batch vectors, 
#          and each with 30, 3, and 2 batches, respectively.
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
model = SCPHERE(n_gene=x.shape[1], n_batch=[30, 3, 2],
                z_dim=2, latent_dist='vmf', batch_invariant=False, 
                observation_dist='nb', seed=0)
                
# For the cases there are no batch vectors, we can set n_batch=0, 
# and create an artificial batch vector (just for running scPhere, 
# the batch vector will not influence the results), e.g.,
# batch = np.zeros(x.shape[0]) * -1
#
# model = SCPHERE(n_gene=x.shape[1], n_batch=0,
#                 z_dim=2, latent_dist='vmf',
#                 observation_dist='nb', seed=0)

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
save_path = '/Users/jding/work/scdata/UC-out/'

model_name = save_path + 'uc_epi_model_250epoch'
model.save_sess(model_name)

# embedding all cells
z_mean = model.encode(x, batch)
np.savetxt(save_path +
           'uc_epi_latent_250epoch.tsv',
           z_mean)

# the log-likelihoods
ll = model.get_log_likelihood(x, batch)
np.savetxt(save_path +
           'uc_epi_ll_250epoch.tsv',
           ll)

# Plotting log-likelihood and kl-divergence at each iteration
plot_trace([np.arange(len(trainer.status['kl_divergence']))*50] * 2,
           [trainer.status['log_likelihood'], trainer.status['kl_divergence']],
           ['log_likelihood', 'kl_divergence'])
plt.show()
plt.savefig(save_path +
            'uc_epi_ll_250epoch_train.png')
            
```

Next we interactively visualize the results in R

```R
library("rgl")
library(densitycut) # https://bitbucket.org/jerry00/densitycut_dev/src/master/
library(robustbase)

PlotSphere = function(x, cluster, col, density=FALSE, legend=FALSE) {
  if (missing(col)) {
    col = distinct.col
  }
  if (density == FALSE) {
    col.point = AssignLabelColor(col, cluster, 
                                 uniq.label = sort(unique(cluster)))
  } else {
    colours = colorRampPalette((brewer.pal(7, "YlOrRd")))(10)
    FUN = colorRamp(colours)
    
    cluster = (cluster - min(cluster)) / diff(range(cluster))
    col.point = rgb(FUN(cluster), maxColorValue=256)
  }
  plot3d(x[, 1:3], col = col.point, 
         xlim=c(-1, 1), ylim=c(-1, 1), zlim=c(-1, 1), 
         box=FALSE, axes=FALSE, xlab='', ylab='', zlab='')
  
  arrow3d(c(0, -1.35, 0), c(0, 1.35, 0), 
          col = 'gray', s=0.04, type='extrusion', lit=FALSE)
  
  spheres3d(0, 0, 0, lit=FALSE, color='#dbd7d2', 
            alpha=0.6, radius=0.99)
  spheres3d(0, 0, 0, radius=0.9999, lit=FALSE, color='gray', 
            front='lines', alpha=0.6)
  
  if (density == FALSE) {
    id = !duplicated(cluster)
    col.leg = AssignLabelColor(col, cluster)[id]
    leg = cluster[id]
    names(col.leg) = leg
    
    if (legend == TRUE) {
      legend3d("topright", legend = leg, 
               pch = 16, col = col.leg, cex=1, inset=c(0.02)) 
    }
    
    cluster.srt = sort(unique(cluster))
    k.centers = sapply(cluster.srt, function(zz) {
      cat(zz, '\t')
      id = cluster == zz
      center = colMedians(as.matrix(x[id, , drop=FALSE]))
    })
    
    k.centers = t(k.centers)
    
    cluster.size = table(cluster)[as.character(cluster.srt)]
    id = which(cluster.size > 0)
    
    if (length(id) > 0) {
      k.centers = k.centers[id, , drop=FALSE]
      cluster.srt = cluster.srt[id]
    }
    
    k.centers = k.centers / sqrt(rowSums(k.centers^2)) * 1.15
    text3d(k.centers, texts=cluster.srt, col='black')
  }
}

# Data can be downloaded from single cell portal (login with a Google account):
# https://singlecell.broadinstitute.org/single_cell/study/SCP551/scphere#study-download
cell.type = read.delim('/Users/jding/work/scdata/uc_epi_celltype.tsv', 
  header=FALSE, stringsAsFactors = FALSE)[,1]

x = read.delim('/Users/jding/work/scdata/UC-out/uc_epi_latent_250epoch.tsv', 
  sep=' ', header=FALSE)

PlotSphere(x, cell.type)

# you can save the 3d plots as png file or html file
rgl.snapshot('/Users/jding/work/scdata/UC-out/uc_epi_latent_250epoch_scphere.png',  fmt='png')

browseURL(writeWebGL(filename=paste('/Users/jding/work/scdata/UC-out/', 
                                    'uc_epi_latent_250epoch_scphere.html', sep='/'),
                     width=1000))
                     
# People may still want to show the 3D plots in 2D. 
# We can simply convert the 3D cartesian coordinates to spherical coordinates using the car2sph function.

library(sphereplot)
y = car2sph(x)

col = AssignLabelColor(distinct.col, cell.type)
NeatPlot(y[, 1:2], col=col, cex=0.25, 
         cex.axis=1, xaxt='n', yaxt='n')

```



