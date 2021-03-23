import tensorflow as tf
from scphere.util.data import DataSet
from scphere.model.vae_sp import OptimizerVAE


# ==============================================================================
class Trainer(object):
    def __init__(self, x, model, batch_id=None, mb_size=128,
                 learning_rate=0.001, max_epoch=100):

        self.model, self.mb_size, self.max_epoch = \
            model, mb_size, max_epoch

        self.max_iter = int(x.shape[0] / self.mb_size) * \
            self.max_epoch

        if batch_id is None:
            self.x = DataSet(x)
        else:
            self.x = DataSet(x, batch_id)

        self.optimizer = OptimizerVAE(self.model, learning_rate=learning_rate)

        self.status = dict()
        self.status['kl_divergence'] = []
        self.status['log_likelihood'] = []
        self.status['elbo'] = []

        self.session = model.session
        self.session.run(tf.compat.v1.global_variables_initializer())

    def train(self):
        for iter_i in range(self.max_iter):
            x_mb, y_mb = self.x.next_batch(self.mb_size)
            feed_dict = {self.model.x: x_mb,
                         self.model.batch_id: y_mb,
                         }

            self.session.run(self.optimizer.train_step, feed_dict)

            if (iter_i % 50) == 0:
                var = [self.model.log_likelihood,
                       self.model.kl, self.model.ELBO,
                       ]

                log_likelihood, kl_divergence, elbo = \
                    self.session.run(var, feed_dict)

                self.status['log_likelihood'].append(log_likelihood)
                self.status['kl_divergence'].append(kl_divergence)
                self.status['elbo'].append(elbo)

                info_print = {'Log-likelihood': log_likelihood,
                              'ELBO': elbo, 'KL': kl_divergence}
                print(iter_i, '/', self.max_iter, info_print)
