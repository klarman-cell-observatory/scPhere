import tensorflow as tf
import numpy as np

from scphere.distributions import HyperbolicWrappedNorm
from scphere.distributions import VonMisesFisher
from scphere.distributions import HypersphericalUniform
from scphere.util.util import log_likelihood_nb, log_likelihood_student


EPS = 1e-6
MAX_SIGMA_SQUARE = 1e10


# ==============================================================================
class SCPHERE(object):
    def __init__(self, n_gene, sp, sp_mask, n_batch=None, z_dim=2,
                 encoder_layer=None, decoder_layer=None, activation=tf.nn.elu,
                 latent_dist='vmf', observation_dist='nb',
                 batch_invariant=False, seed=0):
        # n_batch should be a integer specifying the number of batches

        tf.compat.v1.set_random_seed(seed)

        if encoder_layer is None:
            encoder_layer = [128, 64, 32]
        if decoder_layer is None:
            decoder_layer = [32, 128]

        self.batch_invariant = batch_invariant

        self.n_input_feature = n_gene
        # placeholder for gene expression data
        self.x = tf.compat.v1.placeholder(tf.float32,
                                          shape=[None, n_gene], name='x')

        self.z_dim, self.encoder_layer, self.decoder_layer, self.activation, \
            self.latent_dist, self.observation_dist = \
            z_dim, encoder_layer, decoder_layer, activation, \
            latent_dist, observation_dist

        if self.latent_dist is 'vmf':
            self.z_dim += 1

        if type(n_batch) is not list:
            n_batch = [n_batch]

        # placeholder for batch id of x
        self.n_batch = n_batch
        if len(self.n_batch) > 1:
            self.batch_id = tf.compat.v1.placeholder(tf.int32,
                                                     shape=[None, None],
                                                     name='batch')
            self.batch = self.multi_one_hot(self.batch_id, self.n_batch)
        else:
            self.batch_id = tf.compat.v1.placeholder(tf.int32,
                                                     shape=[None],
                                                     name='batch')
            self.batch = tf.one_hot(self.batch_id, self.n_batch[0])

        self.library_size = tf.reduce_sum(self.x, axis=1, keepdims=True,
                                          name='library-size')

        self.z_mu, self.z_sigma_square = self._encoder(self.x, self.batch)
        self.make_encoder = tf.compat.v1.make_template('encoder', self._encoder)

        with tf.name_scope('latent-variable'):
            if self.latent_dist == 'normal':
                self.q_z = tf.distributions.Normal(self.z_mu, self.z_sigma_square)
            elif self.latent_dist == 'vmf':
                self.q_z = VonMisesFisher(self.z_mu, self.z_sigma_square)
            elif self.latent_dist == 'wn':
                self.q_z = HyperbolicWrappedNorm(self.z_mu, self.z_sigma_square)
            else:
                raise NotImplemented
            self.z = self.q_z.sample()

        self.mu, self.sigma_square = self._decoder(self.z, self.batch)
        self.depth_loss = self._depth_regularizer(self.batch)

        with tf.name_scope('ELBO'):
            if self.observation_dist == 'student':
                self.log_likelihood = tf.reduce_mean(
                    log_likelihood_student(self.x,
                                           self.mu,
                                           self.sigma_square,
                                           df=5.0), name="log_likelihood")
            elif self.observation_dist == 'nb':
                self.log_likelihood = tf.reduce_mean(
                    log_likelihood_nb(self.x,
                                      self.mu,
                                      self.sigma_square,
                                      eps=1e-10), name="log_likelihood")

            if self.latent_dist == 'normal':
                self.p_z = tf.distributions.Normal(tf.zeros_like(self.z),
                                                   tf.ones_like(self.z))
                kl = self.q_z.kl_divergence(self.p_z)
                self.kl = tf.reduce_mean(tf.reduce_sum(kl, axis=1))
            elif self.latent_dist == 'vmf':
                self.p_z = HypersphericalUniform(self.z_dim - 1,
                                                 dtype=self.x.dtype)
                kl = self.q_z.kl_divergence(self.p_z)
                self.kl = tf.reduce_mean(kl)
            elif self.latent_dist == 'wn':
                tmp = self._polar_project(tf.zeros_like(self.z_sigma_square))
                self.p_z = HyperbolicWrappedNorm(tmp,
                                                 tf.ones_like(self.z_sigma_square))

                kl = self.q_z.log_prob(self.z) - self.p_z.log_prob(self.z)

                self.kl = tf.reduce_mean(kl)
            else:
                raise NotImplemented

            self.ELBO = self.log_likelihood - self.kl

        self.sp_gene, _ = tf.split(self.x, [11, self.x.shape.as_list()[-1] - 11], axis=-1)

        self.sp_gene = tf.nn.relu(self.sp_gene - 3)
        self.sp_gene = tf.math.log1p(self.sp_gene) / tf.math.log(tf.constant(10.0))

        # spatial coordinates
        self.sp = sp
        self.sp = tf.Variable(self.sp, trainable=False, dtype='float32')

        M1 = 128
        M2 = self.sp.shape[0]

        p1 = tf.matmul(
            tf.expand_dims(tf.reduce_sum(tf.square(self.z_mu), 1), 1),
            tf.ones(shape=(1, M2))
        )
        p2 = tf.transpose(tf.matmul(
            tf.reshape(tf.reduce_sum(tf.square(self.sp), 1), shape=[-1, 1]),
            tf.ones(shape=(M1, 1)),
            transpose_b=True
        ))

        self.grid_dist = tf.sqrt(tf.add(p1, p2) -
                                 2 * tf.matmul(self.z_mu, self.sp, transpose_b=True))

        self.sp_mask = sp_mask

        self.sp_mask_marginal = tf.Variable(self.sp_mask, trainable=False, dtype='float32')
        B = tf.expand_dims(tf.transpose(self.sp_mask_marginal), 0)
        self.cell_sp_mask_marginal = tf.transpose(B, perm=[0, 2, 1])

        # weighting the distances by gene expression
        AA = tf.broadcast_to(tf.expand_dims(self.grid_dist, 2),
                             shape=[128, 64, 11])
        self.cell_sp_mask_marginal = AA * self.cell_sp_mask_marginal

        ##
        self.sp_mask_neg = 100000.0 * (1.0 - self.sp_mask)
        self.sp_mask_neg = tf.Variable(self.sp_mask_neg, trainable=False, dtype='float32')
        self.sp_mask_neg = tf.broadcast_to(tf.expand_dims(self.sp_mask_neg, 0),
                                           shape=[128, 64, 11])

        ##
        self.cell_sp_mask_marginal = self.cell_sp_mask_marginal + self.sp_mask_neg

        tmp = tf.nn.relu(tf.reduce_min(self.cell_sp_mask_marginal, 1) - 0.15)
        tmp *= self.sp_gene
        self.dist_loss = tf.reduce_mean(tf.reduce_sum(tmp, 1))

        # =====
        # Normlize by elements
        aa = np.zeros(11)
        aa[:6] = 1
        aa[9] = 1

        self.sp_mask_weight = self.sp_mask / np.sum(self.sp_mask, axis=0)
        self.sp_mask_weight = tf.Variable(self.sp_mask_weight * aa, trainable=False, dtype='float32')

        A = tf.expand_dims(self.sp_gene, 2)
        B = tf.expand_dims(tf.transpose(self.sp_mask_weight), 0)
        C = A * B
        self.cell_sp_mask = tf.transpose(C, perm=[0, 2, 1])

        AA = tf.broadcast_to(tf.expand_dims(tf.nn.relu(self.grid_dist - 0.15), 2),
                             shape=[128, 64, 11])
        self.cell_dist_mask = AA * self.cell_sp_mask

        self.dist_loss += tf.reduce_sum(tf.reduce_sum(tf.reduce_sum(self.cell_dist_mask, 1), 1))

        ##
        self.session = tf.compat.v1.Session()
        self.saver = tf.compat.v1.train.Saver()

    def _encoder(self, x, batch):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        if self.observation_dist == 'nb':
            x = tf.math.log1p(x)

            # x = tf.nn.l2_normalize(x, axis=-1)
            if self.latent_dist == 'vmf':
                x = tf.nn.l2_normalize(x, axis=-1)

        if not self.batch_invariant:
            x = tf.concat([x, batch], 1)

        with tf.name_scope('encoder-net'):
            h = tf.keras.layers.Dense(units=self.encoder_layer[0],
                                      activation=self.activation,
                                      kernel_regularizer=regularizer)(x)
            h = tf.keras.layers.BatchNormalization()(h)

            for layer in self.encoder_layer[1:]:
                h = tf.keras.layers.Dense(units=layer, activation=self.activation,
                                          kernel_regularizer=regularizer)(h)
                h = tf.keras.layers.BatchNormalization()(h)

            if self.latent_dist == 'normal':
                z_mu = tf.keras.layers.Dense(units=self.z_dim, activation=None)(h)
                z_sigma_square = tf.keras.layers.Dense(units=self.z_dim,
                                                       activation=tf.nn.softplus)(h)
            elif self.latent_dist == 'vmf':
                z_sigma_square = tf.keras.layers.Dense(units=1,
                                                       activation=tf.nn.softplus)(h) + 1
                z_sigma_square = tf.clip_by_value(z_sigma_square, 1, 10000)

                z_mu = tf.compat.v1.layers.dense(h, units=2,
                                                 activation=lambda t: tf.math.abs(t))
                y = tf.compat.v1.layers.dense(h, units=1)
                z_mu = tf.concat([y, z_mu], axis=-1)
                z_mu = tf.nn.l2_normalize(z_mu, axis=-1)
            elif self.latent_dist == 'wn':
                tmp = tf.keras.layers.Dense(units=self.z_dim,
                                            activation=None)(h)
                z_mu = self._polar_project(tmp)
                z_sigma_square = tf.keras.layers.Dense(units=self.z_dim,
                                                       activation=tf.nn.softplus)(h)
            else:
                raise NotImplemented

        return z_mu, z_sigma_square

    def _decoder(self, z, batch):
        regularizer = tf.contrib.layers.l2_regularizer(scale=0.01)

        z = tf.concat([z, batch], 1)
        with tf.name_scope('decoder-net'):
            h = tf.keras.layers.Dense(units=self.decoder_layer[0],
                                      activation=self.activation,
                                      kernel_regularizer=regularizer)(z)
            h = tf.keras.layers.BatchNormalization()(h)

            for layer in self.decoder_layer[1:]:
                h = tf.keras.layers.Dense(units=layer, activation=self.activation,
                                          kernel_regularizer=regularizer)(h)
                h = tf.keras.layers.BatchNormalization()(h)

            if self.observation_dist == 'nb':
                mu = tf.keras.layers.Dense(units=self.x.shape[-1],
                                           activation=tf.nn.softmax)(h)
                mu *= self.library_size

                sigma_square = tf.keras.layers.Dense(units=self.x.shape[-1],
                                                     activation=tf.nn.softplus)(h)
                sigma_square = tf.reduce_mean(sigma_square, 0)
            else:
                mu = tf.keras.layers.Dense(units=self.x.shape[-1],
                                           activation=None)(h)

                sigma_square = tf.keras.layers.Dense(units=self.x.shape[-1],
                                                     activation=tf.nn.softplus)(h)

            sigma_square = tf.clip_by_value(sigma_square, EPS, MAX_SIGMA_SQUARE)

        return mu, sigma_square

    @staticmethod
    def _clip_min_value(x, eps=EPS):
        return tf.nn.relu(x - eps) + eps

    @staticmethod
    def multi_one_hot(index_tensor, depth_list):
        one_hot_tensor = tf.one_hot(index_tensor[:, 0], depth_list[0], axis=1)
        for col in range(1, len(depth_list)):
            next_one_hot = tf.one_hot(index_tensor[:, col], depth_list[col], axis=1)
            one_hot_tensor = tf.concat([one_hot_tensor, next_one_hot], axis=1)

        return one_hot_tensor

    def _polar_project(self, x):
        x_norm = tf.math.reduce_sum(tf.math.square(x), axis=1, keepdims=True)
        x_norm = tf.math.sqrt(self._clip_min_value(x_norm))

        x_unit = x / tf.reshape(x_norm, (-1, 1))
        x_norm = tf.clip_by_value(x_norm, 0, 32)

        z = tf.concat([tf.math.cosh(x_norm), tf.math.sinh(x_norm) * x_unit], axis=1)

        return z

    def _depth_regularizer(self, batch):
        with tf.name_scope('depth-regularizer'):
            samples = tf.random.poisson(lam=self.x * 0.0, shape=[1])
            samples = tf.reshape(samples, tf.shape(self.x))

        z_mu1, z_sigma_square1 = self.make_encoder(
            tf.nn.relu(self.x - samples), batch)

        mean_diff = tf.reduce_sum(tf.pow(self.z_mu - z_mu1, 2), axis=1)
        loss = tf.reduce_mean(mean_diff)

        return loss

    def get_log_likelihood(self, x, batch):
        import numpy as np

        if self.observation_dist == 'nb':
            log_likelihood = log_likelihood_nb(
                self.x,
                self.mu,
                self.sigma_square,
            )
        else:
            dof = 2.0
            log_likelihood = log_likelihood_student(
                self.x,
                self.mu,
                self.sigma_square,
                df=dof
            )
        num_samples = 5

        feed_dict = {self.x: x, self.batch_id: batch}
        log_likelihood_value = 0

        for i in range(num_samples):
            log_likelihood_value += self.session.run(log_likelihood, feed_dict=feed_dict)

        log_likelihood_value /= np.float32(num_samples)

        return log_likelihood_value

    def save_sess(self, model_name):
        self.saver.save(self.session, model_name)

    def load_sess(self, model_name):
        self.saver.restore(self.session, model_name)

    def encode(self, x, batch):
        feed_dict = {self.x: x, self.batch_id: batch}

        return self.session.run(self.z_mu, feed_dict=feed_dict)


class OptimizerVAE(object):

    def __init__(self, model, learning_rate=0.001, depth_loss=True):
        clip_value = 15.0
        clip_norm = 1000.0

        self.loss = -model.ELBO
        if depth_loss & (model.observation_dist == 'nb'):
            self.loss += model.depth_loss

        with tf.name_scope('optimizer'):
            optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate,
                                                         beta1=0.9,
                                                         beta2=0.999,
                                                         epsilon=0.01)

        with tf.name_scope('gradient-clip'):
            trainable_variable = tf.compat.v1.trainable_variables()
            grad = tf.gradients(self.loss, trainable_variable)
            grad_and_var = zip(grad, trainable_variable)

            grad_and_var = [(grad, var) for grad, var in grad_and_var
                            if grad is not None]
            grad, var = zip(*grad_and_var)
            grad, global_grad_norm = tf.clip_by_global_norm(grad,
                                                            clip_norm=clip_norm)

            grad_clipped_and_var = [(tf.clip_by_value(grad[i],
                                                      -clip_value,
                                                      clip_value), var[i])
                                    for i in range(len(grad_and_var))]

        gradient_1d = [tf.reshape(grad, [-1]) for grad, var in
                       grad_clipped_and_var if grad is not None]
        self.max_grad = tf.reduce_max(tf.concat(axis=0, values=gradient_1d))

        self.train_step = optimizer.apply_gradients(grad_clipped_and_var,
                                                    name='minimize_cost')
