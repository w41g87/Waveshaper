import sys, os, datetime
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import math
import tensorflow as tf
import bessel
from tensorflow.keras import layers, models, losses
from keras.datasets import mnist
from pathlib import Path
from tqdm.autonotebook import trange

# helpers
def po2(n):
    x = 1
    while 2**x < n:
        x += 1
    return 2**x

def zipWith(arr1, arr2, func, dim):
    if (len(arr1) != len(arr2)):
        raise Exception('zipWith: Array dimension mismatch')
    output = np.empty((len(arr1), dim))
    for i in range(len(arr1)):
        output[i] = func(arr1[i], arr2[i])

    return output

@tf.keras.saving.register_keras_serializable()
class JsiKernel:
    def __init__(self, nodes, npump=40, width=40, npass=1, mlength=3, nbessel=8, bw=(lambda x: 1), strides=1, **kwargs):
        self.nodes = nodes
        self.npass = npass
        self.npump = npump
        self.mlength = mlength
        self.width = width
        self.strides = strides
        self.nbessel = nbessel if nbessel <= 8 else 8
        if nbessel > 8:
            warnings.warn("Power series of the Bessel function is only available up to the 8th power, but \'" + str(nbessel) + "\' is provided")
        self.nodes_t = (nodes + npass * mlength * 2) * strides
        self.pi = tf.constant(np.pi, dtype=tf.complex64)
        self.zero = tf.constant(0, dtype=tf.complex64)
        self.one = tf.constant(1, dtype=tf.complex64)
        self.pmask = tf.constant(self.mask())
        self.besselRoots = tf.constant(bessel.roots, dtype=tf.float32)
        self.besselSeries = tf.constant(bessel.series, dtype=tf.float32)
        self.besselIndices = tf.concat([tf.range(self.mlength - 1, 0, -1, dtype=tf.int32), tf.range(self.mlength + 1, dtype=tf.int32)], 0)
        self.cmplxcoef = tf.math.pow(1j, tf.cast(self.besselIndices, dtype=tf.complex64))
        # self.bw = tf.map_fn(lambda x: tf.map_fn(bw, x), tf.constant(np.rot90(self.mask()), dtype=tf.complex64))
        self.bw = tf.cast(bw(tf.constant(np.rot90(self.mask()), dtype=tf.float32)), dtype=tf.complex64)
                                     
    def get_config(self):
        return {'nodes': self.nodes, 
                'npass': self.npass, 
                'mlength': self.mlength, 
                'npump': self.npump, 
                'nbessel': self.nbessel,
                'width': self.width, 
                'strides': self.strides
               }
    
    def mask(self):
        output = np.zeros((self.nodes_t, self.nodes_t, 1), dtype=np.int32)
        for index, _ in np.ndenumerate(output):
            output[index] = index[1] - index[0] if np.abs(index[0] + index[1] + 1 - self.nodes_t) <= np.round(self.width / 2) else self.npump # large enough number s.t. it is excluded from substitution
        return output
    
    @tf.function
    def eom2Bessel(self, eom):
        root = tf.cast(tf.math.argmin(tf.abs(self.besselRoots - eom)), dtype=tf.int32)
        r = eom - self.besselRoots[root]
        return tf.map_fn(lambda x: self.besselJ(x, root, r), self.besselIndices, fn_output_signature=tf.float32)
    
    @tf.function
    def besselJ(self, n, root, r):
        output = 0.0
        for i in tf.range(self.nbessel + 1):
            output *= r
            output += self.besselSeries[n, root, self.nbessel - i]
        return output
    
    # @tf.function
    # def call_flat(self, pred):
    #     js = tf.cast(tf.reshape(pred[0:self.n_rings * self.length], (self.n_rings, self.length)), dtype=tf.complex64) * tf.exp(1j * tf.cast(tf.reshape(pred[self.n_rings * self.length:self.n_rings * self.length * 2], (self.n_rings, self.length)), dtype=tf.complex64))
    #     jr = tf.cast(tf.reshape(pred[self.n_rings*self.length*2 : self.n_rings*self.length*2+self.nodes_t*2*self.n_rings], (self.n_rings, self.nodes_t*2)), dtype=tf.complex64)
    #     g = tf.cast(pred[-1], dtype=tf.complex64)
    #     y0s = tf.cast(tf.reshape(pred[self.n_rings*self.length*2+self.nodes_t*2*self.n_rings:self.n_rings*self.length*2+self.nodes_t*4*self.n_rings], (self.n_rings, self.nodes_t*2)), dtype=tf.complex64)
    #     return self(js, jr, g, y0s)
        
    @tf.function
    def __call__(self, pump, ps, eom):
        x = tf.zeros((self.nodes_t, self.nodes_t, 1), dtype=tf.complex64)
        for i in tf.range(self.npump):
            index = i - tf.cast(tf.math.floordiv(self.npump, 2), dtype=tf.int32)
            x += tf.where(tf.equal(self.pmask, index), pump[index], 0)
            
        x *= self.bw
        for i in tf.range(self.npass):
            # pulse shaper
            # signal
            x *= tf.reshape(ps[i][0], (self.nodes_t, 1, 1))
            
            # idler
            x *= tf.reshape(ps[i][1], (1, self.nodes_t, 1))
            
            # eom
            real = tf.expand_dims(tf.math.real(x), axis=0)
            imag = tf.expand_dims(tf.math.imag(x), axis=0)
            eoms = tf.cast(self.eom2Bessel(tf.math.real(eom[i][0])), dtype=tf.complex64) * self.cmplxcoef
            eomi = tf.cast(self.eom2Bessel(tf.math.real(eom[i][1])), dtype=tf.complex64) * self.cmplxcoef
            # signal
            sig_real = tf.nn.conv2d(real, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.real(eoms), axis=-1), axis=-1), axis=-1), 1, padding="SAME") \
                        - tf.nn.conv2d(imag, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.imag(eoms), axis=-1), axis=-1), axis=-1), 1, padding="SAME")
            sig_imag = tf.nn.conv2d(real, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.imag(eoms), axis=-1), axis=-1), axis=-1), 1, padding="SAME") \
                        + tf.nn.conv2d(imag, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.real(eoms), axis=-1), axis=-1), axis=-1), 1, padding="SAME")
            # idler
            idl_real = tf.nn.conv2d(sig_real, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.real(eomi), axis=-1), axis=-1), axis=0), 1, padding="SAME") \
                        - tf.nn.conv2d(sig_imag, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.imag(eomi), axis=-1), axis=-1), axis=0), 1, padding="SAME")
            idl_imag = tf.nn.conv2d(sig_real, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.imag(eomi), axis=-1), axis=-1), axis=0), 1, padding="SAME") \
                        - tf.nn.conv2d(sig_imag, tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.math.real(eomi), axis=-1), axis=-1), axis=0), 1, padding="SAME")

            x = tf.complex(idl_real[0], idl_imag[0])
        x = layers.AveragePooling2D((self.strides, self.strides), (self.strides, self.strides))(tf.reshape(tf.math.real(tf.math.conj(x) * x), (1, self.nodes_t, self.nodes_t, 1)))
        return tf.cast(tf.squeeze(x), dtype=tf.complex64)[self.npass * self.mlength : self.nodes + self.npass * self.mlength, self.npass * self.mlength : self.nodes + self.npass * self.mlength]

# @tf.keras.saving.register_keras_serializable()
# class JsiError(losses.Loss):
#     def __init__(self, kernel=JsiKernel(28, 0, 5, 5, 5), reduction=losses.Reduction.AUTO, name='jsi_error'):
#         super().__init__(reduction=reduction, name=name)
#         self.kernel = kernel

#     def get_config(self):
#         base_config = super().get_config()
#         return {**base_config, 'kernel': self.kernel}
        
#     def _error_calc(self, params):
#         true, pred = params
#         y_pred = self.kernel.call_flat(pred)
#         y_pred = tf.linalg.l2_normalize(y_pred)
#         has_nan = tf.math.reduce_any(tf.math.is_nan(tf.cast(y_pred, dtype=tf.float32)))
#         if has_nan:
#             return tf.constant(1.0, dtype=tf.float32)
#         y_true = tf.cast(true[:, :, 0], tf.complex64)
#         output = tf.cast(
#             1 - tf.abs( tf.tensordot(tf.math.conj(y_true), y_pred, axes=2) * tf.tensordot(tf.math.conj(y_pred), y_true, axes=2) )
#                        / (tf.abs((tf.tensordot(tf.math.conj(y_pred),  y_pred, axes=2)) * tf.tensordot(tf.math.conj(y_true), y_true, axes=2)))
#             , tf.float32)
#         return tf.constant(1.0, dtype=tf.float32) if tf.math.is_nan(output) else output
#     def call(self, true, pred):
#         error = tf.map_fn(self._error_calc, (true, pred), dtype=tf.float32)
#         return tf.reduce_mean(error)

# returns heatmap figure of magnitude and phase
def pltSect(input, x, y, sx, sy):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), dpi=100)
    
    heatMap_mag = sns.heatmap(np.abs(input[x:sx + x, y:sy + y]), ax = ax[0], linewidth = 0, annot = False, cmap = "viridis")
    heatMap_mag.set_title("Magnitude Plot")
    heatMap_mag.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_mag.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_mag.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_mag.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)

    heatMap_ph = sns.heatmap(np.angle(input[x:sx + x, y:sy + y]), ax = ax[1], vmin = 0, vmax = np.pi * 2, linewidth = 0, annot = False, cmap = sns.color_palette("blend:#491C62,#8E2E71,#C43F66,#E3695C,#EDB181,#E3695C,#C43F66,#8E2E71,#491C62", as_cmap=True))
    heatMap_ph.set_title("Phase Plot")
    heatMap_ph.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_ph.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_ph.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_ph.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)
    try:
        get_ipython
        fig.canvas.layout.width = '100%'
        fig.canvas.layout.height = '100%'
        fig.canvas.layout.overflow = 'scroll'
        fig.canvas.layout.padding = '0px'
        fig.canvas.layout.margin = '0px'
    except:
        pass

    return fig

def pltCtst(target, approx, x, y, sx, sy, normalize=True):
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(14, 5), dpi=100)
    
    heatMap_t = sns.heatmap(np.abs(target[x:sx + x, y:sy + y]), ax = ax[0], linewidth = 0, annot = False, cmap = "viridis")
    heatMap_t.set_title("Target")
    heatMap_t.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_t.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_t.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_t.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)

    heatMap_a = sns.heatmap(np.abs(approx[x:sx + x, y:sy + y]) / np.max(np.abs(approx[x:sx + x, y:sy + y])), ax = ax[1], linewidth = 0, annot = False, cmap = "viridis")
    heatMap_a.set_title("Approximation (Normalized)" if normalize else "Approximation")
    heatMap_a.set_xticks(np.arange(0, sx, int(np.ceil(sx / 20))))
    heatMap_a.set_xticklabels(np.arange(x, x + sx, int(np.ceil(sx / 20))))
    heatMap_a.set_yticks(np.arange(0, sy, int(np.ceil(sy / 20))))
    heatMap_a.set_yticklabels(np.arange(-y, -y - sy, int(-np.ceil(sy / 20))), rotation = 0)
    try:
        get_ipython
        fig.canvas.layout.width = '100%'
        fig.canvas.layout.height = '100%'
        fig.canvas.layout.overflow = 'scroll'
        fig.canvas.layout.padding = '0px'
        fig.canvas.layout.margin = '0px'
    except:
        pass

    return fig


def jsi_backprop(init, EPOCHS=None, lr=None, train=True):
    if train and not EPOCHS:
        raise ValueError("training parameter \'EPOCHS\' not provided")
    
    nodes = init.get('nodes')
    npass = init.get('npass')
    mlength = init.get('mlength')
    npump = init.get('npump')
    nbessel = init.get('nbessel')
    width = init.get('width')
    strides = init.get('strides')
    target = init.get('target', None)
    
    data = init.copy()
    nodes_t = (nodes + 2 * mlength * npass) * strides

    losses = np.zeros(int(EPOCHS if EPOCHS else 1), dtype=np.float32)
    kernel = JsiKernel(**init)

    @tf.function
    def eom_constraint(x):
        # positive real and < 45 (bessel approximation constraint)
        return tf.cast(tf.clip_by_value(tf.abs(tf.math.real(x)), 0, 45), tf.complex64)

    @tf.function
    def gainless(x):
        mag = tf.math.abs(x)
        phase = tf.cast(tf.math.angle(x), dtype=tf.complex64)
        return tf.cast(tf.clip_by_value(mag, 0.0, 1.0), dtype=tf.complex64) * tf.math.exp(1j * phase)

    pump = tf.Variable(init.get('pump', np.ones((npump, ), dtype=np.complex64)), name="pump", dtype=tf.complex64)
    ps = tf.Variable(init.get('ps', np.ones((npass, 2, nodes_t), dtype=np.complex64)), name="pulse_shaper", dtype=tf.complex64, constraint=gainless)
    eom = tf.Variable(init.get('eom', np.ones((npass, 2), dtype=np.complex64) * 0.0), name="eo_modulator", dtype=tf.complex64, constraint=eom_constraint)
    target_t = tf.zeros(shape=(nodes, nodes), dtype=tf.complex64)
    min_loss = 1.0
    best_params = [tf.Variable(p) for p in [pump, ps, eom]]
    
    if not target is None:
        target_t = tf.constant(target, shape=(nodes, nodes), dtype=tf.complex64)
    output = None
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr if lr else 
                                         tf.keras.optimizers.schedules.ExponentialDecay(
                                             1e-3, decay_steps=100, decay_rate=0.9, staircase=True))
    optimizer2 = tf.keras.optimizers.Adam(learning_rate=lr*1e3 if lr else 
                                         tf.keras.optimizers.schedules.ExponentialDecay(
                                             1e0, decay_steps=100, decay_rate=0.9, staircase=True))

    @tf.function(input_signature=(
        tf.TensorSpec(shape=tf.shape(target_t), dtype=tf.complex64),
    ))
    def loss_func(pred):
        # RMS
        # return tf.sqrt(tf.reduce_mean(tf.math.real(tf.math.conj(output - target_t) * (output - target_t))))

        # vector projection
        pred_norm = tf.linalg.l2_normalize(pred)
        return tf.math.real(
                1 - tf.abs( tf.tensordot(tf.math.conj(target_t), pred_norm, axes=2) * tf.tensordot(tf.math.conj(pred_norm), target_t, axes=2) )
                           / (tf.abs((tf.tensordot(tf.math.conj(pred_norm),  pred_norm, axes=2)) * tf.tensordot(tf.math.conj(target_t), target_t, axes=2)))
                )

    @tf.function
    def train_step(loss_func, model, pump, ps, eom):
        with tf.GradientTape() as tape:
            pred = model(pump, ps, eom)
            loss = loss_func(pred)
            if tf.math.reduce_any(tf.math.is_nan(tf.math.real(pred))):
                return tf.constant(-1, dtype=tf.float32)
            else:
                gradients = tape.gradient(loss, [pump, ps, eom])  # Compute gradients
                # clipped = [tf.cast(tf.clip_by_value(tf.abs(grad), -0.1, 0.1), dtype=tf.complex64) * tf.exp(1j * tf.cast(tf.math.angle(grad), dtype=tf.complex64)) for grad in gradients]
                optimizer.apply_gradients(zip(gradients[0:2], [pump, ps]))  # Update weights
                # optimizer2.apply_gradients(zip(gradients[2:3], [eom]))
                return loss
        
    try:
        tf.profiler.experimental.stop()
    except:
        pass
    try:
        # tf.profiler.experimental.start('log/' + datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S') + '.log')
        if train:
            # Create a checkpoint object
            ckpt = tf.train.Checkpoint(optimizer=optimizer, params=[pump, ps, eom])
            for j in trange(int(EPOCHS), desc="iterations"):
                loss = train_step(loss_func, kernel, pump, ps, eom)
                if loss.numpy() == -1: 
                    warnings.warn("Interrupted due to encountering NaN in the resulting JSI\n")
                    break
                else: 
                    losses[j] = loss.numpy()
                    if loss < min_loss:
                        min_loss = loss
                        best_params = [old.assign(new) for old, new in zip(best_params, [pump, ps, eom])]
                if j % 100 == 0:
                    ckpt.save('./chkpt/chkpt')
        else:
            min_loss = loss_func(kernel(pump, ps, eom)).numpy()
            losses = [min_loss, ]

        # tf.profiler.experimental.stop()
    except KeyboardInterrupt:
        # tf.profiler.experimental.stop()
        print("Interrupted. Progress is saved in chkpt/")
        try:
            data['int'] = True
            return (data, losses, kernel(pump, ps, eom).numpy())
        except:
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)

    data['pump'] = pump.numpy()
    data['ps'] = ps.numpy()
    data['eom'] = eom.numpy()
    data['loss'] = min_loss
    return (data, losses, kernel(best_params[0], best_params[1], best_params[2]).numpy())

# def jsi_conv(param, filename, epochs=5):
#     nodes = param.get('nodes')
#     padding = param.get('padding', 0)
#     n_rings = param.get('n_rings', 1)
#     orth_itr = param.get('orth_itr', 3)
#     nodes_t = nodes + 2 * padding
#     length = param.get('length', nodes_t - 1)
#     num_param = n_rings * length * 2 + nodes * 2 * n_rings * 2 + 1

#     kernel = JsiKernel(nodes, padding, n_rings, length, orth_itr)

#     def create_cnn_model():
#         model = models.Sequential([
#             layers.Conv2D(32, (11, 11), padding='same', activation='relu', input_shape=(28, 28, 1)),
#             layers.BatchNormalization(),
#             layers.Conv2D(32, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)),
#             layers.BatchNormalization(),
#             layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
#             layers.BatchNormalization(),
#             layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
#             layers.BatchNormalization(),
#             layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)),
#             layers.Flatten(),
#             layers.Dense(4096, activation='softmax'),
#             layers.Dense(4096, activation='relu'),
#             layers.Dense(2048, activation='relu'),
#             layers.Dense(1024, activation='relu'),
#             layers.Dense(num_param, activation='relu'),
#             layers.Dense(num_param)
#         ])
#         return model
        
#     def _parse_function(proto):
#         # Define features
#         features = {
#             'data': tf.io.FixedLenFeature([nodes * nodes], tf.float32),
#             'label': tf.io.VarLenFeature(tf.float32)
#         }
#         # Load one example
#         parsed = tf.io.parse_single_example(proto, features)
        
#         # Extract the image as a 28x28 array
#         parsed['data'] = tf.reshape(parsed['data'], (nodes, nodes, 1))
        
#         return parsed['data'], parsed['data']

#     dataset = None
#     if filename == 'mnist':
#         (train_images, _), (_, _) = mnist.load_data()
#         train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
#         dataset = tf.data.Dataset.from_tensor_slices((train_images, train_images))
#     else:
#         # Create a dataset from the TFRecord file
#         dataset = tf.data.TFRecordDataset(filenames=filename)
    
#         dataset = dataset.map(_parse_function)

#     # Shuffle, batch, and prefetch the dataset
#     dataset = dataset.shuffle(1024).batch(16).prefetch(tf.data.experimental.AUTOTUNE)

#     optimizer = tf.keras.optimizers.Adam(1e-4)
    
#     model = param.get('model')
#     if not 'model' in param or model is None:
#         model = create_cnn_model()
#         model.compile(optimizer=optimizer,
#                   loss=JsiError(kernel))

#     if isinstance(model, str):
#         model = models.load_model(model)
        
#     model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#         filepath='./nn_chkpt',
#         monitor='loss',
#         mode='min',
#         save_best_only=True
#     )

#     model.fit(dataset, epochs=epochs, callbacks=[model_checkpoint_callback])
#     return model