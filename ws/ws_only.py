import sys, os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import tensorflow as tf
from tensorflow.keras import layers
from tqdm.autonotebook import trange

@tf.keras.saving.register_keras_serializable()
class JsiKernel:
    def __init__(self, nodes, npump=40, width=40, bw=(lambda x: 1), strides=1, **kwargs):
        self.nodes = nodes
        self.npump = npump
        self.width = width
        self.strides = strides
        self.nodes_t = nodes * strides
        self.pi = tf.constant(np.pi, dtype=tf.complex64)
        self.zero = tf.constant(0, dtype=tf.complex64)
        self.one = tf.constant(1, dtype=tf.complex64)
        self.pmask = tf.constant(self.mask())
        # self.bw = tf.map_fn(lambda x: tf.map_fn(bw, x), tf.constant(np.rot90(self.mask()), dtype=tf.complex64))
        if callable(bw): 
            self.bw = tf.cast(bw(tf.constant(np.rot90(self.mask()), dtype=tf.float32)), dtype=tf.complex64)
        else: self.bw = bw
                                     
    def get_config(self):
        return {
            'nodes': self.nodes, 
            'npump': self.npump, 
            'width': self.width,
            'bw': self.bw,
            'strides': self.strides
            }
    
    def mask(self):
        output = np.zeros((self.nodes_t, self.nodes_t, 1), dtype=np.int32)
        for index, _ in np.ndenumerate(output):
            output[index] = index[1] - index[0] if np.abs(index[0] + index[1] + 1 - self.nodes_t) <= np.round(self.width / 2) else self.npump # large enough number s.t. it is excluded from substitution
        return output

    @tf.function
    def __call__(self, pump, ps):
        x = tf.zeros((self.nodes_t, self.nodes_t, 1), dtype=tf.complex64)
        for i in tf.range(self.npump):
            index = i - tf.cast(tf.math.floordiv(self.npump, 2), dtype=tf.int32)
            x += tf.where(tf.equal(self.pmask, index), pump[index], 0)
            
        x *= self.bw
        # pulse shaper
        # signal
        x *= tf.reshape(ps[0], (self.nodes_t, 1, 1))
        
        # idler
        x *= tf.reshape(ps[1], (1, self.nodes_t, 1))

        x = layers.AveragePooling2D((self.strides, self.strides), (self.strides, self.strides))(tf.reshape(tf.math.real(tf.math.conj(x) * x), (1, self.nodes_t, self.nodes_t, 1)))
        return tf.cast(tf.squeeze(x), dtype=tf.complex64)

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


def jsi_backprop(nodes, npump, width, bw=(lambda x: 1), strides=1, target = None, EPOCHS=None, lr=None, train=True, **kwargs):
    if train and not EPOCHS:
        raise ValueError("training parameter \'EPOCHS\' not provided")
    
    data = {}
    nodes_t = nodes * strides

    losses = np.zeros(int(EPOCHS if EPOCHS else 1), dtype=np.float32)
    kernel = JsiKernel(nodes, npump, width, bw, strides, **kwargs)

    @tf.function
    def gainless(x):
        mag = tf.math.abs(x)
        phase = tf.cast(tf.math.angle(x), dtype=tf.complex64)
        return tf.cast(tf.clip_by_value(mag, 0.0, 1.0), dtype=tf.complex64) * tf.math.exp(1j * phase)

    pump = tf.Variable(kwargs.get('pump', np.ones((npump, ), dtype=np.complex64)), name="pump", dtype=tf.complex64)
    ps = tf.Variable(kwargs.get('ps', np.ones((2, nodes_t), dtype=np.complex64)), name="pulse_shaper", dtype=tf.complex64, constraint=gainless)
    target_t = tf.zeros(shape=(nodes, nodes), dtype=tf.complex64)
    min_loss = 1.0
    best_params = [tf.Variable(p) for p in [pump, ps]]
    
    if not target is None:
        target_t = tf.constant(target, shape=(nodes, nodes), dtype=tf.complex64)
    output = None
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr if lr else 
                                         tf.keras.optimizers.schedules.ExponentialDecay(
                                             1e-3, decay_steps=100, decay_rate=0.9, staircase=True))
    
    @tf.function(input_signature=(
        tf.TensorSpec(shape=tf.shape(target_t), dtype=tf.complex64),
    ))
    def loss_func(pred):
        # vector projection
        pred_norm = tf.linalg.l2_normalize(pred)
        return tf.math.real(
                1 - tf.abs( tf.tensordot(tf.math.conj(target_t), pred_norm, axes=2) * tf.tensordot(tf.math.conj(pred_norm), target_t, axes=2) )
                           / (tf.abs((tf.tensordot(tf.math.conj(pred_norm),  pred_norm, axes=2)) * tf.tensordot(tf.math.conj(target_t), target_t, axes=2)))
                )

    @tf.function
    def train_step(loss_func, model, pump, ps):
        with tf.GradientTape() as tape:
            pred = model(pump, ps)
            loss = loss_func(pred)
            if tf.math.reduce_any(tf.math.is_nan(tf.math.real(pred))):
                return tf.constant(-1, dtype=tf.float32)
            else:
                gradients = tape.gradient(loss, [pump, ps])  # Compute gradients
                optimizer.apply_gradients(zip(gradients[0:2], [pump, ps]))  # Update weights
                return loss
        
    try:
        if train:
            # Create a checkpoint object
            ckpt = tf.train.Checkpoint(optimizer=optimizer, params=[pump, ps])
            for j in trange(int(EPOCHS), desc="iterations"):
                loss = train_step(loss_func, kernel, pump, ps)
                if loss.numpy() == -1: 
                    warnings.warn("Interrupted due to encountering NaN in the resulting JSI\n")
                    break
                else: 
                    losses[j] = loss.numpy()
                    if loss < min_loss:
                        min_loss = loss
                        best_params = [old.assign(new) for old, new in zip(best_params, [pump, ps])]
                if j % 100 == 0:
                    ckpt.save('./chkpt/chkpt')
        else:
            min_loss = loss_func(kernel(pump, ps)).numpy()
            losses = [min_loss, ]

    except KeyboardInterrupt:
        print("Interrupted. Progress is saved in chkpt/")
        try:
            data['int'] = True
            return (data, losses, kernel(pump, ps).numpy())
        except:
            try:
                sys.exit(130)
            except SystemExit:
                os._exit(130)

    data.update({
        'pump': pump.numpy(), 
        'ps': ps.numpy(), 
        'loss': min_loss, 
        })
    return (data, losses, kernel(*best_params).numpy())
