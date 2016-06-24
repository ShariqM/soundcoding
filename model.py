import tensorflow as tf
import numpy as np

K = 500 # Number of kernels
B = 50  # Batch Size
D = 256 # Input dimensionality
N = 100 # Neurons

def init_weights(shape, name):
    return tf.Variable(tf.random_normal(shape, stddev=0.05))

# Constants
X  = tf.placeholder("float") # input BxD
Nx = tf.placeholder("float") # input noise BxD
Nr = tf.placeholder("float") # rate noise BxN

U = tf.placeholder("float") # rate noise BxN
V = tf.placeholder("float") # rate noise BxN

Cx =  tf.placeholder("float") # CoVar of input, X
Cnr = tf.placeholder("float") # CoVar of Rate Noise
Cnx = tf.placeholder("float") # CoVar of Input Noise

U_i = np.random.randn(N, K) # Means of the Gaussian Kernels FIXME
U_i = np.broadcast_to(U_i, (B, N, K))
V_i = np.random.randn(N)    # Variances of the Gaussian Kernels FIXME
V_i = np.broadcast_to(V_i, (K, N)).T
V_i = np.broadcast_to(V_i, (B, N, K))

Cnr_i = 0.4 * np.identity(N)
Cnr_i = np.broadcast_to(Cnr_i, (B,N,N))
Cnx_i = 0.1 * np.identity(D)

# Variables
W = init_weights((D,N), "weights")
F = init_weights((N,K), "gweights")

Y = tf.matmul(X + Nx, W) # B x N
Y = tf.tile(tf.expand_dims(Y, 2), (1, 1, K)) # BxNxK # XXX Hmmm how does this effect W
E = tf.exp(-tf.square(Y - U)/(2 * V)) # BxNxK
G = tf.batch_matrix_diag(tf.reduce_sum(tf.log(F) * E, 2)) # BxNxN

WCnxW = tf.matmul(tf.matmul(W, Cnx, transpose_a=True), W) # NxN
WCnxW = tf.tile(tf.expand_dims(WCnxW, 0), (B, 1, 1)) # XXX Ugly

Crx  = tf.batch_matmul(tf.batch_matmul(G, WCnxW), G) + Cnr # BxNxN
iCrx = tf.batch_matrix_inverse(Crx)

GCrxG = tf.batch_matmul(tf.batch_matmul(G, iCrx), G) # BxNxN
# DxN
W_exp = tf.tile(tf.expand_dims(W, 0), (B, 1, 1)) # XXX Is this going to screw up gradients?
Cxr   = tf.batch_matmul(tf.batch_matmul(W_exp, GCrxG), W_exp, adj_y=True) # BxDxD

cost  = tf.log(tf.batch_matrix_determinant(Cxr))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(cost)
normalize = tf.assign(W, tf.nn.l2_normalize(W, 1))

# Launch the graph in a session
with tf.Session() as sess:
    # Initialize Variabels
    tf.initialize_all_variables().run()

    for i in range(100):
        X_i = np.random.randn(B,D)
        Nx_i = np.random.randn(B,D)
        Nr_i = np.random.randn(B,N)
        sess.run(train_op, feed_dict={X: X_i, Nx: Nx_i, Nr: Nr_i,
                                      U: U_i, V: V_i,
                                      Cnr: Cnr_i, Cnx: Cnx_i})
        sess.run(normalize)

# Notes
    # Normalize Data?
