import numpy as np
from theano import shared
import theano.tensor as tt
from theano.ifelse import ifelse

from .normalizing_flows import NormalizingFlow


def _init_made_params(ns, rng):
    """Initialize model parameters of a MADE.
    
    Parameters
    ----------
    l : int
        The numbers of units in all layers, 
        including the input and output layers.
    rng : numpy.random.RandomState
        Random number generator.
    
    Returns
    -------
    (ws, bs) : (List[shared], List[shared])
        Model parameters (weights and biases).
    """
    ws = list()
    bs = list()
    for l in range(len(ns) - 1):
        d_in = ns[l]
        d_out = ns[l + 1]
        ws.append(shared(0.01 * rng.randn(d_in, d_out).astype('float32')))
        bs.append(shared(0.01 * rng.randn(d_out).astype('float32')))

    return ws, bs


def _create_made_mask(d_pre, d, m_pre, output_layer, rev_order=False):
    """Create a mask for MADE.
    
    Parameters
    ----------
    d_pre : int
        The number of rows in the weight matrix. 
    d : int
        The number of columns in the weight matrix. 
    m_pre : numpy.ndarray, shape=(d_pre,)
        The number of inputs to the units in the previous layer.
    output_layer : bool
        True for the output layer. 
    rev_order : bool
        If true, the order of connectivity constraints is reversed. 
        It is used only for the output layer. 
    
    Returns
    -------
    m, mask : (numpy.ndarray, numpy.ndarray)
        Mask indices and Mask.
    """
    d_input = np.max(m_pre)
    mask = np.zeros((d_pre, d)).astype('float32')
    
    if not output_layer:
        m = np.arange(1, d_input).astype(int)
        if len(m) < d:
            m = np.hstack((m, (d_input - 1) * np.ones(d - len(m))))
            
        for ix_col in range(d):
            ixs_row = np.where(m_pre <= m[ix_col])[0]
            (mask[:, ix_col])[ixs_row] = 1

    else:
        m = np.arange(1, d + 1)
        if rev_order:
            m = m[::-1]
        for ix_col in range(d):
            ixs_row = np.where(m_pre < m[ix_col])[0]
            mask[ixs_row, ix_col] = 1
            
    return m, mask


def _create_made_masks(d, ns, rev_order=False):
    """Create masks for all layers.
    
    Parameters
    ----------
    d : int
        Input dimension.
    ns : list of int
        The numbers of units in all hidden layers. 
    rev_order : bool
        If true, the order of connectivity constraints is reversed.
        
    Returns
    -------
    masks : List[numpy.ndarray]
        List of masks.
    """
    # The number of layers
    n_layers = len(ns)
    
    # Mask indices for the input layer
    m = np.arange(1, d + 1).astype(int)
    if rev_order:
        m = m[::-1]
    masks = list()
    
    d_pre = d
    for l, n in zip(range(n_layers), ns):
        m, mask = _create_made_mask(d_pre, n, m, output_layer=False)
        masks.append(mask)
        d_pre = n
    _, mask = _create_made_mask(d_pre, d, m, output_layer=True, 
                                rev_order=rev_order)

    masks.append(mask)
    
    return masks


class SimpleMadeIAF(NormalizingFlow):
    """An inverse autoregressive flow: (z - f1(z, h1, g1)) / f2(z, h2, g2)
    
    f[1/2] are MADEs, whose connectivity constraints are deterministically set. 
    h[1/2] are the activation functions of the hidden layers in f[1/2]. 
    g[1/2] are the activation functions of the output layers in f[1/2]. 
    The number of units in all layers are same with the dimension of the 
    random variable in ADVI. 
    
    Parameters
    ----------
    sh : tuple
        Shape of the random variable. It does not include the dimension of
        mini-batches.
    ns : list of int
        The numbers of units in all layers, shared for f1 and f2.
    hs : (tensor -> tensor, tensor -> tensor)
        Elementwise activation function of the hidden layers in f[1/2].
        Default is (tt.nnet.sigmoid, tt.nnet.sigmoid).
    gs : (tensor -> tensor, tensor -> tensor)
        Elementwise activation function of the output layer in f[1/2].
        Default is (tt.nnet.softplus, tt.nnet.softplus)
    scale : float
        Scale of the log determinant of Jacobian.
        It is used for mini-batch. Default is 1.
    rev_order : bool
        If true, the order of connectivity constraints is reversed.
        Default is False.
    random_seed : int or None
        Random seed for initialization of weight matrices.

    Example
    -------
    # For global RVs
    sh = rv.tag.test_value.shape
    ns = (10, 10) # (np.prod(sh))-(10)-(10)-(np.prod(sh))
    SimpleMadeIAF(sh, ns)

    # For local RVs
    sh = rv.tag.test_value.shape[1:]
    ns = (10, 10)
    scale = total_size / minibatch_size
    SimpleMadeIAF(sh, ns, scale=scale)
    """
    def __init__(
        self, sh, ns, hs=(tt.nnet.sigmoid, tt.nnet.sigmoid),
        gs=(tt.nnet.softplus, tt.nnet.softplus), scale=1, rev_order=False,
        random_seed=None
        ):
        self.sh = sh
        self.ns = ns
        self.hs = hs
        self.gs = gs
        self.scale = scale
        self.rev_order = rev_order
        self.rng = np.random.RandomState(random_seed)
        self.model_params = None
    
    def init(self, rvs):
        d = np.prod(self.sh)
        self.d = d
        w1s, b1s = _init_made_params([d] + self.ns + [d], self.rng) # f1
        w2s, b2s = _init_made_params([d] + self.ns + [d], self.rng) # f2
        self.wss = (w1s, w2s)
        self.bss = (b1s, b2s)
        self.model_params = w1s + w2s + b1s + b2s

        # Masks
        self.mask1s = _create_made_masks(self.d, self.ns, self.rev_order)
        self.mask2s = _create_made_masks(self.d, self.ns, self.rev_order)
        
    def trans(self, zs):
        """Transform random variables and compute log determinant of Jacobian.
        
        Parameters
        ----------
        zs : tensor, shape=(n_mc_samples, dim_rv)
            Random variables. n_mc_samples denotes the number of Monte Carlo samples
            in ADVI. dim_rv is the dimension of (concatenated) random variables. 
            
        Returns
        -------
        (zs_new, ldjs) : (tensor, tensor)
            Transformed random variables and log determinant of Jacobians.
            zs_new.shape is the same with zs. ldjs.shape = (n_mc_samples,). 
        """
        # Mini-batch
        not_mb = tt.eq(zs.ndim, len(self.sh))

        # Flatten
        zs = ifelse(not_mb, zs.reshape((-1, tt.prod(self.sh))), zs.ravel())

        # Outputs of MADEs
        n_layers = len(self.ns)
        h1, h2 = self.hs
        g1, g2 = self.gs
        w1s, w2s = self.wss
        b1s, b2s = self.bss
        
        def f(zs, n_layers, h, g, ws, bs, masks):
            """MADE"""
            xs = zs
            for l in range(n_layers):
                xs = h(xs.dot(masks[l] * ws[l]) + bs[l])
            return g(xs.dot(masks[n_layers] * ws[n_layers]) + bs[n_layers])
        
        f1s = f(zs, n_layers, h1, g1, w1s, b1s, self.mask1s)
        f2s = f(zs, n_layers, h2, g2, w2s, b2s, self.mask2s)
        
        # Outputs
        zs_new = (zs - f1s) / f2s
        zs_new = ifelse(not_mb, zs_new.reshape(self.sh),
                        zs_new.reshape([-1] + self.sh))
        
        # Log determinant of Jacobian
        ldjs = ifelse(not_mb, -tt.sum(tt.log(f2s), axis=-1),
                      -tt.sum(tt.log(f2s), axis=-1) * zs_new.shape[0])
        ldjs *= self.scale

        return zs_new, ldjs
        
    def get_params(self):
        if self.model_params is None:
            raise RuntimeError("get_params() was invoked before trans().")
        else:
            return self.model_params

