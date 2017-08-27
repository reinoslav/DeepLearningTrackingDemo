import numpy as np

from .attention_rnn import RATM

if __name__ == '__main__':
    imsize = (120, 160)
    patchsize = (28, 28)
    nhid = 32
    numpy_rng = np.random.RandomState(1)

    model = RATM(name='RATM', imsize=imsize,
                 patchsize=patchsize, nhid=nhid,
                 numpy_rng=numpy_rng, eps=1e-4,
                 hids_scale=1.,
                 nchannels=1,
                 weight_decay=np.float32(.2))

    print("TEST");
