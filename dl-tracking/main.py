import numpy as np

from .attention_rnn import RATM

model = RATM(name='RATM', imsize=(120, 160),
             patchsize=(28, 28), nhid=32,
             numpy_rng=np.random.RandomState(1), eps=1e-4,
             hids_scale=1.,
             feature_network=feature_network,
             input_feature_layer_name=input_feature_layer_name,
             metric_feature_layer_name=metric_feature_layer_name,
             nchannels=1,
             weight_decay=np.float32(.2))

print("TEST");
