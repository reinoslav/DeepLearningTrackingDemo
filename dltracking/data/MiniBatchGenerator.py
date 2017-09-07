import numpy as np

class MiniBatchGenerator:
    def __init__(self, dataset, batchsize=None, minlen=None, maxlen=None):
        self.dataset = dataset

        self.batchsize = batchsize

        self.numpy_rng = np.random.RandomState(1)

        self.minlen = minlen
        self.maxlen = maxlen

        self.video_indices = np.arange(0, self.dataset.size())

        self.start_indices = []
        self.end_indices = []

        for idx in self.video_indices:
            self.start_indices.append(0)
            self.end_indices.append(dataset.sample(idx)['sample'].shape[0])

    def __create_minibatch(self):
        ...

    def batchsize(self):
        return self.batchsize

    def nbatches(self):
        if self.batchsize is not None:
            return len(self.video_indices) / self.batchsize

    def shuffle(self):
        ...

    def batch(self, bid):
        if bid < 0:
            raise Exception('Your Batch ID is too small. Batch IDs start with 0 and are consecutive.')

        if bid >= self.nbatches():
            raise Exception('Your Batch ID is too big. Batch IDs start with 0 and are consecutive.')

    def get_batch(self, return_all=False):
        shuffled_indices = self.video_indices[self.numpy_rng.permutation(len(self.video_indices))]

        frames = []
        targets = []

        for idx in shuffled_indices:
            if self.batchsize is None or return_all:
                start = self.start_indices[idx]
                end = self.end_indices[idx]
            else:
                if self.end_indices[idx] - self.start_indices[idx] < self.minlen:
                    continue

                start = self.numpy_rng.randint(self.start_indices[idx], self.end_indices[idx] - self.minlen)
                end = start + self.numpy_rng.randint(self.minlen, min(self.maxlen, self.end_indices[idx] - start) + 1)

            frames.append(self.dataset.sample(idx)['sample'][range(start, end)])
            targets.append(self.dataset.sample(idx)['targets'][range(start, end)])

            if not return_all and (self.batchsize is not None and len(frames) == self.batchsize):
                break

        maxlen = 0

        for v in frames:
            if maxlen < len(v):
                maxlen = len(v)

        masks = np.zeros((len(frames), maxlen), dtype=np.float32)
        vids = np.zeros((len(frames), maxlen, self.dataset.image_resolution, self.dataset.image_resolution, 1), dtype=np.float32)
        bboxes = np.zeros((len(frames), maxlen, 4), dtype=np.float32)

        for i, v in enumerate(frames):
            vids[i, :len(v)] = v[..., None]
            masks[i, :len(v)] = 1.
            bboxes[i, :len(v)] = targets[i]

        return {'inputs': vids.transpose(0, 1, 4, 2, 3), 'masks': masks, 'targets': bboxes}
