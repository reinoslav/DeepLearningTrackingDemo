import numpy as np

class VideoDataset:
    def __init__(self, number_of_videos, number_of_frames, image_resolution):
        self.number_of_videos = number_of_videos
        self.number_of_frames = number_of_frames
        self.image_resolution = image_resolution

        self.videos = np.zeros((number_of_videos, number_of_frames, image_resolution, image_resolution), dtype=np.float32)
        self.targets = np.zeros((number_of_videos, number_of_frames, 4), dtype=np.int8)

    def size(self):
        return self.videos.shape[0]

    def sample(self, sid):
        if sid < 0:
            raise Exception('Your Sample ID is too small. Sample IDs start with 0 and are consecutive.')

        if sid >= self.videos.size:
            raise Exception('Your Sample ID is too big. Sample IDs start with 0 and are consecutive.')

        return {'sample': self.videos[sid], 'targets': self.targets[sid]}
