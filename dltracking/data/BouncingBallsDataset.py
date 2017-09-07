from .VideoDataset import VideoDataset

import numpy as np

class BouncingBallsDataset(VideoDataset):
    def __init__(self, number_of_videos, number_of_frames, image_resolution, number_of_balls=1):
        super().__init__(number_of_videos, number_of_frames, image_resolution)

        self.number_of_balls = number_of_balls

        self.__create_videos()

    def __create_videos(self):
        for i in range(self.number_of_videos):
            V = self.__bounce_vec(self.image_resolution, self.number_of_frames, self.number_of_balls, 10)

            self.videos[i] = V.reshape(self.number_of_frames, self.image_resolution, self.image_resolution)

            for j in range(len(self.videos[i])):
                (bb_left, bb_right) = self.__calc_target(self.videos[i][j])
                (bb_upper, bb_lower) = self.__calc_target(self.videos[i][j].transpose(1, 0))

                self.targets[i][j][0] = bb_left
                self.targets[i][j][1] = bb_right
                self.targets[i][j][2] = bb_upper
                self.targets[i][j][3] = bb_lower

    def __calc_target(self, data_dimension):
        non_zero_ind_first = np.zeros(len(data_dimension), dtype=np.int8)
        non_zero_ind_last = np.zeros(len(data_dimension), dtype=np.int8)

        for i in range(len(data_dimension)):
            non_zero_ind = np.asarray(np.where(data_dimension[i] > 0.0)).flatten()

            if len(non_zero_ind) > 0:
                non_zero_ind_first[i] = non_zero_ind[0]
                non_zero_ind_last[i] = non_zero_ind[-1]

        bb_min = non_zero_ind_first[non_zero_ind_first != 0].min() if len(non_zero_ind_first[non_zero_ind_first != 0]) > 0 else 0
        bb_max = non_zero_ind_last[non_zero_ind_last != 0].max() if len(non_zero_ind_last[non_zero_ind_last != 0]) > 0 else 0

        return (bb_min, bb_max)

    def __norm(self, x):
        return np.sqrt((x ** 2).sum())

    def __new_speeds(self, m1, m2, v1, v2):
        new_v2 = (2 * m1 * v1 + v2 * (m2 - m1)) / (m1 + m2)
        new_v1 = new_v2 + (v2 - v1)

        return new_v1, new_v2

    def __bounce_n(self, T, n, b, r=None, m=None):
        if r is None: r=np.array([1.8]*n)
        if m is None: m=np.array([1]*n)

        X = np.zeros((T, n, 2), np.float32)

        v = np.random.randn(n, 2)
        v = v / self.__norm(v) * .5

        good_config = False

        while not good_config:
            x = 2 + np.random.rand(n, 2) * 8

            good_config = True

            for i in range(n):
                for z in range(2):
                    if x[i][z] - r[i] < 0: good_config = False
                    if x[i][z] + r[i] > b: good_config = False

            for i in range(n):
                for j in range(i):
                    if self.__norm(x[i] - x[j]) < r[i] + r[j]:
                        good_config = False

        eps = .5

        for t in range(T):
            for i in range(n):
                X[t, i] = x[i]

            for mu in range(int(1 / eps)):
                for i in range(n):
                    x[i] += eps * v[i]

                for i in range(n):
                    for z in range(2):
                        if x[i][z] - r[i] < 0:  v[i][z] = abs(v[i][z])
                        if x[i][z] + r[i] > b: v[i][z] = -abs(v[i][z])

                for i in range(n):
                    for j in range(i):
                        if self.__norm(x[i] - x[j]) < r[i] + r[j]:
                            w = x[i] - x[j]
                            w = w / self.__norm(w)

                            v_i = np.dot(w.transpose(), v[i])
                            v_j = np.dot(w.transpose(), v[j])

                            new_v_i, new_v_j = self.__new_speeds(m[i], m[j], v_i, v_j)

                            v[i] += w * (new_v_i - v_i)
                            v[j] += w * (new_v_j - v_j)

        return X

    def __ar(self, x, y, z):
        return z / 2 + np.arange(x, y, z, np.float32)

    def __matricize(self, X, res, b, r=None):
        T, n = np.shape(X)[0:2]

        if r is None: r=np.array([1.2]*n)

        A = np.zeros((T, res, res), np.float32)

        [I, J] = np.meshgrid(self.__ar(0, 1, 1. / res) * b, self.__ar(0, 1, 1. / res) * b)

        for t in range(T):
            for i in range(n):
                A[t] += np.exp(-(((I - X[t, i, 0]) ** 2 + (J - X[t, i, 1]) ** 2) / (r[i] ** 2)) ** 4)

        A[t][A[t] > 1] = 1

        return A

    def __bounce_vec(self, res, T, n, b, r=None, m=None):
        if r is None: r = np.array([1.2] * n)

        x = self.__bounce_n(T, n, b, r, m)

        return self.__matricize(x, res, b, r).reshape(T, res ** 2)
