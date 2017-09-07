import matplotlib.pyplot as plt

from dltracking.data.BouncingBallsDataset import BouncingBallsDataset
from dltracking.data.MiniBatchGenerator import MiniBatchGenerator

if __name__ == '__main__':
    dataset = BouncingBallsDataset(
        number_of_videos = 1000,
        number_of_frames = 32,
        image_resolution = 30)

    generator = MiniBatchGenerator(
        dataset,
        batchsize = 10,
        minlen = 4,
        maxlen = 5)

    batch = generator.get_batch()

    fig = plt.figure()

    N, T = batch['inputs'].shape[:2]

    for i in range(N):
        for t in range(T):
            plt.subplot(N, T, i * T + t + 1)

            if batch['masks'][i, t]:
                b = batch['inputs'][i, t]
                plt.imshow(b.reshape(-1, b.shape[-1]), cmap='gray')

                x1, x2, y1, y2 = batch['targets'][i, t]
                plt.scatter([x1, x2, x1, x2], [y1, y1, y2, y2])

            plt.axis('off')

    #plt.savefig('test_bb_annotations.png')

    plt.show()
