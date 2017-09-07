import matplotlib.pyplot as plt
import matplotlib.animation as animation

from dltracking.data.BouncingBallsDataset import BouncingBallsDataset

if __name__ == '__main__':
    dataset = BouncingBallsDataset(
        number_of_videos = 4,
        number_of_frames = 200,
        image_resolution = 50)

    fig = plt.figure()

    ims = []

    data1 = dataset.sample(0)
    data2 = dataset.sample(1)
    data3 = dataset.sample(2)
    data4 = dataset.sample(3)

    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    def drawBoundingBox(ball, target):
        ball[target[2]:target[3], target[0]] = 1
        ball[target[2]:target[3], target[1]] = 1
        ball[target[2], target[0]:target[1]] = 1
        ball[target[3], target[0]:target[1]+1] = 1

        return ball

    for t in range(len(data1['sample'])):
        ball1 = drawBoundingBox(data1['sample'][t], data1['targets'][t])
        ball2 = drawBoundingBox(data2['sample'][t], data2['targets'][t])
        ball3 = drawBoundingBox(data3['sample'][t], data3['targets'][t])
        ball4 = drawBoundingBox(data4['sample'][t], data4['targets'][t])

        im1 = ax1.imshow(ball1, cmap='gray', animated=True)
        im2 = ax2.imshow(ball2, cmap='gray', animated=True)
        im3 = ax3.imshow(ball3, cmap='gray', animated=True)
        im4 = ax4.imshow(ball4, cmap='gray', animated=True)

        ims.append([im1, im2, im3, im4])

    # It is also possible to use FuncAnimation with imshow and to draw the bounding boxes with patches.Rectangle, but
    # according to my experiments this results in a framerate-dependent motion, therefore I draw the bounding boxes
    # directly into the image and use ArtistAnimation.
    anim = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat=False)

    #anim.save('test_bb_animation.mp4')

    plt.show()
