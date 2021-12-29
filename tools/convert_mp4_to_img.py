import cv2
import os
import glob


def vonvert_video_to_img(video_path, img_path):
    if os.path.isdir(video_path):
        files = os.listdir(video_path)
        videos = [os.path.join(video_path, file) for file in files]
    elif os.path.isfile(video_path):
        videos = [video_path]

    fps = 6
    i = 1
    n = 0
    for video in videos:
        vc = cv2.VideoCapture(video)
        while True:
            flag, frame = vc.read()
            if not flag:
                print(video, "is ended.")
                break
            if n % fps == 0:
                i += 1
                cv2.imwrite(
                    os.path.join(img_path, "{:06d}.jpg".format(i)), frame
                )  # 存储为图像
            n += 1
        vc.release()


if __name__ == "__main__":
    video_path = "/home/cairujia1/data/xinjinbo/videos"
    img_path = "/home/cairujia1/data/xinjinbo/videos_images"
    vonvert_video_to_img(video_path, img_path)
