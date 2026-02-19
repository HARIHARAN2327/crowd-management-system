import cv2
import time

class RaspberryPiSimulator:
    def __init__(self, video_path):
        self.video = cv2.VideoCapture(video_path)

    def get_frame(self):
        success, frame = self.video.read()

        if not success:
            # Loop video (simulate continuous CCTV)
            self.video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            success, frame = self.video.read()

        # simulate real camera FPS
        time.sleep(0.03)

        return frame
