# video_recorder.py
# Wrapper around OpenCV VideoCapture for recording interview sessions

import cv2
import threading
import logging
from .config import VideoConfig

logger = logging.getLogger(__name__)

class VideoRecorder:
    """
    VideoRecorder captures video from the default webcam and writes to a file in a background thread.

    Attributes:
        config (VideoConfig): Configuration including output file, fps, and resolution.
        cap (cv2.VideoCapture): OpenCV video capture object.
        writer (cv2.VideoWriter): OpenCV video writer object.
        running (bool): Flag to control the recording thread.
        thread (threading.Thread): Background thread for recording.
    """
    def __init__(self, config: VideoConfig = None):
        if config is None:
            config = VideoConfig()
        self.config = config
        self.output_file = config.output_file
        self.fps = config.fps
        self.resolution = config.resolution
        self.cap = None
        self.writer = None
        self.running = False
        self.thread = None

    def start_recording(self):
        """
        Initialize the webcam and start recording in a separate thread.

        Raises:
            RuntimeError: If the default camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            logger.error("Unable to open default camera (index 0)")
            raise RuntimeError("Failed to open camera for video recording")

        # Set the desired resolution
        width, height = self.resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(
            self.output_file,
            fourcc,
            self.fps,
            (width, height)
        )

        self.running = True
        self.thread = threading.Thread(target=self._record, daemon=True)
        self.thread.start()
        logger.info(f"Started video recording to '{self.output_file}' at {self.fps} FPS and resolution {width}x{height}")

    def _record(self):
        """
        Internal method: read frames from the webcam and write to file while running is True.
        """
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame from camera")
                continue
            self.writer.write(frame)

    def stop_recording(self):
        """
        Stop the recording thread, release camera and file resources, and close any OpenCV windows.
        """
        if not self.running:
            return

        self.running = False
        if self.thread is not None:
            self.thread.join()

        if self.cap is not None:
            self.cap.release()
        if self.writer is not None:
            self.writer.release()

        cv2.destroyAllWindows()
        logger.info(f"Stopped video recording, file saved to '{self.output_file}'")
