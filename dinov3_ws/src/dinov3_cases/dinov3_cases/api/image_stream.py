"""Latest-only ROS Image to shared MJPEG; JPEG encoding stays off the ROS executor."""
import threading

import cv2


class ImageStream:
    def __init__(self, bridge, wake=lambda: None, on_error=lambda error: None):
        self.bridge, self.wake, self.on_error = bridge, wake, on_error
        self.condition = threading.Condition()
        self.readers = 0
        self.closed = False
        self.pending = None
        self.jpeg = None
        self.sequence = 0
        self.revision = 0
        self.worker = threading.Thread(target=self._encode, name="image-jpeg", daemon=True)
        self.worker.start()

    @property
    def has_readers(self):
        with self.condition:
            return self.readers > 0

    @property
    def needs_frame(self):
        with self.condition:
            return self.readers > 0 and self.jpeg is None

    def update(self, message):
        with self.condition:
            if self.readers and not self.closed:
                self.pending = (message, self.revision)
                self.condition.notify_all()

    def reset(self):
        with self.condition:
            self.revision += 1
            self.jpeg = self.pending = None
            self.condition.notify_all()

    def _encode(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                message, revision = self.pending
                self.pending = None
            try:
                pixels = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
                ok, encoded = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not ok:
                    raise ValueError("JPEG encoding failed")
                jpeg = encoded.tobytes()
                with self.condition:
                    if revision == self.revision and self.readers and not self.closed:
                        self.jpeg = jpeg
                        self.sequence += 1
                        self.condition.notify_all()
            except Exception as error:
                self.on_error(error)

    def multipart(self):
        with self.condition:
            if self.closed:
                return
            self.readers += 1
        self.wake()
        sequence = -1
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(
                        lambda: self.closed or (self.jpeg is not None and self.sequence != sequence),
                        timeout=1.0,
                    )
                    if self.closed:
                        return
                    jpeg, sequence = self.jpeg, self.sequence
                # Repeat the cached JPEG on idle sources to detect disconnected clients.
                # While waiting for the first image, MIME whitespace keeps the socket live.
                if jpeg is None:
                    yield b"\r\n"
                else:
                    yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
                           + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
        finally:
            with self.condition:
                self.readers -= 1
                if not self.readers:
                    self.jpeg = self.pending = None
                    self.revision += 1
            self.wake()

    def close(self):
        with self.condition:
            self.closed = True
            self.jpeg = self.pending = None
            self.condition.notify_all()
        self.worker.join(timeout=3)
