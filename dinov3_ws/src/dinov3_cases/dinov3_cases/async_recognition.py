"""Bounded background classification; geometry never waits for the encoder."""
import threading
import time

from .grounded import ObjectGallery, TrackedObjectRecognizer
from .object_visualization import ObjectVisualizationSnapshot


class AsyncRecognition:
    def __init__(self, encoder, root, max_side, stride, *, wake, on_error):
        self.encoder, self.root = encoder, root
        self.max_side, self.stride = max_side, stride
        self.wake, self.on_error = wake, on_error
        self.condition = threading.Condition()
        self.closed = False
        self.pending = self.running_key = self.result = None
        self.identity = None
        self.epoch = 0
        self.version = self.consumed = 0
        self.next_scan = 0.0
        self.retry_after = 0.0
        self.labels = ()
        self.thread = threading.Thread(target=self._run, name="dino-recognition", daemon=True)
        self.thread.start()

    def has_result(self):
        with self.condition:
            return self.version != self.consumed or time.monotonic() >= self.next_scan

    def update(self, pixels, proposals, *, token, threshold, margin, frame_id):
        identity = (token, len(proposals), threshold, margin)
        with self.condition:
            identity_changed = identity != self.identity
            if identity_changed:
                self.identity = identity
                self.epoch += 1
                self.retry_after = 0.0
                self.result = None
                self.pending = None
            key = self.epoch
            result = self.result
            changed = identity_changed or self.version != self.consumed
            self.consumed = self.version
            if (not self.closed and time.monotonic() >= self.retry_after
                    and self.running_key != key
                    and (self.pending is None or self.pending[0] != key)
                    and (result is None or time.monotonic() >= self.next_scan)):
                # Owned copy avoids exhausting the shared RGB frame pool during DINO.
                self.pending = (key, pixels.copy(), [dict(p, bbox=list(p['bbox'])) for p in proposals],
                                token, threshold, margin, frame_id)
                self.condition.notify()
            if result is None:
                detections = [dict(p, label="unknown", score=None, reason="recognition_pending")
                              for p in proposals]
                return detections, None, changed, "pending" if proposals else "none", None
            classes, snapshot, source, recognized_frame = result
            return ([dict(p, **c) for p, c in zip(proposals, classes)], snapshot,
                    changed, source if changed else "cached", recognized_frame)

    def _run(self):
        gallery = ObjectGallery()
        recognizer = TrackedObjectRecognizer()
        revision = 0
        recognized_frame = None
        snapshot = None
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.closed or self.pending is not None)
                if self.closed:
                    return
                job, self.pending = self.pending, None
                self.running_key = job[0]
            key, pixels, proposals, token, threshold, margin, frame_id = job
            try:
                if gallery.refresh(self.root, self.encoder, self.max_side, self.stride):
                    revision += 1
                # Include epoch: a disappearing/reappearing subset cannot reuse old labels.
                _, diagnostics, changed = recognizer.update(
                    pixels, proposals, self.encoder, gallery, self.max_side, self.stride,
                    threshold, margin, token=(token, key), gallery_revision=revision)
                if recognizer.encoded:
                    recognized_frame = frame_id
                if not proposals:
                    recognized_frame = None
                if changed:
                    snapshot = ObjectVisualizationSnapshot(
                        recognizer.diagnostic_detections, diagnostics, gallery,
                        self.max_side, self.stride)
                source = ("none" if not proposals else "dino" if recognizer.encoded
                          else "reclassified" if changed else "cached")
                with self.condition:
                    if not self.closed and key == self.epoch:
                        self.result = (recognizer.classifications, snapshot, source, recognized_frame)
                        self.labels = tuple(gallery.labels)
                        self.version += 1
            except Exception as error:
                with self.condition:
                    if key == self.epoch:
                        self.retry_after = time.monotonic() + 1.0
                self.on_error(error)
            finally:
                with self.condition:
                    self.running_key = None
                    self.next_scan = time.monotonic() + 1.0
                self.wake()

    def close(self):
        with self.condition:
            self.closed = True
            self.pending = None
            self.condition.notify_all()
        self.thread.join(timeout=3)
