"""CPU fixture for shared-memory frame identity and worker failure tests."""

import argparse
import os
import time

import dinov3_cases.grounding_ipc as ipc


class Detector:
    def __init__(self, **config):
        pass

    def validate_prompt(self, prompt):
        if prompt == "invalid":
            raise ValueError("invalid test prompt")

    def detect(self, image, prompt, box_threshold, nms_iou, limit,
               *, min_width=0, min_height=0, min_area=0):
        if prompt == "hang":
            time.sleep(5)
        width, height = image.size
        if width < min_width or height < min_height or width * height < min_area:
            return []
        return [dict(bbox=[0, 0, *image.size], color=list(image.getpixel((0, 0))))]


if __name__ == "__main__":
    if os.environ.get("IPC_FAKE_MISMATCH"):
        send = ipc._send

        def wrong_frame(sock, value):
            if "frame_id" in value:
                value["frame_id"] += 1
            send(sock, value)

        ipc._send = wrong_frame
    parser = argparse.ArgumentParser()
    parser.add_argument("--fd", type=int)
    ipc.worker_main(parser.parse_args().fd, Detector, dict(tensorrt="fixture", pid=os.getpid()))
