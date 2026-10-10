"""Stream BGR frames to another machine as H.264 over RTP/UDP (GStreamer, via OpenCV).

Receiver (e.g. on a Mac with `brew install gstreamer`):
    gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
        ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
"""

from __future__ import annotations

import cv2
import numpy as np


class Streamer:
    def __init__(self, host, port=5000, width=960, height=720, fps=15, kbps=2500):
        self.size = (width, height)
        # software x264 (this Orin has no hardware encoder): fastest preset, no B-frames, a key
        # frame every second so a receiver that joins late shows a picture quickly
        pipeline = (
            f"appsrc ! videoconvert ! video/x-raw,format=I420 ! "
            f"x264enc tune=zerolatency speed-preset=ultrafast bitrate={kbps} key-int-max={fps} ! "
            f"rtph264pay config-interval=1 pt=96 ! udpsink host={host} port={port} sync=false async=false"
        )
        self.writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, self.size, True)
        if not self.writer.isOpened():
            raise RuntimeError("could not open the GStreamer pipeline (OpenCV built without GStreamer?)")
        print(f"streaming {width}x{height} @ {fps} fps to {host}:{port}")

    def send(self, frame):
        if frame.shape[1::-1] != self.size:
            frame = cv2.resize(frame, self.size, interpolation=cv2.INTER_AREA)
        self.writer.write(np.ascontiguousarray(frame))

    def close(self):
        self.writer.release()


def label(img, text):
    cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def grid(tiles, size=(480, 360)):
    """2x2 mosaic of (image, title) pairs."""
    out = [label(cv2.resize(t, size, interpolation=cv2.INTER_AREA), name) for t, name in tiles]
    return np.vstack([np.hstack(out[:2]), np.hstack(out[2:4])])
