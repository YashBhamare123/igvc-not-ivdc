#!/usr/bin/env python3
"""Live RealSense D455 viewer: color camera and colorized depth side by side.

Hover anywhere to read the distance under the cursor (shown on both panels).
Left-click pins a measurement, right-click clears pins.

Keys:  q / Esc  quit        s  save snapshot (color, depth image, raw depth .npy)
       f        toggle depth filters        [ / ]  shrink / grow the max colour range
"""

import os
import time

import cv2
import numpy as np
import pyrealsense2 as rs

W, H, FPS = 640, 480, 30
HEADER, FOOTER = 44, 28
WINDOW = "RealSense D455"
FONT = cv2.FONT_HERSHEY_SIMPLEX
NEAR_M = 0.3  # D455 min range is ~0.4 m; anything closer reads as hottest colour


def put_text(img, text, org, scale=0.55, color=(255, 255, 255), thick=1):
    """Text with a dark outline so it stays legible on any background."""
    cv2.putText(img, text, org, FONT, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def colorize(depth_m, far_m):
    """Near = red, far = blue, no-data = black."""
    norm = np.clip((depth_m - NEAR_M) / (far_m - NEAR_M), 0, 1)
    img = cv2.applyColorMap((255 * (1 - norm)).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[depth_m <= 0] = 0
    return img


def draw_colorbar(img, far_m):
    x0, y0, w, h = img.shape[1] - 30, 40, 14, img.shape[0] - 80
    ramp = np.linspace(255, 0, h).astype(np.uint8)[:, None].repeat(w, axis=1)
    img[y0 : y0 + h, x0 : x0 + w] = cv2.applyColorMap(ramp, cv2.COLORMAP_TURBO)
    cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (255, 255, 255), 1)
    for frac in (0, 0.25, 0.5, 0.75, 1):
        y = int(y0 + frac * h)
        put_text(img, f"{NEAR_M + frac * (far_m - NEAR_M):.1f}", (x0 - 40, y + 5), 0.4)
    put_text(img, "m", (x0, y0 - 10), 0.45)


def distance_at(depth_m, x, y, r=3):
    """Median of valid pixels in a small patch — far less noisy than one pixel."""
    patch = depth_m[max(0, y - r) : y + r + 1, max(0, x - r) : x + r + 1]
    valid = patch[patch > 0]
    return float(np.median(valid)) if valid.size else None


def draw_marker(img, x, y, dist, color):
    cv2.drawMarker(img, (x, y), color, cv2.MARKER_CROSS, 18, 2, cv2.LINE_AA)
    label = f"{dist:.2f} m" if dist is not None else "no data"
    tx = x + 10 if x < img.shape[1] - 90 else x - 90
    put_text(img, label, (tx, y - 10), 0.6, color, 2)


class Mouse:
    def __init__(self):
        self.pos = None  # (x, y) in panel coordinates
        self.pins = []

    def __call__(self, event, x, y, flags, _):
        y -= HEADER
        if not (0 <= y < H):
            self.pos = None
            return
        x %= W  # same pixel whether hovering the colour or depth panel
        self.pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.pins.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.pins.clear()


def main():
    pipeline, config = rs.pipeline(), rs.config()
    config.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
    config.enable_stream(rs.stream.color, W, H, rs.format.bgr8, FPS)
    profile = pipeline.start(config)
    device = profile.get_device()
    scale = device.first_depth_sensor().get_depth_scale()
    name = device.get_info(rs.camera_info.name)

    align = rs.align(rs.stream.color)  # depth pixels line up with colour pixels
    filters = [rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter()]
    use_filters, far_m = True, 6.0

    mouse = Mouse()
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(WINDOW, 2 * W, H + HEADER + FOOTER)
    cv2.setMouseCallback(WINDOW, mouse)

    fps, last = 0.0, time.time()
    try:
        while True:
            frames = align.process(pipeline.wait_for_frames(5000))
            depth_frame, color_frame = frames.get_depth_frame(), frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            if use_filters:
                for f in filters:
                    depth_frame = f.process(depth_frame)

            depth_m = np.asanyarray(depth_frame.get_data()).astype(np.float32) * scale
            color = np.asanyarray(color_frame.get_data()).copy()
            depth_img = colorize(depth_m, far_m)
            draw_colorbar(depth_img, far_m)

            # Centre reticle, then pins, then the live cursor on top
            cx, cy = W // 2, H // 2
            marks = [((cx, cy), (255, 255, 255))]
            marks += [(p, (0, 255, 255)) for p in mouse.pins]
            if mouse.pos:
                marks.append((mouse.pos, (0, 255, 0)))
            for (x, y), col in marks:
                d = distance_at(depth_m, x, y)
                draw_marker(color, x, y, d, col)
                draw_marker(depth_img, x, y, d, col)

            put_text(color, "COLOR", (10, 24), 0.6)
            put_text(depth_img, "DEPTH" + ("  (filtered)" if use_filters else ""), (10, 24), 0.6)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - last, 1e-6)
            last = now

            valid = depth_m[depth_m > 0]
            header = np.full((HEADER, 2 * W, 3), 32, np.uint8)
            stats = (
                f"{name}   |   {fps:4.1f} FPS   |   valid {100 * valid.size / depth_m.size:4.1f}%"
                + (
                    f"   |   nearest {valid.min():.2f} m   median {np.median(valid):.2f} m"
                    if valid.size
                    else ""
                )
            )
            put_text(header, stats, (12, 28), 0.6)

            footer = np.full((FOOTER, 2 * W, 3), 32, np.uint8)
            put_text(
                footer,
                "hover: measure   L-click: pin   R-click: clear pins   "
                "[ ]: range   f: filters   s: snapshot   q: quit",
                (12, 19),
                0.45,
                (190, 190, 190),
            )

            cv2.imshow(WINDOW, np.vstack([header, np.hstack([color, depth_img]), footer]))

            key = cv2.waitKey(1) & 0xFF
            # WND_PROP_VISIBLE is always -1 on this GTK build; AUTOSIZE turns -1 once closed
            if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_AUTOSIZE) < 0:
                break
            elif key == ord("f"):
                use_filters = not use_filters
            elif key == ord("]"):
                far_m = min(far_m + 0.5, 20.0)
            elif key == ord("["):
                far_m = max(far_m - 0.5, NEAR_M + 0.5)
            elif key == ord("s"):
                os.makedirs("snapshots", exist_ok=True)
                stamp = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"snapshots/{stamp}_color.png", np.asanyarray(color_frame.get_data()))
                cv2.imwrite(f"snapshots/{stamp}_depth.png", colorize(depth_m, far_m))
                np.save(f"snapshots/{stamp}_depth_m.npy", depth_m)
                print(f"saved snapshots/{stamp}_*")
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
