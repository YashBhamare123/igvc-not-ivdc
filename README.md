# igvc-not-ivdc

Real-time hazard detection from an Intel RealSense D455, grid pathfinding around those hazards, and a local navigator that drives toward a goal **10 m ahead** of the robot.

| Module | Role |
|--------|------|
| `depth_viewer.py` | Live color + depth preview (debug camera) |
| `hazard_detector.py` | Depth → ground-frame hazard circles (metres) |
| `pathfinder.py` | A* around hazard circles |
| `vehicle.py` | Twist commands + pure pursuit + e-stop |
| `navigator.py` | Full loop: stream → plan → command |

All distances are **metres**, speeds **m/s** / **rad/s**. Ground frame: origin under the camera, **x** right, **y** forward.

---

## Jetson Nano setup

Tested path: **Jetson Nano + JetPack 4.6** (Ubuntu 18.04, Python 3.6) or newer JetPack with Python 3.8+. Use a **USB 3** port for the D455.

### 1. System packages

```bash
sudo apt update
sudo apt install -y \
  python3-pip python3-venv python3-dev \
  python3-numpy python3-opencv \
  git cmake build-essential pkg-config \
  libusb-1.0-0-dev libssl-dev libgtk-3-dev
```

Jetson already ships a CUDA-capable OpenCV; prefer that over `pip install opencv-*` on ARM.

### 2. Intel RealSense (`librealsense` + `pyrealsense2`)

Pip wheels for `pyrealsense2` on Jetson ARM are unreliable. Build librealsense, then install the Python bindings.

```bash
# udev rules so the camera works without root
cd /tmp
git clone https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo ./scripts/setup_udev_rules.sh

# Build (Nano: no CUDA RS flag needed for basic depth)
mkdir -p build && cd build
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_EXAMPLES=false \
  -DBUILD_GRAPHICAL_EXAMPLES=false \
  -DBUILD_PYTHON_BINDINGS=true \
  -DPYTHON_EXECUTABLE=$(which python3)
make -j$(nproc)
sudo make install
sudo ldconfig
```

Confirm the device:

```bash
rs-enumerate-devices
# or
python3 -c "import pyrealsense2 as rs; print(rs.context().query_devices())"
```

If `import pyrealsense2` fails after install, add the bindings path (version may differ):

```bash
export PYTHONPATH="/usr/local/lib:$PYTHONPATH"
# permanent: add that line to ~/.bashrc
```

Plug the D455 into **USB 3**, then power-cycle the Nano if the device is not listed.

### 3. Clone this repo and Python deps

```bash
git clone <your-repo-url> igvc-not-ivdc
cd igvc-not-ivdc

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

`--system-site-packages` lets the venv use Jetson’s `cv2` / `numpy` from apt. Skip installing OpenCV via pip on the Nano.

### 4. Camera mount (important for units)

Set lens height and tilt to match the robot (or press `c` in the hazard viewer aimed at open floor):

| Parameter | Meaning | Typical |
|-----------|---------|---------|
| `--cam-height-m` | Lens height above floor | `0.30` |
| `--cam-tilt-deg` | Pitch; positive = looking down | `0`–`15` |
| `--robot-radius-m` | Half-width; must match hazard margin | `0.35` |

---

## How to run

Always activate the venv first:

```bash
cd igvc-not-ivdc
source .venv/bin/activate
```

### Camera check

```bash
python depth_viewer.py
```

Keys: hover to measure depth, `f` filters, `[` `]` range, `s` snapshot, `q` quit.

### Hazard circles only

```bash
python hazard_detector.py
python hazard_detector.py --cam-height-m 0.30 --cam-tilt-deg 10 --max-range-m 8
```

Keys: `c` calibrate floor, `[` `]` range, `p` print hazards, `q` quit.

### Full navigator (depth → path → vehicle)

**Dry run** (plan and show overlay; always command zero velocity — use this first on the Nano):

```bash
python navigator.py --dry-run \
  --cam-height-m 0.30 --cam-tilt-deg 10 \
  --robot-radius-m 0.35 --goal-ahead-m 10
```

**Live** (sends twists into `Vehicle.send()` — wire motors first, see below):

```bash
python navigator.py \
  --cam-height-m 0.30 --cam-tilt-deg 10 \
  --robot-radius-m 0.35 \
  --max-linear-m-s 0.40 \
  --emergency-stop-m 0.45
```

Useful flags:

| Flag | Default | Notes |
|------|---------|--------|
| `--goal-ahead-m` | `10` | Fixed goal this far ahead at start |
| `--robot-radius-m` | `0.35` | Inflates hazards; keep clear of chassis |
| `--max-range-m` | `8` | Hazard detection range (m) |
| `--emergency-stop-m` | `0.45` | Stop if object edge closer than this |
| `--dry-run` | off | Perception + planning only |
| `--headless` | off | No OpenCV window |

Headless example (SSH without display):

```bash
python navigator.py --dry-run --headless
```

For a GUI over SSH, use `ssh -X` or run on the Nano desktop.

---

## Live view over WiFi (`stream.py`)

`lane_follow.py` can stream what it sees and decides to another computer (e.g. a Mac) on the same network: H.264 over RTP/UDP, encoded on the Orin with GStreamer (software `x264enc`) through OpenCV. The view is a 960×720 mosaic at 15 fps:

| Panel | Shows |
|-------|-------|
| top left | camera image, lanes in yellow, objects in red, mode and L/R command |
| top right | depth colour map, 0–5 m |
| bottom left | masks: lanes white, objects red |
| bottom right | top-down: lane/object points, corridor fan, chosen corridor (green) and heading arrow |

### 1. Receiver (Mac), once

```bash
brew install gstreamer gst-plugins-base gst-plugins-good gst-plugins-bad gst-libav
ipconfig getifaddr en0      # this Mac's IP, to pass to the Orin
```

Both machines must be on the same WiFi. Allow incoming connections for `gst-launch-1.0` if macOS asks.

### 2. Start the viewer (Mac)

```bash
gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96" \
  ! rtph264depay ! avdec_h264 ! videoconvert ! autovideosink sync=false
```

It can be started before or after the Orin side; a key frame is sent every second, so the picture appears within a second of joining.

### 3. Start streaming (Orin)

```bash
sudo modprobe uvcvideo                       # after every reboot (the RealSense blacklist disables it)

# motors off: perception and decisions only
python3 lane_follow.py --dry-run --max-time 600 --stream <MAC_IP>

# driving
python3 lane_follow.py --stream <MAC_IP> --save runs/lf1
```

| Flag | Default | Notes |
|------|---------|--------|
| `--stream HOST` | off | receiver's IP; enables streaming |
| `--stream-port` | `5000` | must match `udpsrc port=` on the receiver |
| `--stream-every` | `2` | send every Nth camera frame (camera runs at 30 fps) |

No picture? Check `ping <MAC_IP>` from the Orin, the macOS firewall, and that nothing else uses port 5000 on the Mac. `stream.Streamer` can be used from other scripts too: `Streamer(host).send(bgr_image)`.

---

## Wiring the vehicle

`Vehicle.send(twist)` defaults to **pose dead-reckoning only** (no motors). Override it before you drop `--dry-run`:

```python
from vehicle import Vehicle, Twist

class MyDrive(Vehicle):
    def send(self, twist: Twist):
        # twist.linear_m_s  — forward speed (m/s)
        # twist.angular_rad_s — yaw rate (rad/s), + = turn right in our frame
        your_motor_api(twist.linear_m_s, twist.angular_rad_s)
        super().send(twist)   # keep open-loop pose for the 10 m goal
```

Or assign a function onto an existing instance. Emergency stop and “no path” already send zero twist.

---

## Safety behaviour

- Hazards are circles in metres; radius includes `robot_radius_m`.
- Path planner adds **5 cm** extra clearance.
- **E-stop** if any object edge ≤ `emergency_stop_m`.
- Speed scales down when clearance &lt; **1.2 m**.
- Robot stops if A* finds no safe path.

Keep `--dry-run` until the camera frame, calibration, and motor mapping look correct.

---

## Desktop (non-Jetson) quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt opencv-python-headless
# install librealsense / pyrealsense2 for your OS, then:
python navigator.py --dry-run
```

---

## Layout

```
igvc-not-ivdc/
├── README.md
├── requirements.txt
├── depth_viewer.py
├── hazard_detector.py
├── pathfinder.py
├── vehicle.py
└── navigator.py
```
