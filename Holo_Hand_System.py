"""
Holo Hand System
=================
Turns your webcam into a "hologram" display: a glowing 3D wireframe shape
(sphere / cube / pyramid / octagonal prism) floats above your palm and
spins as you move your hand. Built with OpenCV (video + rendering) and
MediaPipe's Hand Landmarker (hand tracking).

SETUP
-----
    pip install opencv-python mediapipe numpy

The hand-tracking model (~10 MB, one-time) is downloaded automatically
into the same folder as this script the first time you run it. If that
machine has no internet access, download it yourself from:
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
and save it next to this script as "hand_landmarker.task".

CONTROLS
--------
    1 / 2 / 3 / 4     switch shape: Sphere / Cube / Pyramid / Octagonal Prism
    Point with your control hand (index finger up, other fingers curled)
        and move it   -> the hologram spins with your finger  ("ACTIVE")
    Hold your OTHER hand up showing 1-4 fingers -> also switches shape
    Hold still (no special gesture)             -> slow idle spin ("STANDBY")
    Pinch (thumb + index tip touching)          -> GRAB the hologram
    Pinch and move your hand                    -> DRAG the hologram around
    Open palm (control hand)                    -> hologram DISSOLVES away
    Make a fist while dissolved                 -> hologram MATERIALIZES back
    Make a fist while visible                   -> a pulse/shockwave burst
    Thumbs up                                   -> a bigger power-up burst
    Peace sign (index + middle up)              -> hologram cycles color
    q or Esc          quit

A faint animated grid, drifting particles, and a scanning light band play
continuously in the background for a "sci-fi hologram room" feel.
"""

import colorsys
import math
import os
import time
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/latest/hand_landmarker.task"
)
CAM_INDEX = 0
FRAME_W, FRAME_H = 960, 720

HOLO_COLOR = (255, 255, 0)      # cyan, BGR
HAND_COLOR = (80, 255, 120)     # green, BGR
CAM_DIST = 4.0
FOCAL = 4.0
SHAPE_SCALE_PX = 95
ROT_SENSITIVITY = 0.010
IDLE_SPIN_SPEED = 0.012
MAX_DELTA_PX = 40

# Gesture-effect timings
DISSOLVE_DURATION = 0.7      # seconds for open-palm vanish animation
MATERIALIZE_DURATION = 0.7   # seconds for fist-triggered reappear animation
EFFECT_DURATION = 0.45       # seconds for a pulse / burst flash
SCATTER_MIN, SCATTER_MAX = 110, 240   # px radius range particles fly out to


# --------------------------------------------------------------------------
# One-time model download
# --------------------------------------------------------------------------
def ensure_model(path):
    if os.path.exists(path):
        return path
    print(f"Downloading hand-tracking model to {path} (one-time, ~10 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Download complete.")
    except Exception as exc:
        raise RuntimeError(
            "Could not download the hand-tracking model automatically "
            f"({exc}). Please download it manually from:\n{MODEL_URL}\n"
            f'and save it as "{path}" next to this script.'
        ) from exc
    return path


# --------------------------------------------------------------------------
# 3D shapes: each returns (vertices Nx3 float64, edges list[(i, j)])
# Object space: x = right, y = down, z = into the screen (matches screen
# axes directly so no extra flips are needed when projecting).
# --------------------------------------------------------------------------
def make_cube(size=1.15):
    s = size
    v = np.array([
        [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],
        [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s],
    ], dtype=np.float64)
    e = [(0, 1), (1, 2), (2, 3), (3, 0),
         (4, 5), (5, 6), (6, 7), (7, 4),
         (0, 4), (1, 5), (2, 6), (3, 7)]
    return v, e


def make_pyramid(base=1.6, height=1.7):
    b, h = base / 2, height / 2
    v = np.array([
        [-b, h, -b], [b, h, -b], [b, h, b], [-b, h, b],  # square base
        [0, -h, 0],                                       # apex
    ], dtype=np.float64)
    e = [(0, 1), (1, 2), (2, 3), (3, 0),
         (0, 4), (1, 4), (2, 4), (3, 4)]
    return v, e


def make_octagonal_prism(radius=1.05, height=1.6, sides=8):
    h = height / 2
    top = [[radius * math.cos(2 * math.pi * k / sides), -h,
            radius * math.sin(2 * math.pi * k / sides)] for k in range(sides)]
    bot = [[radius * math.cos(2 * math.pi * k / sides), h,
            radius * math.sin(2 * math.pi * k / sides)] for k in range(sides)]
    v = np.array(top + bot, dtype=np.float64)
    e = []
    for k in range(sides):
        e.append((k, (k + 1) % sides))
        e.append((sides + k, sides + (k + 1) % sides))
        e.append((k, sides + k))
    return v, e


def make_sphere(radius=1.25, lat_steps=6, lon_steps=12):
    verts = [[0.0, -radius, 0.0], [0.0, radius, 0.0]]  # 0: top pole, 1: bottom pole
    ring_offsets = []
    for i in range(1, lat_steps):
        theta = math.pi * i / lat_steps
        y = -radius * math.cos(theta)
        r = radius * math.sin(theta)
        ring_offsets.append(len(verts))
        for j in range(lon_steps):
            phi = 2 * math.pi * j / lon_steps
            verts.append([r * math.cos(phi), y, r * math.sin(phi)])
    e = []
    for ring_start in ring_offsets:
        for j in range(lon_steps):
            e.append((ring_start + j, ring_start + (j + 1) % lon_steps))
    for j in range(lon_steps):
        e.append((0, ring_offsets[0] + j))
        for k in range(len(ring_offsets) - 1):
            e.append((ring_offsets[k] + j, ring_offsets[k + 1] + j))
        e.append((ring_offsets[-1] + j, 1))
    return np.array(verts, dtype=np.float64), e


SHAPES = [
    ("SPHERE", *make_sphere()),
    ("CUBE", *make_cube()),
    ("PYRAMID", *make_pyramid()),
    ("OCTAGONAL PRISM", *make_octagonal_prism()),
]


# --------------------------------------------------------------------------
# Math helpers
# --------------------------------------------------------------------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def smoothstep(t):
    """Ease 0..1 with a soft start/end, used for the vanish/materialize animations."""
    t = clamp(t, 0.0, 1.0)
    return t * t * (3 - 2 * t)


def cycle_color(t, speed=0.6):
    """Rainbow BGR color that shifts over time, used for the peace-sign effect."""
    r, g, b = colorsys.hsv_to_rgb((t * speed) % 1.0, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))


def make_scatter_offsets(n):
    """Random outward (dx, dy) offsets, one per vertex, for the dissolve/
    materialize particle animation."""
    rng = np.random.default_rng()
    angles = rng.uniform(0, 2 * math.pi, n)
    radii = rng.uniform(SCATTER_MIN, SCATTER_MAX, n)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1) * radii[:, None]


def rotation_matrix(rx, ry):
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    rot_x_m = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    rot_y_m = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    return rot_y_m @ rot_x_m


def project(verts3d, anchor_xy, scale_px):
    z = np.clip(verts3d[:, 2] + CAM_DIST, 0.5, None)
    f = FOCAL / z
    x = anchor_xy[0] + verts3d[:, 0] * f * scale_px
    y = anchor_xy[1] + verts3d[:, 1] * f * scale_px
    return np.stack([x, y], axis=1)


# --------------------------------------------------------------------------
# Hand-landmark helpers (each takes a list of 21 landmarks with .x / .y)
# --------------------------------------------------------------------------
def palm_center_px(lm, w, h):
    idxs = (0, 5, 9, 13, 17)
    xs = sum(lm[i].x for i in idxs) / len(idxs)
    ys = sum(lm[i].y for i in idxs) / len(idxs)
    return xs * w, ys * h


def index_tip_px(lm, w, h):
    return lm[8].x * w, lm[8].y * h


def fingers_up(lm):
    """[index, middle, ring, pinky] booleans. Skips the thumb since its
    open/closed direction depends on which hand it is, which we don't
    need to know here."""
    tips = (8, 12, 16, 20)
    pips = (6, 10, 14, 18)
    return [lm[t].y < lm[p].y for t, p in zip(tips, pips)]


def _dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def thumb_extended(lm):
    """True if the thumb is stuck out away from the palm. Uses distances
    (not left/right direction) so it works for either hand."""
    return _dist(lm[4], lm[17]) > _dist(lm[2], lm[17]) * 1.15


def is_pinching(lm):
    """True if thumb tip and index tip are touching. Scaled by the hand's
    own size (wrist-to-middle-MCP) so it works at any distance from camera."""
    hand_scale = max(_dist(lm[0], lm[9]), 1e-6)
    return _dist(lm[4], lm[8]) < hand_scale * 0.4


def pinch_point_px(lm, w, h):
    mx = (lm[4].x + lm[8].x) / 2
    my = (lm[4].y + lm[8].y) / 2
    return mx * w, my * h


def classify_gesture(lm):
    """Classify a hand's pose into one of the named gestures used for
    hologram effects, based on which fingers are up plus the thumb."""
    if is_pinching(lm):
        return "PINCH"
    f = fingers_up(lm)
    thumb = thumb_extended(lm)
    count = sum(f)
    if count >= 3 and thumb:
        return "OPEN_PALM"
    if count == 0 and not thumb:
        return "FIST"
    if f[0] and not f[1] and not f[2] and not f[3]:
        return "POINT"
    if f[0] and f[1] and not f[2] and not f[3]:
        return "PEACE"
    if count == 0 and thumb:
        return "THUMBS_UP"
    return "OTHER"


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------
def make_grid_overlay(w, h, spacing=48, color=(55, 55, 15)):
    """A faint static grid (BGR, dim teal) blended under everything each
    frame to give the scene a sci-fi 'holo room' floor/backdrop feel."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for x in range(0, w, spacing):
        cv2.line(img, (x, 0), (x, h), color, 1, cv2.LINE_AA)
    for y in range(0, h, spacing):
        cv2.line(img, (0, y), (w, y), color, 1, cv2.LINE_AA)
    return img


def make_particles(w, h, n=45):
    """Slowly-rising, twinkling 'data mote' particles for the background."""
    rng = np.random.default_rng()
    return [{
        "x": float(rng.uniform(0, w)),
        "y": float(rng.uniform(0, h)),
        "vy": float(rng.uniform(8, 26)),
        "size": int(rng.integers(1, 3)),
        "phase": float(rng.uniform(0, 2 * math.pi)),
    } for _ in range(n)]


def draw_animated_background(frame, grid_overlay, particles, t, dt, w, h):
    """Blend a faint grid, a downward scanning light band, and drifting
    particles into the frame. Runs every frame, under the hand/hologram."""
    frame = cv2.addWeighted(frame, 1.0, grid_overlay, 0.35, 0)

    band_h = 40
    band_y = int((t * 90) % (h + band_h)) - band_h
    if 0 <= band_y < h:
        y0, y1 = max(0, band_y), min(h, band_y + band_h)
        sub = frame[y0:y1].astype(np.float32)
        sub[:, :, 0] += 22   # B
        sub[:, :, 1] += 36   # G  (cyan-leaning tint)
        frame[y0:y1] = np.clip(sub, 0, 255).astype(np.uint8)

    overlay = np.zeros_like(frame)
    for p in particles:
        p["y"] -= p["vy"] * dt
        if p["y"] < -5:
            p["y"] = float(h + 5)
            p["x"] = float(np.random.uniform(0, w))
        twinkle = 0.5 + 0.5 * math.sin(t * 2 + p["phase"])
        c = int(120 + 100 * twinkle)
        cv2.circle(overlay, (int(p["x"]), int(p["y"])), p["size"],
                   (c, c, int(c * 0.3)), -1, cv2.LINE_AA)
    frame = cv2.add(frame, overlay)
    return frame


def draw_hologram(frame, pts2d, edges, color=HOLO_COLOR, alpha=1.0):
    if alpha <= 0.02:
        return frame
    c = tuple(int(ch * alpha) for ch in color)
    overlay = np.zeros_like(frame)
    pts_int = np.round(pts2d).astype(int)
    for i, j in edges:
        p1 = (int(pts_int[i][0]), int(pts_int[i][1]))
        p2 = (int(pts_int[j][0]), int(pts_int[j][1]))
        cv2.line(overlay, p1, p2, c, 2, cv2.LINE_AA)
    for p in pts_int:
        cv2.circle(overlay, (int(p[0]), int(p[1])), 3, c, -1, cv2.LINE_AA)
    glow = cv2.GaussianBlur(overlay, (0, 0), 9)
    frame = cv2.add(frame, glow)
    frame = cv2.add(frame, overlay)
    return frame


def draw_hand_skeleton(frame, lm, w, h, connections, color=HAND_COLOR):
    pts = [(int(p.x * w), int(p.y * h)) for p in lm]
    for a, b in connections:
        cv2.line(frame, pts[a], pts[b], color, 1, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 2, color, -1, cv2.LINE_AA)


def draw_hud(frame, shape_name, status, gesture_label=None, hud_color=HOLO_COLOR):
    x, y, bw, bh = 15, 15, 300, 78
    box = frame.copy()
    cv2.rectangle(box, (x, y), (x + bw, y + bh), (15, 15, 15), -1)
    frame = cv2.addWeighted(box, 0.55, frame, 0.45, 0)
    cv2.putText(frame, f"HOLO SYSTEM: {shape_name}", (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"STATUS: {status}", (x + 10, y + 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"GESTURE: {gesture_label or '-'}", (x + 10, y + 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_color, 1, cv2.LINE_AA)
    cv2.putText(frame, "1-4 shape  point+move rotate  palm vanish  "
                        "fist return/pulse  peace color  q quit",
                (15, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (200, 200, 200), 1, cv2.LINE_AA)
    return frame


# --------------------------------------------------------------------------
# Main loop
# --------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = ensure_model(os.path.join(script_dir, MODEL_FILENAME))

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    connections = [(c.start, c.end)
                   for c in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS]

    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check CAM_INDEX / permissions.")

    shape_index = 0
    rot_x, rot_y = 0.35, 0.0
    prev_fingertip = None
    start_time = time.perf_counter()
    last_time = start_time

    # Hologram lifecycle: VISIBLE -> DISSOLVING -> HIDDEN -> MATERIALIZING -> VISIBLE
    holo_state = "VISIBLE"
    state_time = 0.0
    scatter_offsets = None      # per-vertex outward (dx, dy) for the current animation
    effect_type = None          # None, "PULSE" (fist) or "BURST" (thumbs up)
    effect_time = 0.0

    # Grab-and-drag: the hologram sits at object_pos until pinched, at which
    # point it follows the pinch point (with an offset so it doesn't jump).
    object_pos = None
    grabbing = False
    grab_offset = (0.0, 0.0)

    # Animated background (built lazily once we know the real frame size).
    grid_overlay = None
    particles = None

    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame = cv2.flip(frame, 1)
                h, w = frame.shape[:2]

                now = time.perf_counter()
                dt = max(now - last_time, 1e-3)
                last_time = now
                if grid_overlay is None:
                    grid_overlay = make_grid_overlay(w, h)
                    particles = make_particles(w, h)
                if object_pos is None:
                    object_pos = (w / 2.0, h / 2.0)

                # Detect on the clean camera frame first, then paint the
                # animated background so it sits under the hand/hologram.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB,
                                     data=np.ascontiguousarray(rgb))
                timestamp_ms = int((now - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                frame = draw_animated_background(frame, grid_overlay, particles,
                                                  now - start_time, dt, w, h)

                control_lm, control_pointing = None, False
                selector_fingers = None

                if result.hand_landmarks:
                    infos = []
                    for lm in result.hand_landmarks:
                        f = fingers_up(lm)
                        pointing = f[0] and not any(f[1:])
                        infos.append((lm, f, pointing))
                        draw_hand_skeleton(frame, lm, w, h, connections)

                    pointing_hands = [info for info in infos if info[2]]
                    control_lm, control_fingers, control_pointing = (
                        pointing_hands[0] if pointing_hands else infos[0]
                    )
                    others = [info for info in infos if info[0] is not control_lm]
                    if others:
                        selector_fingers = others[0][1]

                if selector_fingers is not None:
                    count = sum(selector_fingers)
                    if 1 <= count <= len(SHAPES):
                        shape_index = count - 1

                gesture = classify_gesture(control_lm) if control_lm is not None else None

                if control_lm is not None:
                    if gesture == "PINCH":
                        pinch_pt = pinch_point_px(control_lm, w, h)
                        if not grabbing:
                            grabbing = True
                            grab_offset = (object_pos[0] - pinch_pt[0],
                                           object_pos[1] - pinch_pt[1])
                        object_pos = (pinch_pt[0] + grab_offset[0],
                                      pinch_pt[1] + grab_offset[1])
                        prev_fingertip = None
                        status = "GRABBED"
                    elif control_pointing:
                        grabbing = False
                        tip = index_tip_px(control_lm, w, h)
                        if prev_fingertip is not None:
                            dx = clamp(tip[0] - prev_fingertip[0], -MAX_DELTA_PX, MAX_DELTA_PX)
                            dy = clamp(tip[1] - prev_fingertip[1], -MAX_DELTA_PX, MAX_DELTA_PX)
                            rot_y += dx * ROT_SENSITIVITY
                            rot_x += dy * ROT_SENSITIVITY
                        prev_fingertip = tip
                        status = "ACTIVE"
                    else:
                        grabbing = False
                        rot_y += IDLE_SPIN_SPEED
                        prev_fingertip = None
                        status = "STANDBY"

                    anchor = object_pos

                    # --- gesture-driven lifecycle transitions -------------------
                    if holo_state == "VISIBLE":
                        if gesture == "OPEN_PALM":
                            holo_state = "DISSOLVING"
                            state_time = now
                            effect_type = None
                            scatter_offsets = make_scatter_offsets(len(SHAPES[shape_index][1]))
                        elif gesture == "FIST" and effect_type is None:
                            effect_type, effect_time = "PULSE", now
                        elif gesture == "THUMBS_UP" and effect_type is None:
                            effect_type, effect_time = "BURST", now
                    elif holo_state == "DISSOLVING":
                        if now - state_time >= DISSOLVE_DURATION:
                            holo_state = "HIDDEN"
                    elif holo_state == "HIDDEN":
                        if gesture is not None and gesture != "OPEN_PALM":
                            holo_state = "MATERIALIZING"
                            state_time = now
                            scatter_offsets = make_scatter_offsets(len(SHAPES[shape_index][1]))
                    elif holo_state == "MATERIALIZING":
                        if now - state_time >= MATERIALIZE_DURATION:
                            holo_state = "VISIBLE"

                    name, verts, edges = SHAPES[shape_index]
                    rotated = verts @ rotation_matrix(rot_x, rot_y).T
                    if scatter_offsets is None or len(scatter_offsets) != len(verts):
                        scatter_offsets = make_scatter_offsets(len(verts))

                    color = HOLO_COLOR
                    scale_px = SHAPE_SCALE_PX
                    alpha = 1.0
                    display_status = status

                    if holo_state == "VISIBLE":
                        if gesture == "PEACE":
                            color = cycle_color(now)
                        if effect_type is not None:
                            te = clamp((now - effect_time) / EFFECT_DURATION, 0.0, 1.0)
                            amp = 0.35 if effect_type == "PULSE" else 0.6
                            flash = math.sin(te * math.pi)
                            scale_px = SHAPE_SCALE_PX * (1 + amp * flash)
                            color = tuple(int(c + (255 - c) * flash * 0.6) for c in color)
                            if te >= 1.0:
                                effect_type = None
                        pts2d = project(rotated, anchor, scale_px)
                    elif holo_state == "DISSOLVING":
                        t = smoothstep((now - state_time) / DISSOLVE_DURATION)
                        pts2d = project(rotated, anchor, SHAPE_SCALE_PX) + scatter_offsets * t
                        alpha = 1.0 - t
                        display_status = "VANISHING"
                    elif holo_state == "HIDDEN":
                        pts2d = None
                        alpha = 0.0
                        display_status = "HIDDEN - make a fist to return"
                    else:  # MATERIALIZING
                        t = smoothstep((now - state_time) / MATERIALIZE_DURATION)
                        pts2d = project(rotated, anchor, SHAPE_SCALE_PX) + scatter_offsets * (1.0 - t)
                        alpha = t
                        display_status = "MATERIALIZING"

                    if pts2d is not None:
                        frame = draw_hologram(frame, pts2d, edges, color=color, alpha=alpha)
                    frame = draw_hud(frame, name, display_status, gesture_label=gesture)
                else:
                    prev_fingertip = None
                    grabbing = False
                    frame = draw_hud(frame, SHAPES[shape_index][0], "NO HAND")

                cv2.imshow("Holo Hand System", frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif ord('1') <= key <= ord('4'):
                    shape_index = key - ord('1')
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()