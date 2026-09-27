Holo Hand System :-

A webcam-based "hologram" display — a glowing 3D wireframe shape floats in front of you and responds to hand gestures. No headset, no markers, just your webcam and your hands. Built with OpenCV and MediaPipe's Hand Landmarker.

✨ Features
4 hologram shapes: Sphere, Cube, Pyramid, Octagonal Prism
Gesture-controlled rotation — point and move your finger to spin the object
Grab & drag — pinch to pick up the hologram and move it anywhere on screen
Dissolve / materialize — open your palm to make it vanish in a particle burst, make a fist to bring it back
Pulse & burst effects — fist and thumbs-up trigger shockwave-style flashes
Color cycling — peace sign shifts the hologram through a rainbow
Animated sci-fi background — drifting particles, scanning light band, and a subtle grid layered behind everything
Two-hand support — use one hand to control the object, the other to pick a shape by finger count


🎮 Controls

Gesture	Action

☝️ Point (index only) + move	Rotate the hologram

🤏 Pinch	Grab / select the hologram

🤏 Pinch + move hand	Drag the hologram around

✋ Open palm	Hologram dissolves and disappears

✊ Fist (while hidden)	Hologram materializes back

✊ Fist (while visible)	Pulse / shockwave burst

👍 Thumbs up	Bigger power-up burst

✌️ Peace sign	Hologram cycles through colors

Hold still / open hand idle	Slow ambient idle spin

Other hand: 1–4 fingers	Switch shape (Sphere / Cube / Pyramid / Octagonal Prism)

1 2 3 4 (keyboard)	Switch shape directly

q / Esc	Quit

🛠 Requirements
Python 3.8+
A webcam
📦 Installation
bash
pip install opencv-python mediapipe numpy

The hand-tracking model (~10 MB) downloads automatically the first time you run the script, into the same folder. If your machine has no internet access, grab it manually:

https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task

and save it next to the script as hand_landmarker.task.

🚀 Usage
bash
python Holo_Hand_System.py

A window will open showing your webcam feed with the hologram overlaid. Use the gestures or keys above to control it.
