# String Manipulation

## dev/creator: tubakhxn

### What is this project about?
This project is a real-time webcam visualizer that draws glowing neon lines and circles following your hand and finger movements. As you move, stretch, or twist your hands, the overlay creates a futuristic, interactive light effect that looks like your fingers are painting with neon light. 

### How to fork and run it
1. **Fork this repository** on GitHub to your own account.
2. **Clone your fork** to your local machine:
	```
	git clone https://github.com/YOUR_USERNAME/REPO_NAME.git
	```
3. **Navigate to the project folder**:
	```
	cd REPO_NAME
	```
4. **(Recommended) Create a virtual environment**:
	```
	python -m venv .venv
	# On Windows:
	.venv\Scripts\activate
	# On Mac/Linux:
	source .venv/bin/activate
	```
5. **Install the requirements**:
	```
	pip install -r requirements.txt
	```
6. **Run the project**:
	```
	python hand_overlay.py
	```

Enjoy the glowing neon hand visualizer!
# Hand Tracking Overlay Project

This project uses Python, MediaPipe, and OpenCV to create a real-time hand tracking overlay similar to the provided reference image. It opens your webcam, tracks your hands and fingers accurately and smoothly, and draws interactive overlays that follow your hand movements.

## Features
- Real-time webcam hand and finger tracking
- Smooth, accurate overlays that do not lose track of fingers
- Custom overlay graphics inspired by the reference image
- Robust and impressive performance

## Requirements
- Python 3.8+
- MediaPipe
- OpenCV
- NumPy

## Setup
1. Ensure you have Python 3.8 or newer installed.
2. All dependencies are installed automatically in the virtual environment.

## Run the Project

```
C:/Users/Tuba Khan/Downloads/streach/.venv/Scripts/python.exe hand_overlay.py
```

- Press `Esc` to exit the webcam window.

## Notes
- The overlay is designed for smoothness and accuracy, even with hand movement.
- You can modify the overlay style in `hand_overlay.py` in the `draw_custom_overlay` function.

---

Made with ❤️ by tubakhxn 
