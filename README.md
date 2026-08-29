# Pomodoro Attention Monitor

A lightweight Pomodoro timer that uses the computer's camera to monitor user focus during study sessions.

## Features
- Classic 25-minute Pomodoro session with 5-minute short break
- Real-time attention monitoring using webcam
- Focus score based on color analysis in video frames (red, green, blue)
- Visual feedback showing focus level
- Simple command-line interface

## Setup
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage
- Start a session with `main.py` after running the script
- Press 'q' to quit during video feed
- Focus score is displayed in real-time (0-100 scale)
- Session ends when time elapses or user presses 'q'

## How it works
- Uses OpenCV to access webcam and capture video frames
- Analyzes each frame for focus indicators using color analysis
- Focus score is calculated based on presence of red, green, and blue colors in the frame
- Displays focus score on screen with visual feedback
- Automatically switches between pomodoro and short break phases

## Limitations
- Simple color-based focus detection (not actual attention tracking)
- Accuracy depends on user's environment and lighting conditions
- May not work well in low-light or dark environments
- Does not detect actual user behavior or cognitive state

## Future Improvements
- Implement advanced face recognition for actual attention analysis
- Add audio-based focus detection
- Include machine learning models for attention prediction
- Support for multiple cameras
- More sophisticated focus indicators (eye movement, blink rate, etc.)
- User profile and habit tracking
- Social sharing of study progress
- Integration with calendar or task management tools
- Mobile app version with notifications
- Cross-platform support (Windows, macOS, Linux)