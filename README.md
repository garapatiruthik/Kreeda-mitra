# Kreeda-Mitra AI - Real-Time Fencing Coach

<p align="center">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python Version">
  <img src="https://img.shields.io/badge/streamlit-1.28%2B-FF4B4B" alt="Streamlit">
  <img src="https://img.shields.io/badge/mediapipe-0.10%2B-00B4D8" alt="MediaPipe">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
</p>

EnGarde AI is a comprehensive AI-powered fencing coaching application that uses computer vision and pose estimation to help coaches and students analyze and improve fencing techniques in real-time.

## Features

- **Dual Video Interface** - Compare coach and student videos side-by-side
- **Real-Time Pose Estimation** - MediaPipe-powered skeleton overlay on video frames
- **Angle Calculation** - Automated calculation of key fencing angles (knees, elbows, hips, shoulders)
- **Movement Comparison** - DTW-based movement similarity analysis
- **Theory Lab** - Interactive fencing fundamentals education
- **Performance Analytics** - Track progress over time with session reports
- **Export Reports** - Download session analysis as CSV

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Steps

1. Clone the repository:
```bash
git clone https://github.com/yourusername/engarde-ai.git
cd engarde-ai
```

2. Create a virtual environment (recommended):
```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application

Start the Streamlit server:
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`.

### Application Navigation

1. **Dashboard** - Main interface for uploading and comparing videos
2. **Theory Lab** - Learn fencing fundamentals and common errors
3. **Performance Analytics** - View session history and progress charts

### Analyzing Videos

1. Upload a coach/reference video in the left panel
2. Upload a student practice video in the right panel
3. Click "Process Videos" to run the analysis
4. View the comparison results including:
   - Side-by-side annotated frames
   - Joint angle comparisons
   - Movement similarity charts
5. Download session reports as CSV

## Project Structure

```
engarde-ai/
├── app.py                    # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore               # Git ignore rules
├── pyproject.toml           # Project configuration
├── src/                     # Source code modules
│   ├── __init__.py
│   ├── pose_analyzer.py    # MediaPipe pose detection
│   ├── angle_calculator.py # Angle calculation utilities
│   ├── movement_compare.py # DTW movement comparison
│   └── video_processor.py  # Video processing utilities
├── tests/                   # Test files
│   ├── __init__.py
│   ├── test_angle_calculator.py
│   ├── test_movement_compare.py
│   └── test_video_processor.py
└── data/                    # Data storage (not tracked by git)
    ├── sessions/           # Session reports
    └── models/             # Saved models
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
```

Available settings:
- `MAX_VIDEO_FRAMES` - Maximum frames to process per video (default: 300)
- `VIDEO_WIDTH` - Video resize width (default: 640)
- `VIDEO_HEIGHT` - Video resize height (default: 480)
- `ANGLE_THRESHOLD` - Matching threshold in degrees (default: 15.0)

### MediaPipe Configuration

Edit `src/pose_analyzer.py` to adjust pose detection parameters:
- `min_detection_confidence` - Minimum detection confidence (0.0-1.0)
- `min_tracking_confidence` - Minimum tracking confidence (0.0-1.0)
- `model_complexity` - Model complexity (0, 1, or 2)

## Key Angles in Fencing

| Joint | Target Angle | Description |
|-------|-------------|-------------|
| Front Knee | 90° max | Prevents injury, proper lunge depth |
| Back Leg | 170-180° | Keep straight during push-off |
| Weapon Arm | 180° | Full extension for attacks |
| Non-Weapon Arm | 90-110° | Balance and counterweight |

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Quality

The project uses:
- Type hints for better code documentation
- Docstrings following Google Python style guide
- Modular architecture for easy extension

## Troubleshooting

### Common Issues

1. **Video processing too slow**
   - Reduce `MAX_VIDEO_FRAMES` in configuration
   - Lower video resolution settings

2. **Pose not detected**
   - Ensure good lighting in video
   - Subject should be fully visible in frame
   - Adjust `min_detection_confidence`

3. **Memory issues**
   - Use `opencv-python-headless` instead of `opencv-python` for server deployments
   - Process shorter video clips

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [MediaPipe](https://mediapipe.dev/) for pose estimation
- [Streamlit](https://streamlit.io/) for the web interface
- [OpenCV](https://opencv.org/) for video processing

---

<p align="center">🤺 "En Garde!" - The game begins.</p>
