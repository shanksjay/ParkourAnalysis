# Parkour Analysis System

AI-powered parkour technique analysis tool that uses YOLO pose estimation and SMPL-X 3D body modeling to provide real-time feedback on parkour movements, comparing user performance against professional reference data.

## Features

- 🎯 **Pose Detection**: Real-time human pose estimation using YOLOv8 pose model
- 🤖 **3D Body Modeling**: SMPL-X parametric body model for accurate 3D pose reconstruction
- 📊 **Technique Analysis**: Automatic analysis of knee angles, jump height, and landing form
- 👥 **Pro Comparison**: Overlay professional reference movements (e.g., Dom Tomato Kong Vault)
- 📹 **Video Processing**: Process video files with frame-by-frame analysis
- 📄 **PDF Reports**: Generate detailed feedback reports
- 🗺️ **Trajectory Visualization**: Top-down view of movement path through 3D space
- 💾 **Data Export**: Save 3D keypoints, trajectory data, and mesh vertices for further analysis
- ⚡ **MPS Support**: Optimized for Apple Silicon (M1/M2/M3) with Metal Performance Shaders

## Requirements

- Python 3.9 or higher
- macOS (for MPS support) or Linux/Windows (CPU/CUDA)
- UV package manager (recommended) or pip

## Installation

### Using UV (Recommended)

1. **Install UV** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Navigate to project directory**:
   ```bash
   cd parkour_analysis
   ```

3. **Create and activate UV environment**:
   ```bash
   uv venv
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   ```

4. **Install dependencies**:
   ```bash
   # First install pip (required for chumpy build)
   uv pip install pip
   
   # Then install all dependencies
   uv pip install -r requirements.txt
   ```

   Or install from pyproject.toml:
   ```bash
   uv pip install pip
   uv pip install -e .
   ```

   **Note**: If you encounter build issues with `chumpy`, the `pyproject.toml` includes build dependencies. Alternatively, use:
   ```bash
   uv pip install pip
   uv pip install -r requirements.txt --no-build-isolation
   ```

### Using pip

1. **Create virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On macOS/Linux
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Download SMPL-X Models

The SMPL-X body model files need to be downloaded separately:

1. **Download SMPL-X models** from [SMPL-X website](https://smpl-x.is.tue.mpg.de/)
2. **Extract** the model files to the project directory
3. The code expects `basicModel_neutral_lbs_10_207_0_v1.1.0.pkl` in the current directory

Alternatively, SMPLX will auto-download models on first use if you have the model path configured.

## Usage

### Basic Usage

1. **Prepare your video**:
   - Place your parkour video file in the project directory
   - Name it `input_parkour.mp4` (or modify the path in code)

2. **Run the analysis**:
   ```bash
   python parkour_analysis.py
   ```

3. **Output files** (generated in project directory):
   - `output_video.mp4`: Processed video with overlays and feedback
   - `trajectory_video.mp4`: Trajectory visualization showing movement path
   - `feedback_report.pdf`: Detailed technique analysis report
   - `keypoints_3d.npz`: 3D keypoints data for all frames
   - `trajectory_data.npz`: Trajectory positions (hip, head, center of mass)
   - `vertices_all_frames.npz`: 3D mesh vertices for all frames
   - `pro_kong_smplx.npz`: Professional reference data (auto-generated if not present)

### Custom Video Path

Modify the `main()` function call or pass a custom path:

```python
if __name__ == "__main__":
    main(video_path="path/to/your/video.mp4")
```

## How It Works

1. **Video Input**: Reads video frames sequentially
2. **Pose Detection**: YOLOv8 pose model detects human keypoints in 2D
3. **3D Reconstruction**: 2D keypoints are lifted to 3D using SMPL-X body model
4. **Technique Analysis**: Analyzes:
   - Knee bend angles (should be ~140° for proper vault)
   - Jump height (target: 1.2m+)
   - Landing form (knee flexion for soft landing)
5. **Pro Comparison**: Overlays professional reference movements for comparison
6. **Feedback Generation**: Provides real-time feedback and generates PDF report

## Analysis Metrics

The system analyzes:

- **Knee Angle**: Measures knee flexion during vault (target: ~140°)
- **Jump Height**: Calculates maximum hip height (target: 1.2m+)
- **Landing Form**: Evaluates knee flexion on landing (should be <30° for soft landing)

## Output Features

- **Real-time Feedback**: On-screen text feedback during video processing
- **Pro Overlay**: Semi-transparent professional reference overlay
- **Slow-motion**: Automatically slows down frames with technique mistakes
- **PDF Report**: Comprehensive feedback report with scores and recommendations
- **Trajectory Visualization**: Top-down view of movement path through 3D space
- **Intermediate Artifacts**: Saved 3D keypoints, trajectory data, and mesh vertices for further analysis

## Project Structure

### Source Files

```
parkour_analysis/
├── parkour_analysis.py    # Main analysis script
├── pyproject.toml         # UV/pip project configuration
├── requirements.txt       # Python dependencies
├── requirements-base.txt  # Base dependencies (without PyTorch)
├── setup.sh              # Automated setup script
├── README.md             # This file
├── .gitignore            # Git ignore patterns
└── models/               # SMPL-X model directory (user-provided)
    └── smplx/
        ├── SMPLX_NEUTRAL.pkl
        ├── SMPLX_NEUTRAL.npz
        └── ... (other model files)
```

### Input Files (User-Provided)

```
├── input_parkour.mp4     # Input video file (or specify custom path)
└── flip_in_beach.mp4     # Alternative input video name
```

### Generated Output Files

```
├── output_video.mp4           # Main analysis video with feedback overlays
├── trajectory_video.mp4       # Trajectory visualization (top-down X-Z view)
├── feedback_report.pdf        # Technique analysis PDF report
├── pro_kong_smplx.npz         # Professional reference data (auto-generated)
├── keypoints_3d.npz           # 3D keypoints for all frames
├── trajectory_data.npz        # Trajectory positions (hip, head, center of mass)
└── vertices_all_frames.npz     # 3D mesh vertices for all frames
```

**Output File Descriptions:**

- **output_video.mp4**: Main processed video with real-time feedback, technique analysis, and pro comparison overlays
- **trajectory_video.mp4**: Top-down visualization showing the athlete's movement path through 3D space with color-coded trajectory
- **feedback_report.pdf**: Detailed technique analysis with scores and recommendations
- **keypoints_3d.npz**: NumPy archive containing 3D keypoint coordinates for each frame (useful for further analysis)
- **trajectory_data.npz**: Trajectory data including hip positions, head positions, and center of mass for motion analysis
- **vertices_all_frames.npz**: Complete 3D mesh vertex coordinates for all frames (enables 3D reconstruction and detailed biomechanical analysis)

## Dependencies

- **opencv-python**: Video processing and image manipulation
- **torch/torchvision**: Deep learning framework (with MPS support for Apple Silicon)
- **ultralytics**: YOLOv8 pose estimation
- **smplx**: SMPL-X parametric body model
- **trimesh/pyrender**: 3D mesh processing and rendering
- **scipy**: Scientific computing (rotation transforms)
- **reportlab**: PDF report generation
- **numpy**: Numerical computing
- **matplotlib**: Plotting (for potential visualization)

## Performance Notes

- **MPS Acceleration**: Automatically uses Apple Silicon GPU if available
- **Frame Processing**: Processes frames sequentially (can be parallelized)
- **Model Loading**: YOLO and SMPL-X models load on first run (may take time)
- **Memory Usage**: Moderate - processes one frame at a time

## Troubleshooting

### Model Download Issues

If YOLO model doesn't auto-download:
```python
from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")  # Will download automatically
```

### SMPL-X Model Path

If you get errors about SMPL-X model:
1. Download models from [SMPL-X website](https://smpl-x.is.tue.mpg.de/)
2. Update the `model_path` in the code to point to your model file

### MPS Not Available

If MPS is not detected:
- Ensure you're on macOS with Apple Silicon (M1/M2/M3)
- PyTorch should be installed with MPS support (default on macOS)
- Falls back to CPU automatically

### Chumpy Build Issues (UV)

If you encounter `ModuleNotFoundError: No module named 'pip'` when building `chumpy`:

**Solution 1** (Recommended):
```bash
uv pip install pip
uv pip install -r requirements.txt
```

**Solution 2** (Alternative):
```bash
uv pip install pip
uv pip install -r requirements.txt --no-build-isolation
```

The `pyproject.toml` includes build dependencies for chumpy, but you may still need to install pip first.

### Video Codec Issues

If video output doesn't play:
- Try changing the codec in `cv2.VideoWriter_fourcc(*'mp4v')` to `'avc1'` or `'x264'`
- Ensure ffmpeg is installed for better codec support

## Future Enhancements

- [ ] Real-time webcam processing
- [ ] Multiple movement types (not just Kong Vault)
- [ ] More sophisticated 3D pose optimization
- [ ] Integration with LAAS parkour dataset
- [ ] Web interface for easier use
- [ ] Batch processing for multiple videos
- [ ] Advanced tracking with ByteTrack

## References

- **YOLOv8**: [Ultralytics](https://github.com/ultralytics/ultralytics)
- **SMPL-X**: [SMPL-X: Expressive Body Capture](https://smpl-x.is.tue.mpg.de/)
- **LAAS Parkour Dataset**: [Gepetto Web](https://gepettoweb.laas.fr/parkour/)

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please feel free to submit issues or pull requests.

## Acknowledgments

- Professional reference data inspired by Dom Tomato's parkour techniques
- LAAS parkour motion capture dataset
- SMPL-X body model by Max Planck Institute

