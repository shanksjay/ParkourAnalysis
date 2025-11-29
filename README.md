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
   - You'll need to register and request access
   - Download the model files (SMPLX_NEUTRAL.pkl, SMPLX_NEUTRAL.npz, etc.)

2. **Extract** the model files to the project directory:
   ```bash
   mkdir -p models/smplx
   # Extract downloaded files to models/smplx/
   ```

3. **Model Structure**: The code expects models in `models/smplx/`:
   ```
   models/smplx/
   ├── SMPLX_NEUTRAL.pkl
   ├── SMPLX_NEUTRAL.npz
   ├── SMPLX_MALE.pkl (optional)
   ├── SMPLX_FEMALE.pkl (optional)
   └── ... (other model files)
   ```

4. **Model Loading**: The system will automatically:
   - Detect available models in `models/smplx/`
   - Use SMPLX_NEUTRAL.pkl for neutral gender analysis
   - Display model information on startup
   - Fall back to alternative paths if needed

## Usage

### Command Line Options

The script accepts an optional video path argument:

```bash
# Use default video (flip_in_beach.mp4)
python parkour_analysis.py

# Specify custom video path
python parkour_analysis.py /path/to/your/video.mp4

# Relative path also works
python parkour_analysis.py my_parkour_video.mp4
```

### Basic Usage

1. **Prepare your video**:
   - Place your parkour video file in the project directory
   - Default video name: `flip_in_beach.mp4` (or specify via command line)

2. **Run the analysis**:
   ```bash
   # Use default video (flip_in_beach.mp4)
   python parkour_analysis.py
   
   # Or specify a custom video path
   python parkour_analysis.py path/to/your/video.mp4
   ```

3. **Output files** (generated in project directory):
   - `output_video.mp4`: Processed video with overlays and feedback
   - `trajectory_video.mp4`: Trajectory visualization overlaid on original video (camera view)
   - `feedback_report.pdf`: Detailed technique analysis report
   - `keypoints_3d.npz`: 3D keypoints data for all frames
   - `trajectory_data.npz`: Trajectory positions (hip, head, center of mass)
   - `vertices_all_frames.npz`: 3D mesh vertices for all frames
   - `pro_kong_smplx.npz`: Professional reference data (auto-generated if not present)

### Viewing NPZ Files

Use the included utility to inspect generated NPZ files:

```bash
# View all available NPZ files
python view_npz.py

# View a specific file
python view_npz.py keypoints_3d.npz

# View with limited output
python view_npz.py trajectory_data.npz --max-items 5
```

## How It Works

### Processing Pipeline

1. **Video Input**: 
   - Reads video frames sequentially from input file
   - Stores original frames for trajectory visualization
   - Supports custom video paths via command line argument

2. **Pose Detection**: 
   - YOLOv8 pose model detects human keypoints in 2D
   - Tracks the main athlete (largest bounding box)
   - Extracts 17 COCO-format keypoints per frame

3. **3D Reconstruction**: 
   - 2D keypoints are converted to 3D using SMPL-X body model
   - Estimates depth from keypoint bounding box size
   - Generates full 3D mesh vertices and joint positions
   - Uses neutral gender SMPL-X model for general analysis

4. **Technique Analysis**: 
   - Analyzes knee bend angles (target: ~140° for proper vault)
   - Calculates jump height from hip position (target: 1.2m+)
   - Evaluates landing form (knee flexion for soft landing)
   - Provides frame-by-frame feedback

5. **Trajectory Tracking**: 
   - Tracks hip position, head position, and center of mass
   - Projects 3D trajectory to 2D using camera view
   - Overlays trajectory path on original video frames
   - Color-codes trajectory (blue = start, red = end)

6. **Pro Comparison**: 
   - Overlays professional reference movements for comparison
   - Shows synthetic reference data (can be replaced with real LAAS dataset)

7. **Output Generation**: 
   - Generates main analysis video with feedback overlays
   - Creates trajectory visualization video matching camera view
   - Saves intermediate artifacts (keypoints, trajectory, vertices)
   - Generates PDF report with technique scores

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
- **Trajectory Visualization**: Movement path overlaid on original video matching camera view
- **Intermediate Artifacts**: Saved 3D keypoints, trajectory data, and mesh vertices for further analysis

## Project Structure

### Source Files

```
parkour_analysis/
├── parkour_analysis.py    # Main analysis script
├── pyproject.toml         # UV/pip project configuration
├── requirements.txt       # Python dependencies
├── setup.sh              # Automated setup script
├── view_npz.py           # Utility to view NPZ file contents
├── README.md             # This file
├── .gitignore            # Git ignore patterns
└── models/               # SMPL-X model directory (user-provided)
    └── smplx/
        ├── SMPLX_NEUTRAL.pkl
        ├── SMPLX_NEUTRAL.npz
        ├── SMPLX_MALE.pkl (optional)
        ├── SMPLX_FEMALE.pkl (optional)
        └── ... (other model files)
```

### Input Files (User-Provided)

```
├── flip_in_beach.mp4     # Default input video file
└── [custom].mp4          # Any video file (specify via command line)
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
- **trajectory_video.mp4**: Trajectory visualization overlaid on original video frames, showing movement path from the same camera perspective with color-coded trajectory (blue = start, red = end)
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

- **MPS Acceleration**: Automatically uses Apple Silicon GPU if available (M1/M2/M3)
- **Frame Processing**: Processes frames sequentially with progress updates every 30 frames
- **Model Loading**: 
  - YOLO model auto-downloads on first run (~6MB)
  - SMPL-X models must be downloaded separately (~500MB per model)
  - Models load once and are reused for all frames
- **Memory Usage**: Moderate - processes one frame at a time, stores frames for trajectory video
- **Processing Speed**: ~18-26ms per frame on Apple Silicon with MPS
- **Output Generation**: Trajectory video generation happens after main processing completes

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
2. Ensure models are in `models/smplx/` directory
3. Check that `SMPLX_NEUTRAL.pkl` exists in the models directory
4. The code will automatically detect and load models from `models/smplx/`
5. If using a different path, the code will try alternative locations automatically

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

## Workflow Summary

1. **Setup**: Install dependencies and download SMPL-X models
2. **Input**: Provide video file (default: `flip_in_beach.mp4` or specify path)
3. **Processing**: 
   - System loads YOLO and SMPL-X models
   - Processes each frame: pose detection → 3D reconstruction → analysis
   - Stores intermediate data (keypoints, trajectory, vertices)
4. **Output**: 
   - Main analysis video with feedback
   - Trajectory video with camera-matched overlay
   - PDF report with technique scores
   - NPZ files for further analysis
5. **Analysis**: Use `view_npz.py` to inspect generated data files

## Future Enhancements

- [ ] Real-time webcam processing
- [ ] Multiple movement types (not just Kong Vault)
- [ ] More sophisticated 3D pose optimization using actual SMPL-X parameters
- [ ] Integration with real LAAS parkour dataset (replace synthetic reference)
- [ ] Web interface for easier use
- [ ] Batch processing for multiple videos
- [ ] Advanced tracking with ByteTrack
- [ ] Interactive 3D visualization of trajectory
- [ ] Export to common motion capture formats (BVH, FBX)

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

