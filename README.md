# Parkour Analysis System

AI-powered parkour technique analysis tool that uses **SAM3 (Segment Anything Model 3)** for person tracking and trajectory generation, providing real-time feedback on parkour movements based on trajectory analysis.

## Features

- 🎯 **Person Tracking**: Real-time person segmentation and tracking using SAM3 (Segment Anything Model 3)
- 📍 **Trajectory Generation**: Automatic trajectory extraction from SAM3 segmentation masks
- 📊 **Trajectory Analysis**: Automatic analysis of speed, acceleration, jump height, and movement smoothness
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

### Install SAM3

SAM3 (Segment Anything Model 3) is required for person tracking. If you have already cloned SAM3 into the `parkour_analysis` directory, install it:

1. **Install SAM3 from local directory**:
   ```bash
   cd sam3
   pip install -e .
   pip install -e ".[notebooks]"  # Optional: for notebook support
   cd ..
   ```

   Or if SAM3 is not yet cloned:
   ```bash
   git clone https://github.com/facebookresearch/sam3.git
   cd sam3
   pip install -e .
   pip install -e ".[notebooks]"
   cd ..
   ```

2. **Install Hugging Face Hub** (required for model downloads):
   ```bash
   # Make sure you're in the project's virtual environment
   source .venv/bin/activate  # On macOS/Linux
   # or
   .venv\Scripts\activate  # On Windows
   
   # Install huggingface-hub (includes the CLI)
   pip install huggingface-hub
   # Or if using UV:
   uv pip install huggingface-hub
   ```

3. **Authenticate with Hugging Face**:
   - Request access to SAM3 model at: https://huggingface.co/facebook/sam3
   - Once approved, authenticate (make sure virtual environment is activated):
   ```bash
   source .venv/bin/activate  # Activate virtual environment first
   hf auth login  # Note: newer versions use 'hf auth login' instead of 'huggingface-cli login'
   ```
   - If `hf` command is not found, you can also use: `python -m huggingface_hub.cli.hf auth login`

4. **Model Download**: SAM3 models will auto-download on first use, or download manually:
   ```bash
   hf download facebook/sam3
   # Or: python -m huggingface_hub.cli.hf download facebook/sam3
   ```

**Note**: 
- The code will automatically detect if SAM3 is installed or available in the local `sam3/` directory
- SMPL-X models are optional when using SAM3. The system uses SAM3 for person tracking and trajectory generation directly from segmentation masks
- If SAM3 is not available, the system will fall back to basic tracking

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

2. **Person Tracking with SAM3**: 
   - SAM3 video predictor segments person in each frame
   - Uses text prompt "person" to identify and track the athlete
   - Extracts segmentation masks for accurate person boundaries
   - Calculates center of mass from mask for trajectory tracking

3. **Trajectory Generation**: 
   - Extracts 2D center position from SAM3 segmentation mask
   - Estimates depth from mask area (larger area = closer to camera)
   - Converts 2D position to 3D trajectory coordinates
   - Tracks position, speed, and movement patterns frame-by-frame

4. **Trajectory Analysis**: 
   - **Speed Analysis**: Calculates current, average, and maximum speed
   - **Height Analysis**: Tracks vertical position and jump height
   - **Acceleration**: Monitors acceleration/deceleration patterns
   - **Smoothness**: Evaluates trajectory smoothness and control
   - **Distance**: Calculates total distance traveled

5. **Feedback Generation**: 
   - Provides real-time feedback based on trajectory metrics
   - Scores performance (0-10) based on speed, height, and smoothness
   - Identifies areas for improvement (low speed, erratic movement, etc.)
   - Highlights frames with technique mistakes

6. **Trajectory Visualization**: 
   - Overlays trajectory path on original video frames
   - Uses camera-matched projection for accurate visualization
   - Color-codes trajectory (blue = start, red = end)
   - Highlights current position in green

7. **Output Generation**: 
   - Generates main analysis video with SAM3 mask overlay and feedback
   - Creates trajectory visualization video matching camera view
   - Saves trajectory data (positions, speeds, metrics)
   - Generates comprehensive PDF report with trajectory-based analysis

## Analysis Metrics

The system analyzes trajectory-based metrics:

- **Speed**: Current, average, and maximum speed throughout movement
- **Jump Height**: Maximum vertical displacement (Z coordinate)
- **Acceleration**: Rate of speed change (positive = accelerating, negative = decelerating)
- **Trajectory Smoothness**: Consistency of movement direction (higher = smoother)
- **Total Distance**: Cumulative distance traveled in 3D space
- **Performance Score**: Overall score (0-10) based on all metrics

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
- **sam3**: Segment Anything Model 3 for person tracking and segmentation
- **huggingface-hub**: Access to SAM3 model checkpoints
- **transformers**: Required by SAM3
- **scipy**: Scientific computing (rotation transforms, trajectory analysis)
- **reportlab**: PDF report generation
- **numpy**: Numerical computing
- **matplotlib**: Plotting (for potential visualization)
- **smplx**: SMPL-X parametric body model (optional, for advanced 3D analysis)

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

### SAM3 Installation Issues

If SAM3 is not available:
1. Ensure SAM3 is installed: `git clone https://github.com/facebookresearch/sam3.git && cd sam3 && pip install -e .`
2. Install `huggingface-hub` (includes the CLI): `pip install huggingface-hub` or `uv pip install huggingface-hub`
3. **If `huggingface-cli` or `hf: command not found`**:
   - Make sure you're in the correct virtual environment: `source .venv/bin/activate`
   - Install huggingface-hub: `pip install huggingface-hub`
   - Newer versions use `hf` command instead of `huggingface-cli`
   - Verify installation: `which hf` or `which huggingface-cli`
   - If neither works, use: `python -m huggingface_hub.cli.hf login`
4. Authenticate with Hugging Face: `hf auth login` or `huggingface-cli login` (make sure virtual environment is activated)
5. Request access to SAM3 model at: https://huggingface.co/facebook/sam3
6. The system will fall back to basic tracking if SAM3 is unavailable

### SMPL-X Model Path (Optional)

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

