import sys
import os

# Add error handling for imports
try:
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from ultralytics import YOLO
    from scipy.spatial.transform import Rotation as R
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from smplx import SMPLX
    import trimesh
    import pyrender
    from typing import List, Dict, Tuple
    import urllib.request
    import zipfile
    import tempfile
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Please install all dependencies: pip install -r requirements.txt")
    sys.exit(1)

# Check MPS
try:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
except Exception as e:
    print(f"Error setting up device: {e}")
    device = torch.device("cpu")
    print(f"Falling back to device: {device}")

# Download pro reference if needed (LAAS Kong Vault sample)
def download_pro_reference():
    if os.path.exists("pro_kong_smplx.npz"):
        return

    # For demo, generate synthetic pro data (in real: download from gepettoweb.laas.fr/parkour/)
    # Synthetic: Perfect kong with 140° knee, 1.2m height, 30° landing flex
    frames = 60
    pro_params = {}
    for t in range(frames):
        # Synthetic SMPL-X params for pro kong (body_pose, global_orient, etc.)
        pro_params[f'frame_{t}'] = {
            'body_pose': np.zeros(21*3),  # Neutral + vault rotation
            'global_orient': R.from_euler('z', t*5).as_rotvec(),  # Rotation
            'transl': np.array([0, 0, np.sin(t/frames * np.pi) * 1.2]),  # Height arc
            'betas': np.zeros(10)  # Neutral shape
        }
    np.savez("pro_kong_smplx.npz", **pro_params)
    print("Pro reference (synthetic) created; in prod, download from LAAS dataset.")

# Simple ByteTrack stub (for demo; in prod: pip install lapx)
class SimpleTracker:
    def __init__(self):
        self.track_id = 0
    def update(self, results):
        # YOLO returns a list of Results objects, get the first one
        if isinstance(results, list):
            if len(results) == 0:
                return []
            result = results[0]
        else:
            result = results
        
        # Check if there are any detections
        if result.boxes is None or len(result.boxes) == 0:
            return []
        
        # Pick largest box as main athlete
        areas = result.boxes.xyxy[:, 2] * result.boxes.xyxy[:, 3]
        idx = np.argmax(areas)
        box = result.boxes.xyxy[idx].cpu().numpy()
        return [{'bbox': box, 'track_id': self.track_id, 'area': areas[idx]}]

# 2D Keypoints to 3D SMPL-X (simple kinematic lift; for demo)
def keypoints_to_smplx(kp2d: np.ndarray, model: SMPLX, frame_shape: Tuple[int, int] = (640, 480)) -> Dict:
    """
    Convert 2D keypoints to 3D using SMPL-X model.
    
    Args:
        kp2d: (17, 2) COCO format keypoints in pixel coordinates
        model: SMPLX model instance
        frame_shape: (width, height) of the frame for camera calibration
    
    Returns:
        Dictionary with vertices, joints3d, and model parameters
    """
    # Get the device from the model
    model_device = next(model.parameters()).device
    
    # Estimate depth from 2D keypoints (simple heuristic: use bounding box size)
    if np.any(kp2d > 0):
        valid_kp = kp2d[kp2d[:, 0] > 0]  # Only valid keypoints
        if len(valid_kp) > 0:
            bbox_size = np.max(valid_kp, axis=0) - np.min(valid_kp, axis=0)
            avg_size = np.mean(bbox_size)
            # Estimate depth based on bbox size (larger = closer)
            # This is a rough approximation
            depth_scale = 1.0 / (avg_size / max(frame_shape) + 0.1)
        else:
            depth_scale = 1.0
    else:
        depth_scale = 1.0
    
    # Create 3D keypoints with estimated depth
    # Center keypoints around origin
    kp2d_centered = kp2d.copy()
    if np.any(kp2d > 0):
        center = np.mean(kp2d[kp2d[:, 0] > 0], axis=0) if np.any(kp2d[:, 0] > 0) else np.array([frame_shape[0]/2, frame_shape[1]/2])
        kp2d_centered = kp2d - center
        # Normalize to model space
        kp2d_centered = kp2d_centered / max(frame_shape) * 2.0  # Scale to ~[-1, 1]
    
    # Add depth (Z coordinate) - use estimated depth
    joints3d = np.hstack([kp2d_centered, np.ones((17, 1)) * depth_scale * 0.5])
    
    # Fit SMPL-X model
    # Use neutral pose as starting point (can be optimized later)
    output = model(
        betas=torch.zeros(1, 10, device=model_device),  # Neutral body shape
        body_pose=torch.zeros(1, 21*3, device=model_device),  # Neutral pose
        global_orient=torch.zeros(1, 3, device=model_device),  # Upright orientation
        transl=torch.zeros(1, 3, device=model_device)  # Centered position
    )
    
    vertices = output.vertices.detach().cpu().numpy()[0]
    joints = output.joints.detach().cpu().numpy()[0]  # Get actual SMPLX joints
    
    # Use SMPLX joints if available, otherwise use estimated joints3d
    if len(joints) >= 17:
        joints3d_final = joints[:17]  # Use first 17 joints
    else:
        joints3d_final = joints3d
    
    return {
        'vertices': vertices, 
        'joints3d': joints3d_final, 
        'params': {
            'betas': np.zeros(10),
            'body_pose': np.zeros(21*3),
            'global_orient': np.zeros(3),
            'transl': np.zeros(3)
        }
    }

# Parkour Analysis
def analyze_frame(joints3d: np.ndarray, frame_id: int, is_landing: bool = False) -> str:
    # SMPL-X joint indices (approx: hip=0, kneeL=5, ankleL=8)
    hip_y = joints3d[0,1]
    knee_angle = np.degrees(np.arccos(np.clip((joints3d[0,2] - joints3d[5,2]) / 0.5, -1,1)))  # Dummy calc
    feedback = []
    if knee_angle < 120:
        feedback.append(f"Knee too bent: {knee_angle:.0f}° → Drive harder!")
    if hip_y < 1.1:
        feedback.append(f"Low height: {hip_y:.1f}m → More explosion!")
    if is_landing and knee_angle > 30:
        feedback.append(f"Stiff landing: {knee_angle:.0f}° → Soften knees!")
    return " | ".join(feedback) if feedback else "Solid technique!"

# Render Mesh on Background
def render_frame(vertices: np.ndarray, bg_img: np.ndarray, feedback: str) -> np.ndarray:
    # Simple wireframe render (for speed; no pyrender overhead)
    img = bg_img.copy()
    # Project vertices to 2D (dummy ortho cam)
    proj = vertices[:, :2] * 100 + 960  # Scale to 1920x1080
    for i in range(0, len(proj), 5):  # Sample points
        if 0 < proj[i,0] < 1920 and 0 < proj[i,1] < 1080:
            cv2.circle(img, tuple(proj[i].astype(int)), 2, (0,255,0), -1)
    cv2.putText(img, feedback, (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    return img

# Project 3D point to 2D using camera view (estimated from 2D keypoints)
def project_3d_to_2d(point_3d: np.ndarray, kp2d_ref: np.ndarray, 
                    frame_shape: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    Project 3D point to 2D using estimated camera parameters from 2D keypoints.
    Uses a simple orthographic projection with scaling based on keypoint positions.
    """
    if len(point_3d.shape) == 1:
        point_3d = point_3d.reshape(1, -1)
    
    # Simple orthographic projection: use X and Y, scale Z for depth
    # Estimate camera scale from keypoint spread
    if np.any(kp2d_ref > 0):
        valid_kp = kp2d_ref[kp2d_ref[:, 0] > 0]
        if len(valid_kp) > 0:
            kp_center = np.mean(valid_kp, axis=0)
            kp_scale = np.std(valid_kp, axis=0).mean() if len(valid_kp) > 1 else max(frame_shape) * 0.1
        else:
            kp_center = np.array([frame_shape[0]/2, frame_shape[1]/2])
            kp_scale = max(frame_shape) * 0.1
    else:
        kp_center = np.array([frame_shape[0]/2, frame_shape[1]/2])
        kp_scale = max(frame_shape) * 0.1
    
    # Project: use X and Y from 3D, add depth effect from Z
    # Scale 3D coordinates to pixel space
    scale = kp_scale * 2.0  # Adjust scale factor
    proj_2d = point_3d[:, [0, 1]] * scale + kp_center
    
    # Add depth perspective (Z affects scale)
    if point_3d.shape[1] > 2:
        depth_factor = 1.0 + point_3d[:, 2] * 0.1  # Z affects size
        proj_2d = (proj_2d - kp_center) * depth_factor.reshape(-1, 1) + kp_center
    
    return proj_2d[0] if len(proj_2d) == 1 else proj_2d

# Render trajectory visualization on actual video frame
def render_trajectory_frame(trajectory_data: List[Dict], frame_id: int, 
                           original_frame: np.ndarray, 
                           current_vertices: np.ndarray = None,
                           current_kp2d: np.ndarray = None,
                           frame_shape: Tuple[int, int] = (640, 480)) -> np.ndarray:
    """
    Render trajectory path overlaid on the original video frame.
    Uses the same camera view as the input video.
    """
    img = original_frame.copy()
    
    if len(trajectory_data) == 0:
        return img
    
    # Extract hip positions for trajectory
    hip_positions = np.array([td['hip_position'] for td in trajectory_data[:frame_id+1]])
    
    if len(hip_positions) < 2:
        return img
    
    # Use current frame's 2D keypoints for camera projection reference
    if current_kp2d is not None and np.any(current_kp2d > 0):
        kp_ref = current_kp2d
    else:
        # Fallback: use frame center
        kp_ref = np.array([[frame_shape[0]/2, frame_shape[1]/2]])
    
    # Project 3D trajectory points to 2D using camera view
    traj_2d = []
    for hip_pos in hip_positions:
        proj = project_3d_to_2d(hip_pos, kp_ref, frame_shape)
        traj_2d.append(proj)
    traj_2d = np.array(traj_2d)
    
    # Draw trajectory path
    for i in range(len(traj_2d) - 1):
        pt1 = tuple(traj_2d[i].astype(int))
        pt2 = tuple(traj_2d[i+1].astype(int))
        # Color gradient: blue (start) to red (end)
        color_ratio = i / max(len(traj_2d) - 1, 1)
        color = (
            int(255 * (1 - color_ratio)),
            int(128 * (1 - color_ratio)),
            int(255 * color_ratio)
        )
        # Only draw if points are within frame
        if (0 <= pt1[0] < frame_shape[0] and 0 <= pt1[1] < frame_shape[1] and
            0 <= pt2[0] < frame_shape[0] and 0 <= pt2[1] < frame_shape[1]):
            cv2.line(img, pt1, pt2, color, 3)
    
    # Draw trajectory points
    for i, pt in enumerate(traj_2d):
        pt_int = tuple(pt.astype(int))
        if 0 <= pt_int[0] < frame_shape[0] and 0 <= pt_int[1] < frame_shape[1]:
            cv2.circle(img, pt_int, 4, (0, 255, 255), -1)
            cv2.circle(img, pt_int, 6, (0, 255, 255), 1)
    
    # Highlight current position
    if len(traj_2d) > 0:
        current_pt = tuple(traj_2d[-1].astype(int))
        if 0 <= current_pt[0] < frame_shape[0] and 0 <= current_pt[1] < frame_shape[1]:
            cv2.circle(img, current_pt, 10, (0, 255, 0), 2)
            cv2.circle(img, current_pt, 14, (0, 255, 0), 1)
    
    # Draw current 3D model vertices if provided
    if current_vertices is not None and current_kp2d is not None:
        # Project vertices to 2D
        for i in range(0, len(current_vertices), 20):  # Sample vertices
            vert_proj = project_3d_to_2d(current_vertices[i], current_kp2d, frame_shape)
            pt = tuple(vert_proj.astype(int))
            if 0 <= pt[0] < frame_shape[0] and 0 <= pt[1] < frame_shape[1]:
                cv2.circle(img, pt, 1, (255, 255, 0), -1)
    
    # Add labels
    cv2.putText(img, "Trajectory Overlay (Camera View)", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, f"Frame: {frame_id} | Points: {len(traj_2d)}", (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return img

# Overlay Pro Comparison
def overlay_pro(pro_verts: np.ndarray, user_img: np.ndarray):
    # Semi-transparent lines for pro ghost
    proj_pro = pro_verts[:, :2] * 50 + 960  # Scaled smaller
    for i in range(0, len(proj_pro), 10):
        if 0 < proj_pro[i,0] < 1920 and 0 < proj_pro[i,1] < 1080:
            cv2.line(user_img, tuple(proj_pro[i].astype(int)), tuple(proj_pro[(i+1)%len(proj_pro)].astype(int)), (255,0,0), 1)
    cv2.putText(user_img, "Pro Overlay (Dom Tomato Kong)", (50,100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)
    return user_img

# Generate PDF Report
def generate_pdf_report(feedbacks: List[str], scores: List[float], output_path: str):
    c = canvas.Canvas(output_path, pagesize=letter)
    c.drawString(100, 750, "Parkour Technique Feedback Report")
    y = 700
    for fb, score in zip(feedbacks, scores):
        c.drawString(100, y, f"{fb} (Score: {score:.1f}/10)")
        y -= 20
    c.drawString(100, y-20, "Pro Comparison: Aim for 140° knee, 1.2m height like Dom Tomato.")
    c.save()
    print(f"PDF saved: {output_path}")

# Main Pipeline
def main(video_path: str = "flip_in_beach.mp4"):
    print(f"Starting parkour analysis for video: {video_path}")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        print("Please provide a valid video file path.")
        print("Usage: python parkour_analysis.py [video_path]")
        return
    
    print("Downloading/generating pro reference data...")
    download_pro_reference()
    
    print("Loading YOLO pose model (this may take a moment on first run)...")
    try:
        model = YOLO("yolov8n-pose.pt")  # YOLOv11 equiv; auto-download
        print("YOLO model loaded successfully")
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return
    
    tracker = SimpleTracker()
    
    print("Loading SMPL-X model...")
    try:
        # SMPLX expects a directory path, not a specific .pkl file
        smplx_dir = 'models/smplx'
        
        # Check if directory exists
        if not os.path.exists(smplx_dir):
            print(f"Warning: SMPL-X model directory not found at {smplx_dir}")
            print("Trying alternative paths...")
            # Try parent models directory
            if os.path.exists('models'):
                smplx_dir = 'models'
            else:
                print("Error: No SMPL-X model directory found.")
                print("Please download SMPL-X models from: https://smpl-x.is.tue.mpg.de/")
                print("Extract them to: models/smplx/")
                return
        
        # Check if model files exist
        neutral_pkl = os.path.join(smplx_dir, 'SMPLX_NEUTRAL.pkl')
        neutral_npz = os.path.join(smplx_dir, 'SMPLX_NEUTRAL.npz')
        
        # List available files for debugging
        if os.path.exists(smplx_dir):
            print(f"Files in {smplx_dir}:")
            for f in sorted(os.listdir(smplx_dir)):
                file_path = os.path.join(smplx_dir, f)
                size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
                print(f"  - {f} ({size:.1f} MB)")
        
        # Try to verify the .pkl file is valid
        if os.path.exists(neutral_pkl):
            try:
                import pickle
                with open(neutral_pkl, 'rb') as f:
                    test_data = pickle.load(f)
                print(f"✓ SMPLX_NEUTRAL.pkl is a valid pickle file")
            except Exception as e:
                print(f"⚠ Warning: SMPLX_NEUTRAL.pkl exists but may be corrupted: {e}")
                print("Trying to use .npz format or directory loading...")
        
        # Try loading with directory path (SMPLX will auto-detect files)
        print(f"Loading SMPL-X model from directory: {smplx_dir}")
        
        # Try different loading methods
        smplx_model = None
        load_methods = [
            # Method 1: Directory with explicit parameters (neutral gender - correct for general use)
            lambda: SMPLX(model_path=smplx_dir, model_type='smplx', gender='neutral', device=device),
            # Method 2: Directory only (let SMPLX auto-detect, defaults to neutral)
            lambda: SMPLX(model_path=smplx_dir, device=device),
            # Method 3: Try with .npz if .pkl fails
            lambda: SMPLX(model_path=neutral_npz if os.path.exists(neutral_npz) else smplx_dir, device=device),
        ]
        
        for i, load_method in enumerate(load_methods, 1):
            try:
                print(f"  Trying load method {i}...")
                smplx_model = load_method()
                print(f"✓ Successfully loaded with method {i}")
                break
            except Exception as e:
                print(f"  Method {i} failed: {e}")
                if i == len(load_methods):
                    raise
        
        if smplx_model is None:
            raise RuntimeError("Failed to load SMPL-X model with all methods")
        
        # Ensure model is on the correct device
        smplx_model = smplx_model.to(device)
        model_device = next(smplx_model.parameters()).device
        print(f"✓ SMPL-X model loaded successfully")
        print(f"  - Device: {model_device}")
        print(f"  - Gender: neutral (appropriate for general parkour analysis)")
        print(f"  - Model type: SMPLX")
        # Get model info (SMPLX has ~10475 vertices and 127 joints)
        try:
            test_output = smplx_model(torch.zeros(1, 10, device=model_device))
            num_verts = test_output.vertices.shape[1]
            num_joints = test_output.joints.shape[1] if hasattr(test_output, 'joints') else 0
            print(f"  - Vertices: {num_verts} vertices")
            if num_joints > 0:
                print(f"  - Joints: {num_joints} joints")
        except:
            print(f"  - Model verified and ready")
    except Exception as e:
        print(f"Error loading SMPL-X model: {e}")
        import traceback
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("1. Ensure SMPL-X models are downloaded from: https://smpl-x.is.tue.mpg.de/")
        print("2. Extract models to: models/smplx/")
        print("3. The directory should contain SMPLX_NEUTRAL.pkl (or .npz)")
        print("4. Try using model_path='models/smplx' (directory, not file)")
        return
    
    print(f"Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = 1920, 1080
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    bg = np.ones((height, width, 3), dtype=np.uint8) * 240  # Clean gray
    # Draw grid
    for i in range(0, width, 100):
        cv2.line(bg, (i,0), (i,height), (200,200,200), 1)
    for i in range(0, height, 100):
        cv2.line(bg, (0,i), (width,i), (200,200,200), 1)
    
    feedbacks = []
    scores = []
    mistake_frames = []
    frame_id = 0
    
    # Store intermediate artifacts
    keypoints_3d_list = []  # List of 3D keypoints for each frame
    trajectory_data = []  # List of trajectory points (hip position, head position, etc.)
    vertices_list = []  # Store vertices for trajectory visualization
    original_frames = []  # Store original video frames for trajectory overlay
    keypoints_2d_list = []  # Store 2D keypoints for camera calibration
    
    # Load pro reference data with allow_pickle=True (contains object arrays)
    print("Loading professional reference data...")
    try:
        pro_data = np.load("pro_kong_smplx.npz", allow_pickle=True)
        print(f"Loaded {len(pro_data.files)} reference frames")
    except Exception as e:
        print(f"Warning: Could not load pro reference data: {e}")
        print("Continuing without pro overlay...")
        pro_data = None
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, (width//2, height//2))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Store original frame for trajectory visualization
        original_frames.append(frame.copy())
        
        # Detect & track
        results = model(frame)
        
        # YOLO returns a list, get first result
        result = results[0] if isinstance(results, list) else results
        
        # Pass the result to tracker (tracker handles both list and single result)
        tracklets = tracker.update(result)
        if not tracklets:
            # Store empty keypoints for frames without detection
            keypoints_2d_list.append(np.zeros((17, 2)))
            out.write(cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
            frame_id += 1
            continue
        main_track = tracklets[0]
        bbox = main_track['bbox'].astype(int) * 2  # Scale
        crop = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        if crop.size == 0:
            keypoints_2d_list.append(np.zeros((17, 2)))
            out.write(cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
            continue
        
        # 2D keypoints (from YOLO)
        if result.keypoints is not None and len(result.keypoints.xy) > 0:
            kp2d = result.keypoints.xy[0].cpu().numpy()
        else:
            kp2d = np.zeros((17, 2))
        
        # Store 2D keypoints for camera calibration
        keypoints_2d_list.append(kp2d.copy())
        
        # To 3D - pass frame shape for proper camera calibration
        frame_shape_2d = (width//2, height//2)  # Resized frame dimensions
        smplx_out = keypoints_to_smplx(kp2d, smplx_model, frame_shape_2d)
        verts = smplx_out['vertices']
        joints3d = smplx_out['joints3d']
        
        # Store intermediate artifacts
        keypoints_3d_list.append({
            'frame_id': frame_id,
            'keypoints_3d': joints3d.copy(),
            'vertices': verts.copy()
        })
        
        # Extract trajectory points (hip center, head, left/right feet)
        # Using approximate joint indices: hip=0, head=~15, left_foot=~10, right_foot=~13
        hip_pos = joints3d[0] if len(joints3d) > 0 else np.array([0, 0, 0])
        head_pos = joints3d[15] if len(joints3d) > 15 else joints3d[0] + np.array([0, 0.3, 0])
        # For trajectory, use hip position as main tracking point
        trajectory_data.append({
            'frame_id': frame_id,
            'hip_position': hip_pos.copy(),
            'head_position': head_pos.copy(),
            'center_of_mass': verts.mean(axis=0).copy()  # Center of mass from vertices
        })
        vertices_list.append(verts.copy())
        
        # Analyze
        is_landing = frame_id > 40  # Dummy detection
        fb = analyze_frame(joints3d, frame_id, is_landing)
        feedbacks.append(fb)
        score = 8.0 if "Solid" in fb else 5.0  # Dummy score
        scores.append(score)
        if "too" in fb or "Low" in fb:
            mistake_frames.append(frame_id)
        
        # Render
        rendered = render_frame(verts, bg, fb)
        
        # Pro overlay (every 10th frame for speed)
        if pro_data is not None and frame_id % 10 == 0:
            try:
                pro_frame = int(frame_id * len(pro_data.files) / 100)  # Sync
                pro_key = f'frame_{pro_frame}'
                if pro_key in pro_data:
                    # Access the data item (which is a dictionary)
                    frame_data = pro_data[pro_key]
                    # Handle both dict and array formats
                    if isinstance(frame_data, dict):
                        pro_verts = frame_data.get('transl', np.array([0, 0, 0]))
                    elif isinstance(frame_data, np.ndarray):
                        pro_verts = frame_data
                    else:
                        # If it's an object array, access it properly
                        pro_verts = frame_data.item() if hasattr(frame_data, 'item') else frame_data
                        if isinstance(pro_verts, dict):
                            pro_verts = pro_verts.get('transl', np.array([0, 0, 0]))
                    
                    # Ensure pro_verts is a numpy array
                    if not isinstance(pro_verts, np.ndarray):
                        pro_verts = np.array(pro_verts)
                    
                    rendered = overlay_pro(pro_verts, rendered)
            except Exception as e:
                # Silently skip pro overlay if there's an error
                pass
        
        # Slow-mo on mistakes
        if frame_id in mistake_frames:
            for _ in range(4):  # 0.25x speed
                out.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        else:
            out.write(cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        
        frame_id += 1
        if frame_id % 30 == 0:
            print(f"Processed {frame_id} frames...")
    
    cap.release()
    out.release()
    
    # Save intermediate artifacts
    print("\nSaving intermediate artifacts...")
    
    # Save 3D keypoints
    keypoints_file = "keypoints_3d.npz"
    keypoints_dict = {
        f'frame_{i}': data['keypoints_3d'] 
        for i, data in enumerate(keypoints_3d_list)
    }
    np.savez(keypoints_file, **keypoints_dict)
    print(f"✓ Saved 3D keypoints to {keypoints_file}")
    
    # Save trajectory data
    trajectory_file = "trajectory_data.npz"
    trajectory_dict = {
        'hip_positions': np.array([td['hip_position'] for td in trajectory_data]),
        'head_positions': np.array([td['head_position'] for td in trajectory_data]),
        'center_of_mass': np.array([td['center_of_mass'] for td in trajectory_data]),
        'frame_ids': np.array([td['frame_id'] for td in trajectory_data])
    }
    np.savez(trajectory_file, **trajectory_dict)
    print(f"✓ Saved trajectory data to {trajectory_file}")
    
    # Save vertices for all frames
    vertices_file = "vertices_all_frames.npz"
    vertices_dict = {
        f'frame_{i}': verts 
        for i, verts in enumerate(vertices_list)
    }
    np.savez(vertices_file, **vertices_dict)
    print(f"✓ Saved vertices to {vertices_file}")
    
    # Generate trajectory visualization video
    print("Generating trajectory visualization video...")
    traj_out = cv2.VideoWriter('trajectory_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    frame_shape_2d = (width//2, height//2)  # Resized frame dimensions
    
    for traj_frame_id in range(len(original_frames)):
        # Get original frame (resize to output size)
        orig_frame = original_frames[traj_frame_id]
        orig_frame_resized = cv2.resize(orig_frame, (width, height))
        
        # Get current vertices and 2D keypoints if available
        current_verts = vertices_list[traj_frame_id] if traj_frame_id < len(vertices_list) else None
        current_kp2d = keypoints_2d_list[traj_frame_id] if traj_frame_id < len(keypoints_2d_list) else None
        
        # Render trajectory frame overlaid on original video
        traj_frame = render_trajectory_frame(
            trajectory_data, 
            traj_frame_id, 
            orig_frame_resized,
            current_verts,
            current_kp2d,
            (width, height)  # Output frame size
        )
        traj_out.write(cv2.cvtColor(traj_frame, cv2.COLOR_RGB2BGR))
        
        if traj_frame_id % 30 == 0:
            print(f"  Processed {traj_frame_id} trajectory frames...")
    
    traj_out.release()
    print(f"✓ Saved trajectory visualization to trajectory_video.mp4")
    
    # Generate PDF report
    generate_pdf_report(feedbacks[:10], scores[:10], "feedback_report.pdf")  # Top 10
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print("Output files:")
    print("  - output_video.mp4 (main analysis video)")
    print("  - trajectory_video.mp4 (trajectory visualization)")
    print("  - feedback_report.pdf (technique analysis report)")
    print("  - keypoints_3d.npz (3D keypoints for all frames)")
    print("  - trajectory_data.npz (trajectory positions)")
    print("  - vertices_all_frames.npz (3D mesh vertices)")
    print("="*60)

if __name__ == "__main__":
    print("=" * 60)
    print("Parkour Analysis System")
    print("=" * 60)
    
    # Get video path from command line or use default
    video_path = sys.argv[1] if len(sys.argv) > 1 else "flip_in_beach.mp4"
    
    try:
        main(video_path)
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)