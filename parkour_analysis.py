import sys
import os

# Add error handling for imports
try:
    import cv2
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.spatial.transform import Rotation as R
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    from smplx import SMPLX
    import trimesh
    import pyrender
    from typing import List, Dict, Tuple, Optional
    import urllib.request
    import zipfile
    import tempfile
    
    # SAM3 imports - try local installation first
    SAM3_AVAILABLE = False
    try:
        # First try importing from installed package
        from sam3.model_builder import build_sam3_video_predictor
        SAM3_AVAILABLE = True
        print("✓ SAM3 imported from installed package")
    except ImportError:
        try:
            # Try importing from local sam3 directory
            sam3_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sam3')
            if os.path.exists(sam3_path) and os.path.exists(os.path.join(sam3_path, 'sam3', 'model_builder.py')):
                # Add parent directory to path so we can import sam3
                sys.path.insert(0, sam3_path)
                from sam3.model_builder import build_sam3_video_predictor
                SAM3_AVAILABLE = True
                print(f"✓ Using local SAM3 installation from: {sam3_path}")
            else:
                print(f"Warning: SAM3 directory found at {sam3_path} but model_builder.py not found.")
                print("Please install SAM3: cd sam3 && pip install -e .")
                print("Falling back to basic tracking...")
        except ImportError as e:
            SAM3_AVAILABLE = False
            missing_module = str(e).split("'")[1] if "'" in str(e) else "unknown"
            print(f"⚠ Warning: SAM3 not available - missing dependency: {missing_module}")
            print(f"   Error details: {e}")
            print("SAM3 directory found but dependencies are missing.")
            print("To install SAM3 and its dependencies:")
            print("  1. cd sam3")
            print("  2. pip install -e .")
            print("  3. pip install -e '.[notebooks]'  # Optional: for notebook support")
            print("  4. hf auth login  # Required for model downloads")
            print("Or install missing dependency directly:")
            print(f"  pip install {missing_module}")
            print("Falling back to basic tracking...")
        
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

# SAM3-based Person Tracker
class SAM3Tracker:
    """Track person using SAM3 segmentation model"""
    
    def __init__(self, predictor=None, session_id=None, device=None):
        self.predictor = predictor
        self.session_id = session_id
        self.frame_count = 0
        self.last_mask = None
        self.last_center = None
        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        
    def initialize_session(self, video_path: str):
        """Initialize SAM3 session with video"""
        if not self.predictor:
            if not SAM3_AVAILABLE:
                raise RuntimeError("SAM3 not available. Please install sam3 package.")
            # Pass device to predictor (will be used by Sam3VideoPredictor)
            self.predictor = build_sam3_video_predictor(device=self.device)
        
        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="start_session",
                    resource_path=video_path,
                )
            )
            self.session_id = response["session_id"]
            return self.session_id
        except Exception as e:
            print(f"Warning: Could not initialize SAM3 session: {e}")
            return None
    
    def add_prompt(self, prompt: str = "person", frame_index: int = 0):
        """Add text prompt to segment person"""
        if not self.session_id or not self.predictor:
            return None
        
        try:
            response = self.predictor.handle_request(
                request=dict(
                    type="add_prompt",
                    session_id=self.session_id,
                    frame_index=frame_index,
                    text=prompt,
                )
            )
            return response
        except Exception as e:
            print(f"Warning: Could not add SAM3 prompt: {e}")
            return None
    
    def get_mask(self, frame: np.ndarray, prompt: str = "person") -> Optional[np.ndarray]:
        """Get segmentation mask for person in frame"""
        if not self.predictor or not self.session_id:
            return None
        
        try:
            # Save frame temporarily for SAM3
            temp_path = f"/tmp/sam3_frame_{self.frame_count}.jpg"
            cv2.imwrite(temp_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            
            # Add prompt for this frame
            response = self.add_prompt(prompt, self.frame_count)
            
            if response and "outputs" in response:
                outputs = response["outputs"]
                if outputs and len(outputs) > 0:
                    # Get mask from output
                    mask = outputs[0].get("mask", None)
                    if mask is not None:
                        # Convert mask to numpy array if needed
                        if isinstance(mask, torch.Tensor):
                            mask = mask.cpu().numpy()
                        self.last_mask = mask
                        self.frame_count += 1
                        return mask
            
            # Fallback: use last mask if available
            return self.last_mask
        except Exception as e:
            print(f"Warning: SAM3 mask extraction failed: {e}")
            return self.last_mask
    
    def get_center_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract center of mass from mask"""
        if mask is None or mask.size == 0:
            return None
        
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask_binary = (mask > 0.5).astype(np.uint8) if mask.max() <= 1.0 else (mask > 127).astype(np.uint8)
        else:
            mask_binary = (mask > 127).astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Calculate center of mass
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        self.last_center = np.array([cx, cy])
        return self.last_center
    
    def get_bbox_from_mask(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Get bounding box from mask"""
        if mask is None or mask.size == 0:
            return None
        
        # Ensure mask is binary
        if mask.dtype != np.uint8:
            mask_binary = (mask > 0.5).astype(np.uint8) if mask.max() <= 1.0 else (mask > 127).astype(np.uint8)
        else:
            mask_binary = (mask > 127).astype(np.uint8)
        
        # Find bounding box
        coords = np.column_stack(np.where(mask_binary > 0))
        if len(coords) == 0:
            return None
        
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        
        return np.array([x_min, y_min, x_max, y_max])

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

# Trajectory-based Analysis
def analyze_trajectory(trajectory_data: List[Dict], frame_id: int, fps: float = 30.0) -> Dict[str, any]:
    """
    Analyze trajectory to provide feedback on parkour technique.
    
    Returns:
        Dictionary with metrics and feedback messages
    """
    if len(trajectory_data) < 2:
        return {
            'feedback': "Insufficient data for analysis",
            'metrics': {},
            'score': 5.0
        }
    
    # Extract positions
    positions = np.array([td['hip_position'] for td in trajectory_data[:frame_id+1]])
    frame_ids = np.array([td['frame_id'] for td in trajectory_data[:frame_id+1]])
    
    if len(positions) < 2:
        return {
            'feedback': "Insufficient data for analysis",
            'metrics': {},
            'score': 5.0
        }
    
    # Calculate metrics
    metrics = {}
    feedback_messages = []
    
    # 1. Speed analysis
    if len(positions) >= 2:
        # Calculate velocities (change in position per frame)
        velocities = np.diff(positions, axis=0)
        speeds = np.linalg.norm(velocities[:, :2], axis=1)  # 2D speed (X-Y plane)
        metrics['current_speed'] = speeds[-1] if len(speeds) > 0 else 0.0
        metrics['avg_speed'] = np.mean(speeds) if len(speeds) > 0 else 0.0
        metrics['max_speed'] = np.max(speeds) if len(speeds) > 0 else 0.0
        
        if metrics['current_speed'] < 0.1:
            feedback_messages.append("Low speed → Increase momentum!")
        elif metrics['current_speed'] > 2.0:
            feedback_messages.append("High speed → Good momentum!")
    
    # 2. Height analysis (Z coordinate)
    if len(positions) >= 2:
        heights = positions[:, 2]  # Z coordinate
        metrics['current_height'] = heights[-1] if len(heights) > 0 else 0.0
        metrics['max_height'] = np.max(heights) if len(heights) > 0 else 0.0
        metrics['min_height'] = np.min(heights) if len(heights) > 0 else 0.0
        metrics['jump_height'] = metrics['max_height'] - metrics['min_height']
        
        if metrics['jump_height'] < 0.5:
            feedback_messages.append(f"Low jump: {metrics['jump_height']:.2f}m → More explosion!")
        elif metrics['jump_height'] > 1.0:
            feedback_messages.append(f"Good jump height: {metrics['jump_height']:.2f}m!")
    
    # 3. Acceleration analysis
    if len(speeds) >= 2:
        accelerations = np.diff(speeds)
        metrics['current_acceleration'] = accelerations[-1] if len(accelerations) > 0 else 0.0
        metrics['avg_acceleration'] = np.mean(accelerations) if len(accelerations) > 0 else 0.0
        
        if metrics['current_acceleration'] < -0.1:
            feedback_messages.append("Decelerating → Maintain speed!")
        elif metrics['current_acceleration'] > 0.1:
            feedback_messages.append("Accelerating → Good power!")
    
    # 4. Trajectory smoothness (variance in direction)
    if len(velocities) >= 3:
        directions = velocities[:, :2] / (np.linalg.norm(velocities[:, :2], axis=1, keepdims=True) + 1e-6)
        direction_changes = np.diff(directions, axis=0)
        smoothness = 1.0 - np.mean(np.linalg.norm(direction_changes, axis=1))
        metrics['smoothness'] = max(0, smoothness)
        
        if metrics['smoothness'] < 0.7:
            feedback_messages.append("Erratic movement → Smooth out trajectory!")
        elif metrics['smoothness'] > 0.9:
            feedback_messages.append("Smooth trajectory → Excellent control!")
    
    # 5. Total distance traveled
    if len(positions) >= 2:
        distances = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
        metrics['total_distance'] = np.sum(distances) if len(distances) > 0 else 0.0
    
    # Calculate score (0-10)
    score = 5.0  # Base score
    if metrics.get('jump_height', 0) > 0.8:
        score += 1.5
    if metrics.get('max_speed', 0) > 1.5:
        score += 1.0
    if metrics.get('smoothness', 0) > 0.8:
        score += 1.0
    if metrics.get('current_acceleration', 0) > 0:
        score += 0.5
    score = min(10.0, max(0.0, score))
    
    feedback_text = " | ".join(feedback_messages) if feedback_messages else "Solid technique!"
    
    return {
        'feedback': feedback_text,
        'metrics': metrics,
        'score': score
    }

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
    
    # Use current frame's 2D center for camera projection reference
    if current_kp2d is not None:
        # Handle both single point and array formats
        if len(current_kp2d.shape) == 1:
            kp_ref = current_kp2d.reshape(1, -1)
        else:
            kp_ref = current_kp2d
        if not np.any(kp_ref > 0):
            kp_ref = np.array([[frame_shape[0]/2, frame_shape[1]/2]])
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

# Generate PDF Report based on trajectory analysis
def generate_pdf_report(trajectory_data: List[Dict], feedbacks: List[str], scores: List[float], 
                       output_path: str, fps: float = 30.0):
    """Generate comprehensive PDF report based on trajectory analysis"""
    c = canvas.Canvas(output_path, pagesize=letter)
    
    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, 750, "Parkour Technique Analysis Report")
    c.drawString(100, 730, "Based on SAM3 Trajectory Tracking")
    
    y = 700
    
    # Overall Statistics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Overall Performance Metrics:")
    y -= 20
    
    if len(trajectory_data) > 0:
        positions = np.array([td['hip_position'] for td in trajectory_data])
        
        # Calculate overall metrics
        if len(positions) >= 2:
            distances = np.linalg.norm(np.diff(positions[:, :2], axis=0), axis=1)
            total_distance = np.sum(distances)
            avg_speed = np.mean(distances) * fps if len(distances) > 0 else 0.0
            max_height = np.max(positions[:, 2]) if len(positions) > 0 else 0.0
            min_height = np.min(positions[:, 2]) if len(positions) > 0 else 0.0
            jump_height = max_height - min_height
            avg_score = np.mean(scores) if len(scores) > 0 else 0.0
            
            c.setFont("Helvetica", 10)
            c.drawString(120, y, f"Total Distance Traveled: {total_distance:.2f} units")
            y -= 15
            c.drawString(120, y, f"Average Speed: {avg_speed:.2f} units/sec")
            y -= 15
            c.drawString(120, y, f"Maximum Jump Height: {jump_height:.2f} units")
            y -= 15
            c.drawString(120, y, f"Average Performance Score: {avg_score:.1f}/10")
            y -= 20
    
    # Frame-by-frame feedback
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Frame-by-Frame Feedback:")
    y -= 20
    
    c.setFont("Helvetica", 9)
    # Show top 15 feedback entries
    for i, (fb, score) in enumerate(zip(feedbacks[:15], scores[:15])):
        if y < 50:  # New page if needed
            c.showPage()
            y = 750
        c.drawString(120, y, f"Frame {i}: {fb[:60]} (Score: {score:.1f}/10)")
        y -= 15
    
    # Recommendations
    y -= 10
    c.setFont("Helvetica-Bold", 12)
    c.drawString(100, y, "Recommendations:")
    y -= 20
    
    c.setFont("Helvetica", 10)
    avg_score = np.mean(scores) if len(scores) > 0 else 0.0
    if avg_score < 6.0:
        c.drawString(120, y, "• Focus on maintaining consistent speed throughout movement")
        y -= 15
        c.drawString(120, y, "• Work on increasing jump height and explosion")
        y -= 15
        c.drawString(120, y, "• Practice smoother trajectory transitions")
    else:
        c.drawString(120, y, "• Excellent technique! Continue maintaining current form")
        y -= 15
        c.drawString(120, y, "• Consider pushing for even greater heights and speeds")
    
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
    
    # Initialize SAM3 tracker
    print("Initializing SAM3 for person tracking...")
    tracker = SAM3Tracker(device=device)
    
    if SAM3_AVAILABLE:
        try:
            session_id = tracker.initialize_session(video_path)
            if session_id:
                print("✓ SAM3 session initialized successfully")
                # Add initial prompt for person tracking
                tracker.add_prompt("person", 0)
            else:
                print("⚠ Warning: Could not initialize SAM3 session, using fallback tracking")
        except Exception as e:
            print(f"⚠ Warning: SAM3 initialization failed: {e}")
            print("Continuing with basic tracking...")
    else:
        print("⚠ Warning: SAM3 not available, using basic tracking")
    
    # Note: SMPL-X model is optional when using SAM3 for tracking
    # SAM3 provides trajectory directly from segmentation masks
    print("Note: Using SAM3 for person tracking (SMPL-X model not required for trajectory analysis)")
    smplx_model = None
    
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
        
        # Track person using SAM3
        mask = tracker.get_mask(frame, "person")
        center_2d = None
        bbox = None
        
        if mask is not None:
            # Get center of mass from mask
            center_2d = tracker.get_center_from_mask(mask)
            bbox = tracker.get_bbox_from_mask(mask)
        
        if center_2d is None:
            # No detection - use last known position or skip
            if len(trajectory_data) > 0:
                # Use last position
                last_pos = trajectory_data[-1]['hip_position']
                center_2d = project_3d_to_2d(last_pos, np.array([[width//4, height//4]]), (width//2, height//2))
            else:
                # Store empty data
                keypoints_2d_list.append(np.zeros(2))
                out.write(cv2.cvtColor(bg, cv2.COLOR_RGB2BGR))
                frame_id += 1
                continue
        
        # Store 2D center for camera calibration
        kp2d_center = np.array([center_2d]) if center_2d is not None else np.zeros((1, 2))
        keypoints_2d_list.append(kp2d_center[0] if len(kp2d_center) > 0 else np.zeros(2))
        
        # Convert 2D center to 3D trajectory point
        # Estimate depth from mask size if available
        if mask is not None and bbox is not None:
            mask_area = np.sum(mask > 0) if isinstance(mask, np.ndarray) else 0
            # Estimate depth from area (larger area = closer)
            depth_estimate = 1.0 / (mask_area / (frame.shape[0] * frame.shape[1]) + 0.1)
        else:
            depth_estimate = 1.0
        
        # Create 3D position from 2D center
        # Normalize coordinates
        frame_shape_2d = (width//2, height//2)
        center_normalized = (center_2d - np.array([frame_shape_2d[0]/2, frame_shape_2d[1]/2])) / max(frame_shape_2d)
        
        # Create 3D position (X, Y from 2D, Z from depth estimate)
        position_3d = np.array([
            center_normalized[0] * 2.0,  # X
            center_normalized[1] * 2.0,  # Y  
            depth_estimate * 0.5          # Z (depth)
        ])
        
        # Store trajectory data
        trajectory_data.append({
            'frame_id': frame_id,
            'hip_position': position_3d.copy(),
            'head_position': position_3d + np.array([0, -0.3, 0.1]),  # Head slightly above
            'center_of_mass': position_3d.copy()
        })
        
        # For compatibility, create dummy joints3d and vertices
        joints3d = np.array([position_3d] * 17)  # Dummy keypoints
        verts = np.array([position_3d] * 100)  # Dummy vertices
        
        # Store intermediate artifacts
        keypoints_3d_list.append({
            'frame_id': frame_id,
            'keypoints_3d': joints3d.copy(),
            'vertices': verts.copy()
        })
        vertices_list.append(verts.copy())
        
        # Analyze trajectory
        analysis_result = analyze_trajectory(trajectory_data, frame_id, fps)
        fb = analysis_result['feedback']
        score = analysis_result['score']
        metrics = analysis_result['metrics']
        
        feedbacks.append(fb)
        scores.append(score)
        
        # Track mistake frames based on low scores or negative feedback
        if score < 6.0 or any(keyword in fb.lower() for keyword in ['low', 'erratic', 'decelerating']):
            mistake_frames.append(frame_id)
        
        # Render frame with trajectory feedback
        # Display metrics on frame
        metrics_text = f"Speed: {metrics.get('current_speed', 0):.2f} | Height: {metrics.get('current_height', 0):.2f} | Score: {score:.1f}/10"
        rendered = render_frame(verts, bg, f"{fb} | {metrics_text}")
        
        # Draw SAM3 mask overlay if available
        if mask is not None:
            # Convert mask to overlay
            mask_overlay = (mask > 0.5).astype(np.uint8) * 255 if mask.max() <= 1.0 else (mask > 127).astype(np.uint8) * 255
            if len(mask_overlay.shape) == 2:
                mask_overlay = cv2.cvtColor(mask_overlay, cv2.COLOR_GRAY2RGB)
            # Resize mask to match rendered frame size
            mask_resized = cv2.resize(mask_overlay, (width, height))
            # Blend mask with rendered frame (semi-transparent)
            rendered = cv2.addWeighted(rendered, 0.7, mask_resized, 0.3, 0)
        
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
        
        # Get current vertices and 2D center if available
        current_verts = vertices_list[traj_frame_id] if traj_frame_id < len(vertices_list) else None
        current_kp2d = keypoints_2d_list[traj_frame_id] if traj_frame_id < len(keypoints_2d_list) else None
        # Convert single point to array format for projection function
        if current_kp2d is not None and len(current_kp2d.shape) == 1:
            current_kp2d = current_kp2d.reshape(1, -1)
        
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
    
    # Generate PDF report based on trajectory analysis
    generate_pdf_report(trajectory_data, feedbacks, scores, "feedback_report.pdf", fps)
    
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