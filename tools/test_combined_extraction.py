"""
Test script to visualize MediaPipe body landmarks + YOLO stick detector
Run this to see if combined feature extraction works before full implementation
"""
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import sys
import argparse

# paths
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
stick_model_path = os.path.join(project_root, 'runs', 'pose', 'arnis_stick_detector', 'weights', 'best.pt')

def debug_print(msg, level="INFO"):
    """print debug messages with level prefix"""
    print(f"[{level}] {msg}")

def get_screen_size():
    """get screen resolution"""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except:
        return 1920, 1080  # fallback

def resize_to_screen(image, max_width_ratio=0.9, max_height_ratio=0.85):
    """resize image to fit screen while maintaining aspect ratio"""
    screen_w, screen_h = get_screen_size()
    max_w = int(screen_w * max_width_ratio)
    max_h = int(screen_h * max_height_ratio)
    
    h, w = image.shape[:2]
    
    # calculate scaling factor
    scale_w = max_w / w
    scale_h = max_h / h
    scale = min(scale_w, scale_h)
    
    if scale < 1:  # only resize if image is larger than screen
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA), scale
    return image, 1.0

def test_combined_extraction(image_path, verbose=True):
    """run both mediapipe and stick detector on an image"""
    
    debug_print(f"Loading image: {image_path}")
    
    # load image
    image = cv2.imread(image_path)
    if image is None:
        debug_print(f"Could not load image: {image_path}", "ERROR")
        return
    
    h, w = image.shape[:2]
    debug_print(f"Original image size: {w}x{h}")
    
    # get screen size for reference
    screen_w, screen_h = get_screen_size()
    debug_print(f"Screen resolution: {screen_w}x{screen_h}")
    
    # === MEDIAPIPE BODY POSE ===
    print("\n" + "="*60)
    print("  MEDIAPIPE BODY LANDMARKS")
    print("="*60)
    
    mp_pose = mp.solutions.pose
    mp_draw = mp.solutions.drawing_utils
    mp_draw_styles = mp.solutions.drawing_styles
    
    debug_print("Initializing MediaPipe Pose (complexity=2)...")
    pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, min_detection_confidence=0.5)
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    debug_print("Running pose detection...")
    results = pose.process(image_rgb)
    
    output_image = image.copy()
    
    body_detected = False
    left_wrist_px, right_wrist_px = None, None
    
    if results.pose_landmarks:
        body_detected = True
        debug_print("Body pose DETECTED!", "OK")
        
        # draw with custom style for better visibility
        mp_draw.draw_landmarks(
            output_image, 
            results.pose_landmarks, 
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=3),
            connection_drawing_spec=mp_draw.DrawingSpec(color=(255,255,255), thickness=2)
        )
        
        # extract key body features (world coordinates)
        lm = results.pose_world_landmarks.landmark
        
        if verbose:
            print("\n  World Coordinates (hip-centered, meters):")
            print(f"    Left shoulder:  ({lm[11].x:+.4f}, {lm[11].y:+.4f}, {lm[11].z:+.4f})")
            print(f"    Right shoulder: ({lm[12].x:+.4f}, {lm[12].y:+.4f}, {lm[12].z:+.4f})")
            print(f"    Left elbow:     ({lm[13].x:+.4f}, {lm[13].y:+.4f}, {lm[13].z:+.4f})")
            print(f"    Right elbow:    ({lm[14].x:+.4f}, {lm[14].y:+.4f}, {lm[14].z:+.4f})")
            print(f"    Left wrist:     ({lm[15].x:+.4f}, {lm[15].y:+.4f}, {lm[15].z:+.4f})")
            print(f"    Right wrist:    ({lm[16].x:+.4f}, {lm[16].y:+.4f}, {lm[16].z:+.4f})")
            print(f"    Left hip:       ({lm[23].x:+.4f}, {lm[23].y:+.4f}, {lm[23].z:+.4f})")
            print(f"    Right hip:      ({lm[24].x:+.4f}, {lm[24].y:+.4f}, {lm[24].z:+.4f})")
        
        # get pixel coordinates
        lm_px = results.pose_landmarks.landmark
        left_wrist_px = (int(lm_px[15].x * w), int(lm_px[15].y * h))
        right_wrist_px = (int(lm_px[16].x * w), int(lm_px[16].y * h))
        
        if verbose:
            print("\n  Pixel Coordinates:")
            print(f"    Left wrist:  {left_wrist_px}")
            print(f"    Right wrist: {right_wrist_px}")
        
        # draw labels on wrists
        cv2.putText(output_image, "L", (left_wrist_px[0]-20, left_wrist_px[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(output_image, "R", (right_wrist_px[0]+10, right_wrist_px[1]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
    else:
        debug_print("No body pose detected!", "WARN")
    
    pose.close()
    debug_print("MediaPipe closed.")
    
    # === YOLO STICK DETECTOR ===
    print("\n" + "="*60)
    print("  YOLO STICK DETECTOR")
    print("="*60)
    
    if not os.path.exists(stick_model_path):
        debug_print(f"Stick model not found: {stick_model_path}", "ERROR")
        return
    
    debug_print(f"Loading stick model: {stick_model_path}")
    stick_model = YOLO(stick_model_path)
    
    debug_print("Running stick detection...")
    stick_results = stick_model(image, verbose=False)
    
    stick_detected = False
    grip_point = None
    tip_point = None
    stick_conf = 0
    
    for result in stick_results:
        if result.keypoints is not None and len(result.boxes) > 0:
            stick_detected = True
            n_sticks = len(result.boxes)
            debug_print(f"Detected {n_sticks} stick(s)!", "OK")
            
            for i, (box, kpts) in enumerate(zip(result.boxes, result.keypoints)):
                stick_conf = box.conf.item()
                bbox = box.xyxy[0].cpu().numpy()
                
                if verbose:
                    print(f"\n  Stick {i+1}:")
                    print(f"    Confidence: {stick_conf:.3f}")
                    print(f"    BBox: ({bbox[0]:.0f}, {bbox[1]:.0f}) -> ({bbox[2]:.0f}, {bbox[3]:.0f})")
                    print(f"    BBox size: {bbox[2]-bbox[0]:.0f}x{bbox[3]-bbox[1]:.0f} px")
                
                # draw bounding box with thicker line
                cv2.rectangle(output_image, 
                             (int(bbox[0]), int(bbox[1])), 
                             (int(bbox[2]), int(bbox[3])), 
                             (0, 255, 0), 3)
                
                # keypoints
                kpts_data = kpts.data[0].cpu().numpy()
                
                if verbose:
                    print(f"    Keypoints ({len(kpts_data)}):")
                
                for j, kpt in enumerate(kpts_data):
                    kpt_x, kpt_y, kpt_conf = kpt[0], kpt[1], kpt[2]
                    kpt_name = "GRIP" if j == 0 else "TIP" if j == 1 else f"PT{j}"
                    
                    if verbose:
                        print(f"      {kpt_name}: ({kpt_x:.1f}, {kpt_y:.1f}) conf={kpt_conf:.3f}")
                    
                    # draw keypoint with larger circles
                    if kpt_conf > 0.3:
                        color = (255, 100, 0) if j == 0 else (0, 100, 255)  # blue=grip, red=tip
                        cv2.circle(output_image, (int(kpt_x), int(kpt_y)), 12, color, -1)
                        cv2.circle(output_image, (int(kpt_x), int(kpt_y)), 14, (255,255,255), 2)
                        cv2.putText(output_image, kpt_name, (int(kpt_x)+15, int(kpt_y)+5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        
                        if j == 0:
                            grip_point = (kpt_x, kpt_y, kpt_conf)
                        elif j == 1:
                            tip_point = (kpt_x, kpt_y, kpt_conf)
                
                # draw line from grip to tip
                if grip_point and tip_point:
                    cv2.line(output_image, 
                            (int(grip_point[0]), int(grip_point[1])),
                            (int(tip_point[0]), int(tip_point[1])),
                            (0, 255, 255), 4)
        else:
            debug_print("No stick detected in this result!", "WARN")
    
    if not stick_detected:
        debug_print("No stick detected in image!", "WARN")
    
    # === COMBINED FEATURE EXTRACTION ===
    print("\n" + "="*60)
    print("  COMBINED FEATURES (PROPOSED)")
    print("="*60)
    
    combined_features = {}
    
    if body_detected and stick_detected and grip_point and tip_point:
        debug_print("Both body and stick detected - extracting combined features!", "OK")
        
        # stick length (pixels)
        stick_length = np.sqrt((tip_point[0] - grip_point[0])**2 + (tip_point[1] - grip_point[1])**2)
        combined_features['stick_length_px'] = stick_length
        
        # stick angle (degrees from horizontal, -180 to +180)
        stick_angle = np.degrees(np.arctan2(tip_point[1] - grip_point[1], tip_point[0] - grip_point[0]))
        combined_features['stick_angle_deg'] = stick_angle
        
        # normalize stick to [0,1] coordinates
        grip_norm = (grip_point[0] / w, grip_point[1] / h)
        tip_norm = (tip_point[0] / w, tip_point[1] / h)
        combined_features['grip_x_norm'] = grip_norm[0]
        combined_features['grip_y_norm'] = grip_norm[1]
        combined_features['tip_x_norm'] = tip_norm[0]
        combined_features['tip_y_norm'] = tip_norm[1]
        
        # distance from grip to each wrist
        if left_wrist_px and right_wrist_px:
            dist_to_left = np.sqrt((grip_point[0] - left_wrist_px[0])**2 + (grip_point[1] - left_wrist_px[1])**2)
            dist_to_right = np.sqrt((grip_point[0] - right_wrist_px[0])**2 + (grip_point[1] - right_wrist_px[1])**2)
            combined_features['grip_to_left_wrist_px'] = dist_to_left
            combined_features['grip_to_right_wrist_px'] = dist_to_right
            combined_features['holding_hand'] = 0 if dist_to_left < dist_to_right else 1  # 0=left, 1=right
        
        # tip height relative to shoulders (normalized)
        lm_px = results.pose_landmarks.landmark
        shoulder_y = (lm_px[11].y + lm_px[12].y) / 2
        combined_features['tip_vs_shoulder_y'] = tip_norm[1] - shoulder_y
        
        # tip position relative to hip center
        hip_center_x = (lm_px[23].x + lm_px[24].x) / 2
        hip_center_y = (lm_px[23].y + lm_px[24].y) / 2
        combined_features['tip_vs_hip_x'] = tip_norm[0] - hip_center_x
        combined_features['tip_vs_hip_y'] = tip_norm[1] - hip_center_y
        
        # confidence values
        combined_features['stick_conf'] = stick_conf
        combined_features['grip_conf'] = grip_point[2]
        combined_features['tip_conf'] = tip_point[2]
        
        print("\n  Extracted Features:")
        for k, v in combined_features.items():
            if isinstance(v, float):
                print(f"    {k}: {v:.4f}")
            else:
                print(f"    {k}: {v}")
        
        print(f"\n  Total new stick features: {len(combined_features)}")
        print(f"  Combined with 58 body features = {58 + len(combined_features)} total features")
        
    elif body_detected and not stick_detected:
        debug_print("Body detected but NO STICK - would use fallback (zero) features", "WARN")
    elif stick_detected and not body_detected:
        debug_print("Stick detected but NO BODY - cannot compute relative features", "WARN")
    else:
        debug_print("Neither body nor stick detected!", "ERROR")
    
    # === DISPLAY AND SAVE ===
    print("\n" + "="*60)
    print("  OUTPUT")
    print("="*60)
    
    # save full resolution
    output_path = os.path.join(project_root, 'tools', 'combined_test_output.jpg')
    cv2.imwrite(output_path, output_image)
    debug_print(f"Saved full-res output: {output_path}")
    
    # resize for display
    display_image, scale = resize_to_screen(output_image)
    new_h, new_w = display_image.shape[:2]
    debug_print(f"Display size: {new_w}x{new_h} (scale: {scale:.2f})")
    
    # add info overlay
    info_text = [
        f"Image: {os.path.basename(image_path)}",
        f"Original: {w}x{h} | Display: {new_w}x{new_h}",
        f"Body: {'YES' if body_detected else 'NO'} | Stick: {'YES' if stick_detected else 'NO'}",
    ]
    if stick_detected:
        info_text.append(f"Stick angle: {combined_features.get('stick_angle_deg', 0):.1f} deg")
    
    y_offset = 30
    for text in info_text:
        cv2.putText(display_image, text, (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
        y_offset += 25
    
    # show window
    window_name = "Combined Detection Test (Press any key to close)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, new_w, new_h)
    cv2.imshow(window_name, display_image)
    
    debug_print("Displaying window... Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    debug_print("Done!", "OK")
    return combined_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test combined MediaPipe + YOLO stick detection")
    parser.add_argument("image", nargs="?", help="Path to test image")
    parser.add_argument("-q", "--quiet", action="store_true", help="Reduce output verbosity")
    parser.add_argument("-l", "--list", action="store_true", help="List sample images from dataset")
    args = parser.parse_args()
    
    # default test image
    default_image = os.path.join(project_root, 'dataset', 'right_temple_block_correct', '5.jpg')
    
    if args.list:
        print("\nSample images from dataset:")
        dataset_dir = os.path.join(project_root, 'dataset')
        for cls in sorted(os.listdir(dataset_dir))[:5]:
            cls_path = os.path.join(dataset_dir, cls)
            if os.path.isdir(cls_path):
                images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg','.png'))][:2]
                for img in images:
                    print(f"  {cls}/{img}")
        sys.exit(0)
    
    test_image = args.image if args.image else default_image
    
    if not os.path.exists(test_image):
        print(f"[ERROR] Image not found: {test_image}")
        print(f"\nUsage: python test_combined_extraction.py [image_path]")
        print(f"       python test_combined_extraction.py -l  # list sample images")
        sys.exit(1)
    
    print("\n" + "="*60)
    print("  COMBINED FEATURE EXTRACTION TEST")
    print("="*60)
    debug_print(f"Test image: {test_image}")
    
    test_combined_extraction(test_image, verbose=not args.quiet)
