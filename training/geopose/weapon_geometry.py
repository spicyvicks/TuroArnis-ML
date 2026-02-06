import numpy as np

def slerp(p0, p1, t):
    """Spherical Linear Interpolation between two vectors."""
    omega = np.arccos(np.clip(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)), -1, 1))
    so = np.sin(omega)
    if so == 0:
        return (1.0 - t) * p0 + t * p1
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

def correct_weapon_geometry(grip, tip, wrist, elbow, alpha=1.35, max_deviation_deg=20.0):
    """
    Corrects the weapon tip position based on biomechanical constraints.
    
    Args:
        grip: (np.array) [x, y] coordinates of the weapon grip.
        tip: (np.array) [x, y] coordinates of the weapon tip.
        wrist: (np.array) [x, y] coordinates of the holding wrist.
        elbow: (np.array) [x, y] coordinates of the holding elbow.
        alpha: (float) Target ratio of stick length to forearm length (default 1.35 for 28" stick).
        max_deviation_deg: (float) Maximum allowed deviation from forearm vector in degrees.
        
    Returns:
        corrected_tip: (np.array) [x, y] coordinates of the corrected tip.
    """
    # 1. Forearm Vector
    forearm_vec = wrist - elbow
    forearm_len = np.linalg.norm(forearm_vec)
    
    if forearm_len == 0:
        return tip # Cannot correct if forearm length is 0
        
    forearm_dir = forearm_vec / forearm_len
    
    # 2. Stick Vector (Observed)
    stick_vec = tip - grip
    stick_len = np.linalg.norm(stick_vec)
    
    if stick_len == 0:
        # If tip == grip, assume stick is along forearm
        return grip + forearm_dir * (forearm_len * alpha)

    stick_dir = stick_vec / stick_len
    
    # 3. Angle Constraint
    # Calculate angle between forearm and stick
    dot_product = np.dot(forearm_dir, stick_dir)
    angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
    angle_deg = np.degrees(angle_rad)
    
    # If angle exceeds limit, interpolate towards forearm vector
    if angle_deg > max_deviation_deg:
        # Determine correction factor t (0 to 1)
        # We want to clamp it to max_deviation_deg
        # But for MVP, simple interpolation or clamping is specified?
        # The prompt says "Slerp ... if deviation > 20 deg, interpolate back towards 20 deg limit"
        # Since we are in 2D, we can just rotate the vector. 
        # But Slerp with the forearm_dir is a robust way to bring it closer.
        
        # Calculate t such that result is max_deviation_deg
        # This is non-trivial to solve exactly for t in one step without rotation matrices.
        # Simplification for MVP: Clamp direction to max deviation.
        
        # Cross product to determine sign (direction of deviation)
        cross_prod = forearm_dir[0] * stick_dir[1] - forearm_dir[1] * stick_dir[0]
        sign = np.sign(cross_prod)
        
        # Rotate forearm_dir by +/- max_deviation_deg
        theta_rad = np.radians(max_deviation_deg) * sign
        c, s = np.cos(theta_rad), np.sin(theta_rad)
        # Rotation matrix [[c, -s], [s, c]]
        new_dir_x = c * forearm_dir[0] - s * forearm_dir[1]
        new_dir_y = s * forearm_dir[0] + c * forearm_dir[1]
        
        corrected_dir = np.array([new_dir_x, new_dir_y])
        
    else:
        corrected_dir = stick_dir
        
    # 4. Length Constraint
    target_len = forearm_len * alpha
    
    # Final Tip Position
    corrected_tip = grip + corrected_dir * target_len
    
    return corrected_tip
