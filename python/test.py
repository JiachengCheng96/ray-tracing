import trimesh
import numpy as np
import io
import requests
import open3d as o3d


fov_deg = 90
resolution_x = 5000
resolution_y = 5000
target_voxel = np.array([-0.12, 0.12, 0.2])
voxel_size = 0.02

# ==========================================
# 1. CORE ALGORITHM: ID BUFFER
# ==========================================
def find_visible_triangles(mesh, camera_pose, resolution=(40000, 40000), return_ray_vis=False, target_point=None, voxel_size=0.2):
    """
    Uses Open3D Raycasting to find visible triangles.
    """
    # 1. Convert Trimesh to Open3D Tensor Mesh
    # Open3D Raycasting expects float32 vertices
    vertices = np.array(mesh.vertices, dtype=np.float32)
    triangles = np.array(mesh.faces, dtype=np.uint32)
    
    t_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices),
        o3d.core.Tensor(triangles)
    )
    t_mesh.compute_vertex_normals()
    
    # 2. Setup Scene
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)
    
    # 3. Setup Camera Rays
    width, height = resolution
    
    # --- LIDAR / RADAR SETUP (Spherical Projection) ---
    # Elevation: -FOV/2 to +FOV/2
    fov_rad = np.deg2rad(fov_deg)
    
    # Azimuth: -FOV/2 to +FOV/2
    azimuths = np.linspace(-fov_rad/2, fov_rad/2, width)
    elevations = np.linspace(-fov_rad/2, fov_rad/2, height)
    
    # Create grid
    az_grid, el_grid = np.meshgrid(azimuths, elevations)
    az = az_grid.flatten()
    el = el_grid.flatten()
    
    # Directions in Sensor Frame (assuming -Z forward to match camera look direction)
    # x = sin(az) * cos(el)
    # y = sin(el)
    # z = -cos(az) * cos(el)
    dirs_local = np.stack([
        np.cos(el) * np.sin(az),
        np.sin(el),
        -np.cos(el) * np.cos(az)
    ], axis=1)
    
    # Transform to World Frame
    R = camera_pose[:3, :3]
    t = camera_pose[:3, 3]
    
    dirs_world = dirs_local @ R.T
    origins = np.tile(t, (dirs_world.shape[0], 1))
    
    rays_np = np.concatenate([origins, dirs_world], axis=1).astype(np.float32)
    rays = o3d.core.Tensor(rays_np)
    
    # 4. Cast Rays
    ans = scene.cast_rays(rays)
    
    # 5. Extract Visible IDs
    # primitive_ids is a tensor of shape [height, width]
    ids = ans['primitive_ids'].numpy()
    
    # Filter out invalid hits
    valid_mask = ids != scene.INVALID_ID
    visible_ids = np.unique(ids[valid_mask])
    
    vis_rays = None
    vis_reflected = None
    if return_ray_vis:
        # Create visualization for a subset of rays
        t_hit = ans['t_hit'].numpy().flatten()
        rays_np = rays.numpy()
        
        # --- INTELLIGENT SAMPLING ---
        # Instead of random sampling first, we check ALL hits to find the ones hitting the voxel.
        
        # 1. Identify all valid hits
        hit_mask_all = np.isfinite(t_hit)
        all_hit_indices = np.where(hit_mask_all)[0]
        
        indices_of_interest = np.array([], dtype=np.int64)
        
        # 2. If we have a target, filter ALL hits to find those that reflect to it
        if target_point is not None and len(all_hit_indices) > 0 and 'primitive_normals' in ans:
            # Extract data for ALL hits (this might be large but ensures we don't miss)
            # Using a subset if memory is an issue, but for 25M rays it's manageable on modern RAM
            
            # Get normals for all hits
            normals_all = ans['primitive_normals'].numpy().reshape(-1, 3)[all_hit_indices]
            dirs_all = rays_np[all_hit_indices, 3:]
            origins_all = rays_np[all_hit_indices, :3]
            dists_all = t_hit[all_hit_indices]
            ends_all = origins_all + dirs_all * dists_all[:, np.newaxis]
            
            # Calculate Reflection for ALL hits
            dot_all = np.einsum('ij,ij->i', dirs_all, normals_all)
            R_all = dirs_all - 2 * dot_all[:, np.newaxis] * normals_all
            # Normalize
            R_norm_all = R_all / (np.linalg.norm(R_all, axis=1)[:, np.newaxis] + 1e-6)
            
            # Check intersection with Voxel
            to_target_all = target_point - ends_all
            cross_prod_all = np.cross(to_target_all, R_norm_all)
            dist_to_line_all = np.linalg.norm(cross_prod_all, axis=1)
            dot_prod_all = np.einsum('ij,ij->i', to_target_all, R_norm_all)
            
            # Filter: Distance < voxel_size AND moving towards target
            voxel_hit_mask = (dist_to_line_all < voxel_size) & (dot_prod_all > 0)
            indices_of_interest = all_hit_indices[voxel_hit_mask]
            
        # 3. Combine with a random sample for context
        num_context = 2000
        if len(all_hit_indices) > num_context:
            random_indices = np.random.choice(all_hit_indices, size=num_context, replace=False)
        else:
            random_indices = all_hit_indices
            
        # Final indices to visualize: Context + Voxel Hits
        indices = np.unique(np.concatenate((random_indices, indices_of_interest)))
        
        # Get data for sampled rays
        dists_subset = t_hit[indices]
        hit_mask = np.isfinite(dists_subset)
        
        origins = rays_np[indices][hit_mask, :3]
        dirs = rays_np[indices][hit_mask, 3:]
        dists = dists_subset[hit_mask]
        ends = origins + dirs * dists[:, np.newaxis]
        
        # --- Compute Reflected Rays ---
        vis_reflected = o3d.geometry.LineSet()
        
        if 'primitive_normals' in ans:
            # Get normals (N, 3)
            normals = ans['primitive_normals'].numpy().reshape(-1, 3)
            hit_normals = normals[indices][hit_mask]
            
            # Reflection: R = D - 2(D.N)N
            dot = np.einsum('ij,ij->i', dirs, hit_normals)
            R = dirs - 2 * dot[:, np.newaxis] * hit_normals
            
            # --- "Near Hit" Filtering ---
            # If a target is provided, we check which rays pass close to it.
            # Distance from point P to line (A, d): || (P-A) x d || / ||d||
            # R is normalized? No, dirs is normalized, R should be normalized.
            # Let's ensure R is normalized.
            R_norm = R / np.linalg.norm(R, axis=1)[:, np.newaxis]
            
            # Vector from Hit Point (ends) to Target
            if target_point is not None:
                to_target = target_point - ends
                
                # Cross product magnitude
                cross_prod = np.cross(to_target, R_norm)
                dist_to_line = np.linalg.norm(cross_prod, axis=1)
                
                # Check if target is "in front" of the reflection (dot product > 0)
                dot_prod = np.einsum('ij,ij->i', to_target, R_norm)
                forward_mask = dot_prod > 0
                
                # Filter: Close enough AND forward
                # We use a generous tolerance (e.g., 2x voxel size) to make sure we see something
                near_hit_mask = (dist_to_line < voxel_size) & forward_mask
                
                # Color logic:
                # Standard rays: Purple
                # Near-hit rays: Bright Cyan or White
                
                # We will visualize ALL sampled rays as short purple lines
                # AND extend the "near hit" rays to the target
                
                # 1. Standard Short Reflections (Purple)
                ref_len = 0.1
                ref_ends = ends + R_norm * ref_len
                
                points = np.vstack((ends, ref_ends))
                lines = [[i, i + len(ends)] for i in range(len(ends))]
                colors = np.tile([0.5, 0, 0.5], (len(lines), 1)) # Purple
                
                # 2. Highlight Near Hits (Cyan, Full Length)
                if np.any(near_hit_mask):
                    hit_starts = ends[near_hit_mask]
                    hit_dirs = R_norm[near_hit_mask]
                    # Length to target (projected)
                    hit_lens = dot_prod[near_hit_mask]
                    hit_ends = hit_starts + hit_dirs * hit_lens[:, np.newaxis]
                    
                    # Append to geometry
                    start_idx = len(points)
                    points = np.vstack((points, hit_starts, hit_ends))
                    new_lines = [[start_idx + i, start_idx + len(hit_starts) + i] for i in range(len(hit_starts))]
                    lines.extend(new_lines)
                    
                    # Add Cyan color for hits
                    hit_colors = np.tile([0, 1, 1], (len(new_lines), 1))
                    colors = np.vstack((colors, hit_colors))
                    
                    print(f"  [Visualizer] Found {np.sum(near_hit_mask)} rays passing within {voxel_size:.3f} of target.")

                vis_reflected.points = o3d.utility.Vector3dVector(points)
                vis_reflected.lines = o3d.utility.Vector2iVector(lines)
                vis_reflected.colors = o3d.utility.Vector3dVector(colors)
                
            else:
                # Fallback: Old behavior
                ref_len = 0.1
                ref_ends = ends + R * ref_len
                vis_reflected.points = o3d.utility.Vector3dVector(np.vstack((ends, ref_ends)))
                vis_reflected.lines = o3d.utility.Vector2iVector([[i, i + len(ends)] for i in range(len(ends))])
                vis_reflected.paint_uniform_color([0.5, 0, 0.5])

        # Create LineSet for primary rays (Yellow)
        vis_rays = o3d.geometry.LineSet()
        vis_rays.points = o3d.utility.Vector3dVector(np.vstack((origins, ends)))
        vis_rays.lines = o3d.utility.Vector2iVector([[i, i + len(origins)] for i in range(len(origins))])
        vis_rays.paint_uniform_color([1, 0.8, 0]) # Yellow

    if return_ray_vis:
        return set(visible_ids), vis_rays, vis_reflected
    return set(visible_ids)

# ==========================================
# 2. HELPER: CAMERA POSITIONING
# ==========================================
def get_auto_camera_pose(mesh):
    """f
    """
    center = mesh.bounding_box.centroid
    scale = np.linalg.norm(mesh.bounds[1] - mesh.bounds[0])
    distance = 0.35
    
    pose = np.eye(4)
    # Move camera to +Z distance
    pose[0:3, 3] = center + [0, 0, distance]
    # (Default pyrender camera looks -Z, so this places it "in front")
    
    return pose

# ==========================================
# 2.5 HELPER: CREATE HUMAN MESH
# ==========================================
def create_dummy_human():
    """Attempts to load a real human mesh, falls back to primitives."""
    # 1. Try downloading a real mesh
    url = "https://raw.githubusercontent.com/zalo/MathUtilities/master/Assets/Models/Man.obj"
    try:
        resp = requests.get(url, timeout=5)
        if resp.status_code == 200:
            mesh = trimesh.load(io.BytesIO(resp.content), file_type='obj')
            if isinstance(mesh, trimesh.Scene):
                mesh = trimesh.util.concatenate(mesh.dump())
            # Normalize scale (approx 1.8m height)
            mesh.apply_scale(1.8 / mesh.extents[1])
            # Center bottom at 0
            mesh.apply_translation([0, -mesh.bounds[0][1] - 0.9, 0])
            return mesh
    except Exception as e:
        print(f"Human mesh download failed: {e}")

    # 2. Fallback: Primitives
    parts = []
    # Head
    parts.append(trimesh.creation.icosphere(radius=0.15).apply_translation([0, 0.65, 0]))
    # Body
    parts.append(trimesh.creation.box(extents=[0.3, 0.5, 0.15]).apply_translation([0, 0.25, 0]))
    # Legs
    parts.append(trimesh.creation.box(extents=[0.1, 0.6, 0.1]).apply_translation([-0.1, -0.3, 0]))
    parts.append(trimesh.creation.box(extents=[0.1, 0.6, 0.1]).apply_translation([0.1, -0.3, 0]))
    # Arms
    parts.append(trimesh.creation.box(extents=[0.1, 0.5, 0.1]).apply_translation([-0.25, 0.25, 0]))
    parts.append(trimesh.creation.box(extents=[0.1, 0.5, 0.1]).apply_translation([0.25, 0.25, 0]))
    return trimesh.util.concatenate(parts)

# ==========================================
# 2.6 ALGORITHM: ONE-BOUNCE REFLECTION
# ==========================================
def find_one_bounce_paths(mesh, camera_pose, target_point, voxel_size=0.1):
    """
    Finds all paths: Camera -> Mesh (1 bounce) -> Target.
    Returns a list of points [P_bounce, ...] on the mesh.
    """
    # 1. Setup Open3D scene for visibility checks
    vertices = np.array(mesh.vertices, dtype=np.float32)
    triangles = np.array(mesh.faces, dtype=np.uint32)
    t_mesh = o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(vertices),
        o3d.core.Tensor(triangles)
    )
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(t_mesh)

    cam_pos = camera_pose[:3, 3]
    
    # 2. Vectorized Reflection Calculation
    # Get triangle vertices: (N, 3, 3)
    tris = mesh.vertices[mesh.faces]
    A = tris[:, 0, :]
    B = tris[:, 1, :]
    C = tris[:, 2, :]
    
    # Compute Normals
    normals = np.cross(B - A, C - A)
    normals /= np.linalg.norm(normals, axis=1)[:, np.newaxis]
    
    # Plane equation: dot(n, x) + d = 0  =>  d = -dot(n, A)
    d = -np.einsum('ij,ij->i', normals, A)
    
    # Distance from Camera to Plane
    dist_cam = np.einsum('ij,j->i', normals, cam_pos) + d
    
    # Reflected Camera Position (Virtual Source) for each plane
    # C' = C - 2 * dist * n
    cam_reflected = cam_pos - (2 * dist_cam[:, np.newaxis] * normals)
    
    # --- Handle Voxel Volume (Multi-sample) ---
    if voxel_size > 0:
        r = voxel_size / 2.0
        # Sample center + 8 corners
        offsets = np.array([
            [0, 0, 0],
            [r, r, r], [r, r, -r], [r, -r, r], [r, -r, -r],
            [-r, r, r], [-r, r, -r], [-r, -r, r], [-r, -r, -r]
        ])
        targets = target_point + offsets # (K, 3)
    else:
        targets = target_point[np.newaxis, :] # (1, 3)

    # Prepare for broadcasting (N, K)
    # normals: (N, 3) -> (N, 1, 3)
    n_exp = normals[:, np.newaxis, :]
    # cam_reflected: (N, 3) -> (N, 1, 3)
    c_exp = cam_reflected[:, np.newaxis, :]
    # targets: (K, 3) -> (1, K, 3)
    t_exp = targets[np.newaxis, :, :]
    
    # Ray dirs: (N, K, 3)
    ray_dirs = t_exp - c_exp
    
    # Intersect Line (C', Target) with Plane
    # t = -(dot(n, C') + d) / dot(n, dir)
    denom = np.einsum('ijk,ijk->ij', n_exp, ray_dirs)
    valid_mask = np.abs(denom) > 1e-6
    
    # numer = -(dot(n, C') + d) -> (N, 1)
    val = np.einsum('ij,ij->i', normals, cam_reflected) + d
    numer = -val[:, np.newaxis]
    
    # Flatten indices for valid intersections
    tri_indices, target_indices = np.where(valid_mask)
    
    t = numer[tri_indices, 0] / denom[tri_indices, target_indices]
    
    # Intersection Point P
    P = cam_reflected[tri_indices] + t[:, np.newaxis] * ray_dirs[tri_indices, target_indices]
    
    # 3. Check if P is inside the Triangle (Barycentric)
    A_v = A[tri_indices]; B_v = B[tri_indices]; C_v = C[tri_indices]
    
    v0 = B_v - A_v
    v1 = C_v - A_v
    v2 = P - A_v
    
    d00 = np.einsum('ij,ij->i', v0, v0)
    d01 = np.einsum('ij,ij->i', v0, v1)
    d11 = np.einsum('ij,ij->i', v1, v1)
    d20 = np.einsum('ij,ij->i', v2, v0)
    d21 = np.einsum('ij,ij->i', v2, v1)
    
    denom_bary = d00 * d11 - d01 * d01
    valid_bary = np.abs(denom_bary) > 1e-8
    
    v = np.zeros_like(denom_bary)
    w = np.zeros_like(denom_bary)
    
    v[valid_bary] = (d11[valid_bary] * d20[valid_bary] - d01[valid_bary] * d21[valid_bary]) / denom_bary[valid_bary]
    w[valid_bary] = (d00[valid_bary] * d21[valid_bary] - d01[valid_bary] * d20[valid_bary]) / denom_bary[valid_bary]
    u = 1.0 - v - w
    
    # Check bounds (with small epsilon)
    inside_mask = (u >= -1e-4) & (v >= -1e-4) & (w >= -1e-4) & valid_bary
    
    # --- Visibility Check (Occlusion Culling) ---
    # Filter geometrically valid points first
    candidate_indices = np.where(inside_mask)[0]
    if len(candidate_indices) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
        
    P_cand = P[candidate_indices]
    # We need to know which target corresponds to each P to check P->Target visibility
    target_inds_cand = target_indices[candidate_indices]
    targets_cand = targets[target_inds_cand]
    
    # 1. Check Visibility: Camera -> P
    # Ray: Origin=Cam, Dir=(P-Cam). We check if the first hit is approximately at distance ||P-Cam||
    dirs_cam_p = P_cand - cam_pos
    dists_cam_p = np.linalg.norm(dirs_cam_p, axis=1)
    dirs_cam_p_norm = dirs_cam_p / (dists_cam_p[:, np.newaxis] + 1e-6)
    
    rays_cam_p = np.concatenate([
        np.tile(cam_pos, (len(P_cand), 1)), 
        dirs_cam_p_norm
    ], axis=1).astype(np.float32)
    
    ans_cam = scene.cast_rays(o3d.core.Tensor(rays_cam_p))
    t_hit_cam = ans_cam['t_hit'].numpy()
    
    # Visible if hit distance is close to expected distance (tolerance for float precision)
    # If t_hit < dist - epsilon, it hit something in front (occluded)
    is_visible_cam = t_hit_cam >= (dists_cam_p - 1e-3)
    
    # 2. Check Visibility: P -> Target
    # Ray: Origin=P + eps*N, Dir=(Target-P). We check if it hits anything before Target.
    cand_tri_indices = tri_indices[candidate_indices]
    cand_normals = normals[cand_tri_indices]
    
    dirs_p_target = targets_cand - P_cand
    dists_p_target = np.linalg.norm(dirs_p_target, axis=1)
    dirs_p_target_norm = dirs_p_target / (dists_p_target[:, np.newaxis] + 1e-6)
    
    # Offset origin slightly along normal to avoid self-intersection with the triangle itself
    origins_p = P_cand + cand_normals * 1e-3
    
    rays_p_target = np.concatenate([origins_p, dirs_p_target_norm], axis=1).astype(np.float32)
    
    ans_target = scene.cast_rays(o3d.core.Tensor(rays_p_target))
    t_hit_target = ans_target['t_hit'].numpy()
    
    # Visible if NO hit (inf), OR hit is further than target distance
    is_visible_target = t_hit_target > (dists_p_target - 1e-3)
    
    final_mask = is_visible_cam & is_visible_target
    return P_cand[final_mask], targets_cand[final_mask]

# ==========================================
# 3. VISUALIZATION
# ==========================================
def visualize_results(mesh, visible_ids, camera_pose, reflection_paths=None, target_point=None, lidar_rays=None, reflected_rays=None, voxel_size=None):
    """
    Opens Open3D viewer with:
    - GREEN Mesh = Visible Triangles
    - GREY Mesh  = Hidden Triangles
    - BLUE Frustum = Camera Position
    - RED Lines = Reflection Paths
    - YELLOW Lines = LiDAR Rays
    - PURPLE Lines = Reflected Rays
    - CYAN Line = Direct Path (Camera -> Target)
    """
    # --- 1. PREPARE MESH (EXPLODED FOR FLAT SHADING) ---
    # Open3D interpolates vertex colors. To get per-face colors (flat look),
    # we duplicate vertices for every face.
    triangles = mesh.faces
    vertices = mesh.vertices[triangles.flatten()]
    
    # Create new faces: (0, 1, 2), (3, 4, 5), ...
    new_triangles = np.arange(len(vertices)).reshape(-1, 3)
    
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # --- 2. COLORING ---
    # Default Grey: [0.5, 0.5, 0.5]
    colors = np.full((len(vertices), 3), 0.5)
    
    # Green: [0, 1, 0]
    green = np.array([0.0, 1.0, 0.0])
    
    for fid in visible_ids:
        idx = fid * 3
        colors[idx:idx+3] = green
        
    o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    o3d_mesh.compute_vertex_normals()

    # --- 3. CAMERA MARKER ---
    cam = trimesh.scene.Camera(fov=[fov_deg, fov_deg]) 
    marker_ret = trimesh.creation.camera_marker(cam, marker_height=mesh.scale * 0.2)
    marker_geo = marker_ret[1] # This is a Path3D
    marker_geo.apply_transform(camera_pose)
    
    # Convert Path3D to Open3D LineSet
    line_points = marker_geo.vertices
    line_indices = []
    for entity in marker_geo.entities:
        indices = entity.points
        for i in range(len(indices) - 1):
            line_indices.append([indices[i], indices[i+1]])
            
    o3d_lines = o3d.geometry.LineSet()
    o3d_lines.points = o3d.utility.Vector3dVector(line_points)
    o3d_lines.lines = o3d.utility.Vector2iVector(line_indices)
    o3d_lines.paint_uniform_color([0, 0, 1]) # Blue

    geometries = [o3d_mesh, o3d_lines]

    # --- 4. REFLECTION PATHS ---
    if reflection_paths is not None:
        # Handle tuple (bounces, targets) or just bounces (backward compatibility)
        if isinstance(reflection_paths, tuple):
            bounces, targets = reflection_paths
        else:
            bounces = reflection_paths
            # If no specific targets provided, default to center
            targets = np.tile(target_point, (len(bounces), 1)) if target_point is not None else np.zeros_like(bounces)

    if reflection_paths is not None and len(bounces) > 0:
        cam_pos = camera_pose[:3, 3]
        points = []
        indices = []
        
        for i in range(len(bounces)):
            bounce_point = bounces[i]
            tgt = targets[i]
            # Path: Camera -> Bounce -> Specific Target Point
            base_idx = len(points)
            points.extend([cam_pos, bounce_point, tgt])
            indices.extend([[base_idx, base_idx+1], [base_idx+1, base_idx+2]])
            
        path_lines = o3d.geometry.LineSet()
        path_lines.points = o3d.utility.Vector3dVector(points)
        path_lines.lines = o3d.utility.Vector2iVector(indices)
        path_lines.paint_uniform_color([1, 0, 0]) # Red
        geometries.append(path_lines)

    if lidar_rays is not None:
        geometries.append(lidar_rays)

    if reflected_rays is not None:
        geometries.append(reflected_rays)

    # --- 5. DIRECT LINE TO TARGET ---
    if target_point is not None:
        cam_pos = camera_pose[:3, 3]
        direct_line = o3d.geometry.LineSet()
        direct_line.points = o3d.utility.Vector3dVector(np.array([cam_pos, target_point]))
        direct_line.lines = o3d.utility.Vector2iVector([[0, 1]])
        direct_line.paint_uniform_color([0, 1, 1]) # Cyan
        geometries.append(direct_line)
        
        # Draw Target Voxel (Cube)
        if voxel_size is not None:
            s = voxel_size
        else:
            s = mesh.scale * 0.05
        voxel = o3d.geometry.TriangleMesh.create_box(width=s, height=s, depth=s)
        voxel.compute_vertex_normals()
        voxel.translate(target_point - np.array([s/2, s/2, s/2]))
        voxel.paint_uniform_color([1, 0, 1]) # Magenta
        geometries.append(voxel)

    # --- 6. WORLD AXIS ---
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=mesh.scale * 0.5, origin=[0, 0, 0])
    geometries.append(axis)

    print("\n-------------------------------------------")
    print("Opening Open3D Viewer...")
    print("  GREEN Mesh = Visible Triangles")
    print("  GREY Mesh  = Hidden Triangles")
    print("  BLUE Frustum = Camera Position")
    print("  RED Lines    = Reflection Paths")
    print("  YELLOW Lines = LiDAR Rays")
    print("  PURPLE Lines = Reflected Rays")
    print("  CYAN Line    = Direct Path (Camera -> Target)")
    print("  MAGENTA Box  = Target Voxel")
    print("  RGB Axis     = World Frame (Red=X, Green=Y, Blue=Z)")
    print("  Press ENTER to exit")
    print("-------------------------------------------")
    
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name="Visibility Result")
    for geo in geometries:
        vis.add_geometry(geo)
    vis.get_render_option().light_on = False
    # Register Enter key (257) to close the window
    vis.register_key_callback(257, lambda v: v.destroy_window())
    vis.run()
    vis.destroy_window()
    
# ==========================================
# 4. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Load Stanford Bunny
    print("Downloading Stanford Bunny...")
    url = "https://raw.githubusercontent.com/mikedh/trimesh/master/models/bunny.ply"
    try:
        data = requests.get(url).content
        mesh = trimesh.load(io.BytesIO(data), file_type='ply')
    except Exception as e:
        print("Download failed. Loading a sphere instead.")
        mesh = trimesh.creation.icosphere(subdivisions=3)

    print(f"Mesh: {len(mesh.faces)} triangles")

    # Create a simple scene: 1 bunny and 1 human
    print("Generating scene (1 bunny, 1 human)...")
    human = create_dummy_human()
    human.apply_scale(mesh.scale / human.scale) # Match scale
    
    mesh.apply_translation([-mesh.scale * 0.6, 0, 0])
    human.apply_translation([mesh.scale * 0.6, 0, 0])
    # mesh = trimesh.util.concatenate([mesh])
    mesh = trimesh.util.concatenate([mesh])

    # 2. Position Camera
    pose = get_auto_camera_pose(mesh)
    
    # 3. Compute Visibility
    # Pass target info to visualize near-hits
    visible, vis_rays, vis_reflected = find_visible_triangles(
        mesh, pose, 
        resolution=(resolution_x, resolution_y), 
        return_ray_vis=True, 
        target_point=target_voxel, 
        voxel_size=voxel_size# Match the visualization size
    )
    
    print(f"Found {len(visible)} visible triangles.")

    # 4. Compute Reflections
    # Define a target voxel (point) somewhere above/to the side
    print(f"Calculating reflections to target: {target_voxel}")
    
    reflection_hits, reflection_targets = find_one_bounce_paths(mesh, pose, target_voxel, voxel_size=voxel_size)
    print(f"Found {len(reflection_hits)} potential reflection paths.")

    # 5. Visualize
    visualize_results(mesh, visible, pose, (reflection_hits, reflection_targets), target_voxel, lidar_rays=vis_rays, reflected_rays=vis_reflected, voxel_size=voxel_size)