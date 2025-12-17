import open3d as o3d
import numpy as np
import time
from utils import load_bunny_mesh, create_pixel_grid, plot_heatmap

# --- PARAMETERS ---
target_triangles = 5000       # Mesh simplification target
check_occlusion = False
num_vis = 45    

source_points = [
    np.array([2.5, 2.3, 1.4]),
    np.array([2.5, 2.35, 1.4]),
    np.array([2.5, 2.4, 1.40]),
    np.array([2.5, 2.45, 1.40]),
    np.array([2.5, 2.5, 1.45]),
    np.array([2.5, 2.5, 1.5]),
    np.array([2.5, 2.5, 1.55]),
    np.array([2.5, 2.5, 1.6])
]

# Grid settings
grid_center = np.array([1.5, 3.6, 5.4])   # Center of the pixel grid
grid_normal = np.array([-0.2, -0.5, -0.8]) # Direction the grid is facing (towards scene)
grid_up     = np.array([0.0, 1.0, 0.0])   # Up vector
grid_width  = 2.0
grid_height = 2.0

res_x   = 30                  # Horizontal resolution
res_y   = 30                  # Vertical resolution

def compute_reflections(mesh, source_point, grid_center, grid_normal, grid_up, grid_width, grid_height, res_x, res_y, samples=4, check_occlusion=False):
    """
    Finds all valid one-bounce paths: Source -> Mesh -> Pixel.
    Uses supersampling (samples x samples) to treat pixels as areas.
    Returns: list of tuples (pixel_point, bounce_point)
    """
    print(f"3. Computing Reflections (Vectorized) with {samples}x{samples} supersampling...")
    
    # Generate dense grid for area calculation (Supersampling)
    pixel_grid = create_pixel_grid(grid_center, grid_normal, grid_up, grid_width, grid_height, res_x * samples, res_y * samples, verbose=False)
    
    # --- A. Prepare Data ---
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    # Triangle vertices (N, 3, 3)
    tris = vertices[triangles]
    A = tris[:, 0, :]
    B = tris[:, 1, :]
    C = tris[:, 2, :]
    
    # Face Normals (N, 3)
    normals = np.asarray(mesh.triangle_normals)
    
    # --- CULLING ---
    # Filter out triangles that face away from the light source.
    # Light cannot reflect off the back face of an opaque triangle.
    vec_source = source_point - A
    dot_source = np.einsum('ij,ij->i', normals, vec_source)
    valid_tri_mask = dot_source > 1e-6
    print(f"   [Culling] Active triangles: {np.sum(valid_tri_mask)} / {len(normals)}")
    
    if np.sum(valid_tri_mask) == 0:
        return [], np.zeros((res_y, res_x), dtype=np.int32)

    A = A[valid_tri_mask]
    B = B[valid_tri_mask]
    C = C[valid_tri_mask]
    normals = normals[valid_tri_mask]
    
    # --- B. Virtual Source Calculation ---
    # Plane equation: dot(n, x) + d = 0  => d = -dot(n, A)
    d = -np.einsum('ij,ij->i', normals, A)
    
    # Distance from Source to Plane
    dist_s = np.einsum('ij,j->i', normals, source_point) + d
    
    # Virtual Sources S' for each triangle (N, 3)
    # S' = S - 2 * (dist) * n
    virtual_sources = source_point - (2 * dist_s[:, np.newaxis] * normals)
    
    # --- C. Intersection: Line(S', Pixel) vs Plane ---
    # We need to check every Triangle vs every Pixel.
    # N = num_triangles, K = num_pixels
    # Broadcasting: (N, 1, 3) vs (1, K, 3)
    
    S_prime = virtual_sources[:, np.newaxis, :] # (N, 1, 3)
    Pixels = pixel_grid[np.newaxis, :, :]       # (1, K, 3)
    
    # Ray direction from Virtual Source to Pixel
    Ray_Dirs = Pixels - S_prime # (N, K, 3)
    
    # Intersect Ray with Plane
    # t = -(dot(n, S') + d) / dot(n, Ray_Dir)
    
    # Expand normals and d for broadcasting
    N_exp = normals[:, np.newaxis, :] # (N, 1, 3)
    d_exp = d[:, np.newaxis]          # (N, 1)
    
    # Denominator: dot(n, Ray_Dir)
    denom = np.einsum('ijk,ijk->ij', N_exp, Ray_Dirs)
    
    # Filter parallel rays
    valid_denom = np.abs(denom) > 1e-6
    
    # Numerator: -(dot(n, S') + d)
    # Note: dot(n, S') is actually -dot(n, S) because S' is reflected S.
    # Let's calculate explicitly to be safe.
    numer = -(np.einsum('ijk,ijk->ij', N_exp, S_prime) + d_exp)
    
    # Broadcast numer to match denom shape (N, K) for boolean indexing
    numer = np.broadcast_to(numer, denom.shape)
    
    # Calculate t
    # We only compute t where denom is valid to avoid div/0 warnings
    t = np.zeros_like(denom)
    t[valid_denom] = numer[valid_denom] / denom[valid_denom]
    
    # Intersection Points P (Bounce Points)
    # P = S' + t * Ray_Dir
    P = S_prime + t[..., np.newaxis] * Ray_Dirs
    
    # --- D. Barycentric Check (Is P inside Triangle?) ---
    # Vectors for barycentric
    # A, B, C are (N, 3). Expand to (N, K, 3)
    A_exp = A[:, np.newaxis, :]
    B_exp = B[:, np.newaxis, :]
    C_exp = C[:, np.newaxis, :]
    
    v0 = B_exp - A_exp
    v1 = C_exp - A_exp
    v2 = P - A_exp
    
    d00 = np.einsum('ijk,ijk->ij', v0, v0)
    d01 = np.einsum('ijk,ijk->ij', v0, v1)
    d11 = np.einsum('ijk,ijk->ij', v1, v1)
    
    # Broadcast triangle properties to (N, K) to match per-pixel calculations
    d00 = np.broadcast_to(d00, denom.shape)
    d01 = np.broadcast_to(d01, denom.shape)
    d11 = np.broadcast_to(d11, denom.shape)
    
    d20 = np.einsum('ijk,ijk->ij', v2, v0)
    d21 = np.einsum('ijk,ijk->ij', v2, v1)
    
    denom_bary = d00 * d11 - d01 * d01
    
    # Avoid div/0 in barycentric
    valid_bary_denom = np.abs(denom_bary) > 1e-10
    
    v = np.zeros_like(denom_bary)
    w = np.zeros_like(denom_bary)
    
    mask_calc = valid_denom & valid_bary_denom
    
    v[mask_calc] = (d11[mask_calc] * d20[mask_calc] - d01[mask_calc] * d21[mask_calc]) / denom_bary[mask_calc]
    w[mask_calc] = (d00[mask_calc] * d21[mask_calc] - d01[mask_calc] * d20[mask_calc]) / denom_bary[mask_calc]
    u = 1.0 - v - w
    
    # Inside Triangle Test
    inside_mask = mask_calc & (u >= 0) & (v >= 0) & (w >= 0)
    
    # Extract Candidates
    tri_indices, pixel_indices = np.where(inside_mask)
    
    if len(tri_indices) == 0:
        return [], np.zeros((res_y, res_x), dtype=np.int32)

    print(f"   Found {len(tri_indices)} geometric candidates. Checking visibility...")
    
    # Gather candidate points
    candidate_P = P[tri_indices, pixel_indices]
    candidate_Pixels = pixel_grid[pixel_indices]
    
    # --- E. Visibility Check (Raycasting) ---
    if check_occlusion:
        # We need to check:
        # 1. Source -> P
        # 2. P -> Pixel
        
        # Setup Raycasting Scene
        t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
        scene = o3d.t.geometry.RaycastingScene()
        scene.add_triangles(t_mesh)
        
        # 1. Check Source -> P
        # Ray: Origin=Source, Dir=(P-Source)
        dirs_s_p = candidate_P - source_point
        dists_s_p = np.linalg.norm(dirs_s_p, axis=1)
        dirs_s_p_norm = dirs_s_p / (dists_s_p[:, np.newaxis] + 1e-6)
        
        rays_s_p = np.concatenate([
            np.tile(source_point, (len(candidate_P), 1)),
            dirs_s_p_norm
        ], axis=1).astype(np.float32)
        
        hits_s_p = scene.cast_rays(o3d.core.Tensor(rays_s_p))
        t_hit_s_p = hits_s_p['t_hit'].numpy()
        
        # Visible if hit distance is approx equal to distance to P
        vis_s_p = t_hit_s_p >= (dists_s_p - 1e-3)
        
        # 2. Check P -> Pixel
        # Ray: Origin=P, Dir=(Pixel-P)
        # Offset P slightly by normal to avoid self-intersection
        cand_normals = normals[tri_indices]
        origins_p = candidate_P + cand_normals * 1e-3
        
        dirs_p_pix = candidate_Pixels - candidate_P
        dists_p_pix = np.linalg.norm(dirs_p_pix, axis=1)
        dirs_p_pix_norm = dirs_p_pix / (dists_p_pix[:, np.newaxis] + 1e-6)
        
        rays_p_pix = np.concatenate([
            origins_p,
            dirs_p_pix_norm
        ], axis=1).astype(np.float32)
        
        hits_p_pix = scene.cast_rays(o3d.core.Tensor(rays_p_pix))
        t_hit_p_pix = hits_p_pix['t_hit'].numpy()
        
        # Visible if NO hit (inf) OR hit is further than pixel
        vis_p_pix = t_hit_p_pix > (dists_p_pix - 1e-3)
        
        # Final Valid Paths
        final_mask = vis_s_p & vis_p_pix
    else:
        final_mask = np.ones(len(candidate_P), dtype=bool)
    
    valid_P = candidate_P[final_mask]
    valid_Pixels = candidate_Pixels[final_mask]
    valid_pixel_indices = pixel_indices[final_mask]
    
    # --- Count rays per pixel ---
    heatmap = np.zeros((res_y, res_x), dtype=np.int32)
    
    if len(valid_pixel_indices) > 0:
        width_dense = res_x * samples
        
        # Map dense index to coarse row/col
        row_dense = valid_pixel_indices // width_dense
        col_dense = valid_pixel_indices % width_dense
        
        row_coarse = row_dense // samples
        col_coarse = col_dense // samples
        
        # Accumulate
        np.add.at(heatmap, (row_coarse, col_coarse), 1)
    
    print(f"   Confirmed {len(valid_P)} valid paths.")
    
    return list(zip(valid_Pixels, valid_P)), heatmap

def main():
    t0 = time.time()

    # ------------------
    # 1. Setup
    mesh = load_bunny_mesh(target_triangles=target_triangles)
    
    # Change pose: Rotate 45 degrees around Y-axis
    R = mesh.get_rotation_matrix_from_xyz((0, np.pi / 4, 0))
    mesh.rotate(R, center=(0, 0, 0))
    mesh.translate([0.7, 0.0, 0.5])
    
    # 2. Compute
    pixel_grid = create_pixel_grid(grid_center, grid_normal, grid_up, grid_width, grid_height, res_x, res_y) # For visualization
    
    all_paths = []
    total_heatmap = np.zeros((res_y, res_x), dtype=np.int32)
    
    for sp in source_points:
        print(f"Computing for source: {sp}")
        paths, heatmap = compute_reflections(mesh, sp, grid_center, grid_normal, grid_up, grid_width, grid_height, res_x, res_y, samples=1, check_occlusion=check_occlusion)
        total_heatmap += heatmap
        for p in paths:
            all_paths.append((sp, p[0], p[1]))
            
    # print("Hit Counts per Pixel (Heatmap):")
    # print(total_heatmap)
    # print(f"Max hits in a single pixel: {np.max(total_heatmap)}")

    # Plot the heatmap using matplotlib
    plot_heatmap(total_heatmap)
    print(f"Total duration: {time.time() - t0:.4f} seconds")
    
    # 3. Visualize
    print("4. Visualizing...")
    geometries = [mesh]
    
    # A. Draw Light Sources (Yellow Spheres)
    for sp in source_points:
        light_geo = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        light_geo.translate(sp)
        light_geo.paint_uniform_color([1, 0.4, 0])
        geometries.append(light_geo)
    
    # B. Draw Pixel Grid (Blue Points)
    pcd_grid = o3d.geometry.PointCloud()
    pcd_grid.points = o3d.utility.Vector3dVector(pixel_grid)
    pcd_grid.paint_uniform_color([0, 0, 1])
    geometries.append(pcd_grid)
    
    # C. Draw Reflection Paths (Red Lines)
    if all_paths:
        points = []
        lines = []
        colors = []
        
        # Sample rays for visualization
        num_vis = num_vis
        indices = np.arange(len(all_paths))
        if len(all_paths) > num_vis:
            indices = np.random.choice(len(all_paths), num_vis, replace=False)

        for i in indices:
            src, pix, bounce = all_paths[i]
            # Path: Source -> Bounce -> Pixel
            base = len(points)
            points.append(src)
            points.append(bounce)
            points.append(pix)
            
            lines.append([base, base+1])
            lines.append([base+1, base+2])
            
            # Red color
            colors.append([0.6, 0, 0.2])
            colors.append([0.2, 0.5, 0.1])
            
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(points)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(line_set)
    
    # D. Coordinate Frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0]))
    
    o3d.visualization.draw_geometries(geometries, window_name="One-Bounce Paths")

if __name__ == "__main__":
    main()
