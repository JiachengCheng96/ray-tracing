import open3d as o3d
import numpy as np
import time
import copy
from utils import load_bunny_mesh, create_pixel_grid, plot_heatmap, create_ground_grid, create_sensor_wireframe



print(o3d.__version__)
# print(o3d.__doc__)
# --- CONFIGURATION ---

num_samples_per_source=2100000

target_triangles = int(2e9)       # Mesh simplification target
check_occlusion = False
num_vis = 50    

source_center = np.array([3., .5, .5]) / 1
spacing_size = 0.01
source_points = [
    source_center + spacing_size * np.array([1., .0, .0]),
    # source_center + spacing_size * np.array([2., .0, .0]),
    # source_center + spacing_size * np.array([3., .0, .0]),
    # source_center + spacing_size * np.array([4., .0, .0]),

    # source_center + spacing_size * np.array([.0, 1., .0]),
    # source_center + spacing_size * np.array([.0, 2., .0]),
    # source_center + spacing_size * np.array([.0, 3., .0]),
    # source_center + spacing_size * np.array([.0, 4., .0]),
    
    # source_center + spacing_size * np.array([.0, .0, 1.]),
    # source_center + spacing_size * np.array([.0, .0, 2.]),
    # source_center + spacing_size * np.array([.0, .0, 3.]),
    # source_center + spacing_size * np.array([.0, .0, 4.]),

]

# Grid settings
grid_center = np.array([1.5, 3.6, 3.6]) /3  # Center of the pixel grid
grid_normal = np.array([-0.2, -0.5, -0.8]) # Direction the grid is facing (towards scene)
grid_up     = np.array([0.0, 1.0, 0.0])   # Up vector
grid_width  = 0.5
grid_height = 0.5

pixel_size = 0.5 # unit: cm

# (grid_width * 180) cm = pixel_size * res_x
# res_x = pixel_size * 180/1  
res_x   = int((grid_width * 180) / pixel_size) #                 # Horizontal resolution
res_y   = int((grid_height * 180) / pixel_size)                  # Vertical resolution

def compute_reflections_forward(mesh, source_points, grid_center, grid_normal, grid_up, grid_width, grid_height, res_x, res_y, num_samples=100000, check_occlusion=False):
    """
    Performs Forward Ray Tracing for multiple light sources:
    1. Sample points on the mesh (once).
    2. For each source:
       a. Calculate reflection vectors.
       b. Intersect with sensor plane.
       c. Map hits to heatmap.
    """
    print(f"3. Computing Reflections (Forward Tracing) with {num_samples} rays for {len(source_points)} sources...")
    
    # Use CPU for everything
    device = o3d.core.Device("cpu:0")
    print(f"   Using Device: {device}")

    # Ensure source_points is iterable
    if isinstance(source_points, np.ndarray) and source_points.ndim == 1:
        source_points = [source_points]
    
    # --- A. Sample Mesh ---
    pcd = mesh.sample_points_uniformly(number_of_points=num_samples)
    
    # Convert to Open3D Tensors
    P = o3d.core.Tensor(np.asarray(pcd.points), dtype=o3d.core.float32, device=device)
    N = o3d.core.Tensor(np.asarray(pcd.normals), dtype=o3d.core.float32, device=device)
    
    # Constants to Tensor
    grid_center_t = o3d.core.Tensor(grid_center, dtype=o3d.core.float32, device=device)
    grid_normal_t = o3d.core.Tensor(grid_normal, dtype=o3d.core.float32, device=device)
    
    # Pre-calculate Sensor Culling (independent of light source)
    # Vector from Surface to Sensor Center
    vec_to_sensor = grid_center_t - P
    dot_sensor = (N * vec_to_sensor).sum(dim=1)
    sensor_visible_mask = dot_sensor > 1e-6
    
    heatmap = np.zeros((res_y, res_x), dtype=np.int32)
    all_paths = []
    hit_triangle_indices = set()
    
    # Setup Scene for Occlusion and Triangle ID retrieval (Always needed)
    t_mesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh).to(device)
    scene = o3d.t.geometry.RaycastingScene(device=device)
    scene.add_triangles(t_mesh)
    
    # Pre-calculate Grid Basis on CPU, then move to Tensor
    gn = grid_normal / np.linalg.norm(grid_normal)
    gu = grid_up / np.linalg.norm(grid_up)
    gr = np.cross(gn, gu)
    gr /= np.linalg.norm(gr)
    gtu = np.cross(gr, gn)
    
    gr_t = o3d.core.Tensor(gr, dtype=o3d.core.float32, device=device)
    gtu_t = o3d.core.Tensor(gtu, dtype=o3d.core.float32, device=device)
        
    for source_point in source_points:
        source_point_t = o3d.core.Tensor(source_point, dtype=o3d.core.float32, device=device)
        
        # --- B. Culling (Back-face) ---
        # Vector from Surface to Source
        vec_to_source = source_point_t - P
        # Dot product > 0 means surface faces light
        dot_source = (N * vec_to_source).sum(dim=1)
        
        # Combine with pre-calculated sensor mask
        valid_mask = (dot_source > 1e-6) & sensor_visible_mask
        
        if not valid_mask.any():
            continue

        P_curr = P[valid_mask]
        N_curr = N[valid_mask]
        
        # --- C. Calculate Reflection ---
        # Incoming direction D_in (from Source to P)
        D_in = P_curr - source_point_t
        dist_in = (D_in * D_in).sum(dim=1).sqrt()
        D_in = D_in / (dist_in.reshape((-1, 1)) + 1e-6)
        
        # Reflection R = D_in - 2(D_in . N)N
        dot_ref = (D_in * N_curr).sum(dim=1)
        R = D_in - 2 * dot_ref.reshape((-1, 1)) * N_curr
        
        # --- D. Intersect with Sensor Plane ---
        # Plane: dot(x - Center, Normal) = 0
        # Ray: x = P + t*R
        # t = dot(Center - P, Normal) / dot(R, Normal)
        
        denom = (R * grid_normal_t).sum(dim=1)
        
        # Ray must hit plane from the front (opposing normal)
        valid_denom = denom < -1e-6
        
        t = ((grid_center_t - P_curr) * grid_normal_t).sum(dim=1) / (denom + 1e-10)
        
        # Check t > 0 (hit is forward)
        valid_t = (t > 0) & valid_denom
        
        # Apply mask
        P_curr = P_curr[valid_t]
        R = R[valid_t]
        t = t[valid_t]
        Hit = P_curr + t.reshape((-1, 1)) * R
        
        # --- E. Project to 2D Grid Coords ---
        # Recompute Grid Basis
        # (Already done outside loop)
        
        V = Hit - grid_center_t
        x_local = (V * gr_t).sum(dim=1)
        y_local = (V * gtu_t).sum(dim=1)
        
        # Check bounds
        in_bounds = (x_local.abs() <= grid_width/2) & (y_local.abs() <= grid_height/2)
        
        candidate_P = P_curr[in_bounds]
        candidate_Hit = Hit[in_bounds]
        x_local = x_local[in_bounds]
        y_local = y_local[in_bounds]
        
        # --- F. Visibility Check (Raycasting) ---
        # Convert to numpy for occlusion check logic
        cand_P_np = candidate_P.numpy()
        cand_Hit_np = candidate_Hit.numpy()
        cand_x_np = x_local.numpy()
        cand_y_np = y_local.numpy()

        if check_occlusion and len(cand_P_np) > 0:
            # 1. Source -> P
            dirs_s_p = cand_P_np - source_point
            dists_s_p = np.linalg.norm(dirs_s_p, axis=1)
            dirs_s_p_norm = dirs_s_p / (dists_s_p[:, np.newaxis] + 1e-6)
            
            rays_s_p = np.concatenate([
                np.tile(source_point, (len(cand_P_np), 1)),
                dirs_s_p_norm
            ], axis=1).astype(np.float32)
            
            hits_s_p = scene.cast_rays(o3d.core.Tensor(rays_s_p, device=device))
            t_hit_s_p = hits_s_p['t_hit'].numpy()
            vis_s_p = t_hit_s_p >= (dists_s_p - 1e-3)
            
            # 2. P -> Hit
            dirs_p_hit = cand_Hit_np - cand_P_np
            dists_p_hit = np.linalg.norm(dirs_p_hit, axis=1)
            dirs_p_hit_norm = dirs_p_hit / (dists_p_hit[:, np.newaxis] + 1e-6)
            
            rays_p_hit = np.concatenate([
                cand_P_np + dirs_p_hit_norm * 1e-3, # Offset
                dirs_p_hit_norm
            ], axis=1).astype(np.float32)
            
            hits_p_hit = scene.cast_rays(o3d.core.Tensor(rays_p_hit, device=device))
            t_hit_p_hit = hits_p_hit['t_hit'].numpy()
            vis_p_hit = t_hit_p_hit > (dists_p_hit - 1e-3)
            
            final_mask = vis_s_p & vis_p_hit
        else:
            final_mask = np.ones(len(cand_P_np), dtype=bool)
        
        # Apply final mask to Tensors for efficient Open3D queries
        final_mask_t = o3d.core.Tensor(final_mask, device=device)
        valid_P_t = candidate_P[final_mask_t]
        
        # Convert to Numpy for Heatmap and List operations
        valid_P = valid_P_t.numpy()
        valid_Hit = candidate_Hit[final_mask_t].numpy()
        valid_x = x_local[final_mask_t].numpy()
        valid_y = y_local[final_mask_t].numpy()
        
        # --- G. Map to Heatmap ---
        if len(valid_P) > 0:
            # Map local coords to pixel indices (0 to res-1)
            # x: -w/2 -> 0, +w/2 -> res_x
            col_idx = ((valid_x + grid_width/2) / grid_width * res_x).astype(np.int32)
            # y: +h/2 -> 0, -h/2 -> res_y (Top-down)
            row_idx = ((grid_height/2 - valid_y) / grid_height * res_y).astype(np.int32)
            
            col_idx = np.clip(col_idx, 0, res_x - 1)
            row_idx = np.clip(row_idx, 0, res_y - 1)
            
            np.add.at(heatmap, (row_idx, col_idx), 1)
            
            # Store paths: (Source, Hit, Bounce)
            # Optimized list extension
            all_paths.extend([(source_point, h, p) for h, p in zip(valid_Hit, valid_P)])

            # Identify triangles for valid_P using Closest Point Query (Faster than Raycasting)
            ans = scene.compute_closest_points(valid_P_t)
            prim_ids = ans['primitive_ids'].numpy()
            
            # Filter valid hits (Open3D uses 0xFFFFFFFF for invalid)
            valid_ids = prim_ids[prim_ids != scene.INVALID_ID]
            hit_triangle_indices.update(valid_ids)
    
    print(f"   Confirmed {len(all_paths)} valid paths across all sources.")
    return all_paths, heatmap, hit_triangle_indices


def main():
    t0 = time.time()
    
  
    
    # 1. Load Mesh
    # Load a human mesh (e.g., 'human.obj') or fallback to Armadillo

    # mesh = load_bunny_mesh(target_triangles=target_triangles)

    # mesh_path = "/Users/jiacheng/GitHub/ray-tracing/python/Human.obj"
    mesh_path = "triangulated.obj"

    mesh = o3d.io.read_triangle_mesh(mesh_path)



    # target_tris = 50000
    # mesh_dec = mesh.simplify_quadric_decimation(target_number_of_triangles=target_tris)

    # # Optional cleanup
    # mesh_dec.remove_duplicated_vertices()
    # mesh_dec.remove_duplicated_triangles()
    # mesh_dec.remove_degenerate_triangles()
    # mesh_dec.remove_non_manifold_edges()


    # --- Measure height BEFORE normalization ---
    bbox_original = mesh.get_axis_aligned_bounding_box()
    dimensions_original = bbox_original.get_extent()
    print(f"Original Mesh Dimensions (X, Y, Z): {dimensions_original[0]:.4f}, {dimensions_original[1]:.4f}, {dimensions_original[2]:.4f}")
    print(f"Original Mesh Height (Y-axis): {dimensions_original[1]:.4f} units")


    # import trimesh
    # mesh = trimesh.load("Male.OBJ", process=True)
    # # mesh = mesh.triangulate()
    # mesh.export("triangulated.obj")


    # mesh = o3d.io.read_triangle_mesh(mesh_path)

    # print(mesh)
    # print('Vertices:')
    # print(np.asarray(mesh.vertices))
    # print('Triangles:')
    # print(np.asarray(mesh.triangles))

    # # Alternative: Load as a PointCloud if you only need the geometry
    # pcd = o3d.io.read_point_cloud("Human.obj")
    # o3d.visualization.draw_geometries([pcd])

    # if not mesh.has_vertices():
    #     raise FileNotFoundError(f"File {mesh_path} not found or empty.")
    

    # Normalize mesh (Center and Scale)
    mesh.compute_vertex_normals()
    mesh.translate(-mesh.get_center())
    
    # Scale to approx 1.0 unit size to match scene scale
    bbox = mesh.get_axis_aligned_bounding_box()
    max_extent = bbox.get_max_extent()
    if max_extent > 0:
        mesh.scale(1.0 / max_extent, center=(0, 0, 0))

    # --- Measure height AFTER normalization ---
    bbox_normalized = mesh.get_axis_aligned_bounding_box()
    dimensions_normalized = bbox_normalized.get_extent()
    print(f"\nNormalized Mesh Dimensions (X, Y, Z): {dimensions_normalized[0]:.4f}, {dimensions_normalized[1]:.4f}, {dimensions_normalized[2]:.4f}")
    print(f"Normalized Mesh Height (Y-axis): {dimensions_normalized[1]:.4f} units\n")
    print(f'mesh.triangles={len(mesh.triangles)}')
    # 1 unit: 1.8m 
    # Simplify
    if len(mesh.triangles) > target_triangles:
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=target_triangles)
    
    # # Add a second bunny
    # mesh2 = copy.deepcopy(mesh)
    # R2 = mesh2.get_rotation_matrix_from_xyz((0, -np.pi / 3, 0))
    # mesh2.rotate(R2, center=(0, 0, 0))
    # mesh2.translate([-0.7, 0.0, 0.5])

    # R = mesh.get_rotation_matrix_from_xyz((0, np.pi / 4, 0))
    # mesh.rotate(R, center=(0, 0, 0))
    # mesh.translate([0.7, 0.0, 0.5])
    # mesh += mesh2
    
    # 2. Compute Reflections
    all_paths, total_heatmap, hit_tri_indices = compute_reflections_forward(
        mesh, source_points, grid_center, grid_normal, grid_up, grid_width, grid_height, 
        res_x, res_y, num_samples=num_samples_per_source, check_occlusion=check_occlusion
    )
            
    print(f"Total duration: {time.time() - t0:.4f} seconds")
    filename = f"heatmap_samples_{num_samples_per_source}_tris_{target_triangles}_res_{res_x}x{res_y}.png"
    plot_heatmap(total_heatmap, save_path=filename, title_str=f'Ray Hits per Pixel (samples_per_triangle={num_samples_per_source:.1E})')
    
    # 3. Visualize 3D
    # Paint mesh grey
    mesh.paint_uniform_color([0.5, 0.5, 0.5])
    
    # Paint hit triangles green
    if hit_tri_indices:
        triangles = np.asarray(mesh.triangles)
        hit_tri_array = triangles[list(hit_tri_indices)]
        hit_vertices = np.unique(hit_tri_array)
        colors = np.asarray(mesh.vertex_colors)
        colors[hit_vertices] = [0, 1, 0] # Green
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)

    geometries = [mesh]
    # Add grid points for context
    grid_pts = create_pixel_grid(grid_center, grid_normal, grid_up, grid_width, grid_height, res_x, res_y, verbose=False)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(grid_pts))
    pcd.paint_uniform_color([0, 0, 1])
    geometries.append(pcd)

    # Add Ground Grid
    geometries.append(create_ground_grid(size=5, step=0.5, y_level=-0.5))
    
    # Add Sensor Wireframe
    geometries.append(create_sensor_wireframe(grid_center, grid_normal, grid_up, grid_width, grid_height))
    
    # Add Coordinate Frame
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]))

    # A. Draw Light Sources (Black Spheres)
    for sp in source_points:
        light_geo = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        light_geo.translate(sp)
        # light_geo.paint_uniform_color([1, 0.4, 0])
        light_geo.paint_uniform_color([0, 0, 0])
        geometries.append(light_geo)

    # Add sample rays (random subset)
    if all_paths:
        vis_indices = np.random.choice(len(all_paths), min(len(all_paths), 100), replace=False)
        points, lines, colors = [], [], []
        for i in vis_indices:
            src, hit, bounce = all_paths[i]
            base = len(points)
            points.extend([src, bounce, hit])
            lines.extend([[base, base+1], [base+1, base+2]])
            colors.extend([[1, 0.8, 0], [1, 0, 0]]) # Yellow -> Red
            
        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(points)
        ls.lines = o3d.utility.Vector2iVector(lines)
        ls.colors = o3d.utility.Vector3dVector(colors)
        geometries.append(ls)
        
    vis=o3d.visualization.draw_geometries(geometries, window_name="Forward Ray Tracing")

    def close_on_enter(vis):
        vis.close()
        return False

    vis.register_key_callback(13, close_on_enter)   # try 13, or 257/335 if needed

if __name__ == "__main__":
    main()
