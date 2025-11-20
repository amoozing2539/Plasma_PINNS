import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
import subprocess
import platform
from PIL import Image

def read_compound_dataset(filename, dataset_path):
    """
    Read a compound dataset from an HDF5 file correctly.
    """
    with h5py.File(filename, 'r') as f:
        if dataset_path not in f:
            raise KeyError(f"Dataset path {dataset_path} not found in {filename}")
        
        dataset = f[dataset_path]
        
        # Handle compound dataset
        if not hasattr(dataset, 'dtype') or not dataset.dtype.names:
            raise ValueError(f"Dataset {dataset_path} is not a compound dataset")
            
        # Get field names
        field_names = dataset.dtype.names
        
        # Create a dictionary to hold the data for each field
        data = {}
        
        # Read the dataset field by field
        for field in field_names:
            # This correctly extracts a single field from the compound dataset
            field_data = dataset[field][()]
            data[field] = field_data
            
        return data, dataset.shape

def check_ffmpeg():
    """
    Check if FFmpeg is installed and working correctly.
    Returns True if FFmpeg is available, False otherwise.
    """
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        if result.returncode == 0:
            line = result.stdout.split('\n')[0]
            print(f"FFmpeg is available: {line}")
            #print(f"FFmpeg is available: {result.stdout.split('\\n')[0]}")
            return True
        else:
            print("FFmpeg is installed but returned an error.")
            return False
            
    except FileNotFoundError:
        print("FFmpeg is not installed or not in the PATH.")
        return False

def create_frames_for_gif(data, field_name, shape, z_pos, output_dir, 
                         title=None, cmap='viridis',
                         start_time=0, end_time=None, step=1):
    """
    Create individual frames for later conversion to GIF or video.
    """
    # Get field data
    field_data = data[field_name]
    
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # Set end_time if not specified
    if end_time is None:
        end_time = n_timesteps - 1
    else:
        end_time = min(end_time, n_timesteps - 1)
    
    # Find global min/max for consistent colormap
    vmin = np.min(field_data[start_time:end_time+1:step, z_pos])
    vmax = np.max(field_data[start_time:end_time+1:step, z_pos])
    
    # Create output directory if it doesn't exist
    frames_dir = os.path.join(output_dir, f"{field_name}_z{z_pos}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Creating frames for {field_name} (Z={z_pos})...")
    
    # Create each frame as an image
    frame_files = []
    for i, time_idx in enumerate(range(start_time, end_time+1, step)):
        # Create figure and axes
        fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
        
        # Get data for this timestep
        frame = field_data[time_idx, z_pos]
        
        # Plot as image
        im = ax.imshow(frame, origin='lower', cmap=cmap, 
                     interpolation='none', aspect='equal',
                     vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(field_name)
        
        # Set labels
        if title:
            ax.set_title(f"{title} (Z={z_pos}, Time={time_idx})")
        else:
            ax.set_title(f"{field_name} (Z={z_pos}, Time={time_idx})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Save this frame as an image
        frame_file = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        frame_files.append(frame_file)
        
        # Print progress occasionally
        if i % 10 == 0:
            print(f"  Created frame {i}/{(end_time-start_time)//step + 1}")
    
    return frames_dir, frame_files

def create_exb_drift_magnitude_frames(b_data, e_data, shape, z_pos, output_dir,
                                    title="ExB Drift Magnitude",
                                    start_time=0, end_time=None, step=1,
                                    cmap='plasma'):
    """
    Create frames showing the magnitude of ExB drift velocity as a scalar field.
    """
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # Set end_time if not specified
    if end_time is None:
        end_time = n_timesteps - 1
    else:
        end_time = min(end_time, n_timesteps - 1)
    
    # Create output directory if it doesn't exist
    frames_dir = os.path.join(output_dir, f"exb_magnitude_z{z_pos}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Creating ExB drift magnitude frames (Z={z_pos})...")
    
    # Calculate ExB drift magnitudes for the time range to get global min/max
    magnitudes = []
    for time_idx in range(start_time, end_time+1, max(1, (end_time-start_time)//10)):
        # Get field components
        ex = e_data['DATA_X'][time_idx, z_pos]
        ey = e_data['DATA_Y'][time_idx, z_pos]
        ez = e_data['DATA_Z'][time_idx, z_pos]
        
        bx = b_data['DATA_X'][time_idx, z_pos]
        by = b_data['DATA_Y'][time_idx, z_pos]
        bz = b_data['DATA_Z'][time_idx, z_pos]
        
        # Calculate B magnitude
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate ExB drift velocity components
        exb_x = (ey * bz - ez * by) / (b_mag**2 + 1e-10)
        exb_y = (ez * bx - ex * bz) / (b_mag**2 + 1e-10)
        exb_z = (ex * by - ey * bx) / (b_mag**2 + 1e-10)
        
        # Calculate magnitude
        exb_mag = np.sqrt(exb_x**2 + exb_y**2 + exb_z**2)
        magnitudes.append(exb_mag)
    
    # Find global min/max for consistent colormap
    vmin = min(np.min(mag) for mag in magnitudes)
    vmax = max(np.max(mag) for mag in magnitudes)
    
    # Create each frame as an image
    frame_files = []
    for i, time_idx in enumerate(range(start_time, end_time+1, step)):
        # Create figure and axes
        fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
        
        # Get field components
        ex = e_data['DATA_X'][time_idx, z_pos]
        ey = e_data['DATA_Y'][time_idx, z_pos]
        ez = e_data['DATA_Z'][time_idx, z_pos]
        
        bx = b_data['DATA_X'][time_idx, z_pos]
        by = b_data['DATA_Y'][time_idx, z_pos]
        bz = b_data['DATA_Z'][time_idx, z_pos]
        
        # Calculate B magnitude
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate ExB drift velocity components
        exb_x = (ey * bz - ez * by) / (b_mag**2 + 1e-10)
        exb_y = (ez * bx - ex * bz) / (b_mag**2 + 1e-10)
        exb_z = (ex * by - ey * bx) / (b_mag**2 + 1e-10)
        
        # Calculate magnitude
        exb_mag = np.sqrt(exb_x**2 + exb_y**2 + exb_z**2)
        
        # Plot as image
        im = ax.imshow(exb_mag, origin='lower', cmap=cmap, 
                     interpolation='none', aspect='equal',
                     vmin=vmin, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude')
        
        # Set labels
        ax.set_title(f"{title} (Z={z_pos}, Time={time_idx})")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        
        # Save this frame as an image
        frame_file = os.path.join(frames_dir, f"frame_{i:04d}.png")
        plt.savefig(frame_file, dpi=100, bbox_inches='tight')
        plt.close(fig)
        
        frame_files.append(frame_file)
        
        # Print progress occasionally
        if i % 10 == 0:
            print(f"  Created frame {i}/{(end_time-start_time)//step + 1}")
    
    return frames_dir, frame_files

def create_3d_field_frames(data, field_names, shape, output_dir, 
                         title=None, start_time=0, end_time=None, step=1, skip=2):
    """
    Create frames for a 3D visualization of the field.
    """
    if len(field_names) != 3:
        raise ValueError("Need exactly 3 components for 3D visualization")
    
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # Set end_time if not specified
    if end_time is None:
        end_time = n_timesteps - 1
    else:
        end_time = min(end_time, n_timesteps - 1)
    
    # Create output directory if it doesn't exist
    frames_dir = os.path.join(output_dir, f"3d_{field_names[0]}_{field_names[1]}_{field_names[2]}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Creating 3D field frames...")
    
    # Create a grid of points (more sparse for 3D visualization)
    # We'll use every 'skip' points in each dimension
    Z, Y, X = np.meshgrid(
        np.arange(n_z),
        np.arange(0, n_y, skip),
        np.arange(0, n_x, skip),
        indexing='ij'
    )
    
    # Flatten for easier plotting
    x_pos = X.flatten()
    y_pos = Y.flatten()
    z_pos = Z.flatten()
    
    # Find global scale for consistent vector scaling
    max_mag = 0
    for t in range(start_time, end_time+1, max(1, (end_time-start_time)//5)):
        # Get vector components for all z positions
        x_comp = np.array([data[field_names[0]][t, z, ::skip, ::skip].flatten() for z in range(n_z)])
        y_comp = np.array([data[field_names[1]][t, z, ::skip, ::skip].flatten() for z in range(n_z)])
        z_comp = np.array([data[field_names[2]][t, z, ::skip, ::skip].flatten() for z in range(n_z)])
        
        # Reshaping for all points
        x_comp = x_comp.flatten()
        y_comp = y_comp.flatten()
        z_comp = z_comp.flatten()
        
        # Calculate magnitudes
        mag = np.sqrt(x_comp**2 + y_comp**2 + z_comp**2)
        max_mag = max(max_mag, np.max(mag))
    
    scale = max_mag * 2.0  # Adjust this factor for clarity
    
    # Create an array of viewing angles for rotation
    # This creates a 360-degree rotation with 36 frames (10 degrees per frame)
    elevation = 30  # Fixed elevation
    azimuths = np.linspace(0, 360, 36, endpoint=False)
    
    # Create each frame as an image, using different viewing angles
    frame_files = []
    
    # For each timestep
    for i, time_idx in enumerate(range(start_time, end_time+1, step)):
        # Get vector components for all z positions
        x_comp = np.array([data[field_names[0]][time_idx, z, ::skip, ::skip].flatten() for z in range(n_z)])
        y_comp = np.array([data[field_names[1]][time_idx, z, ::skip, ::skip].flatten() for z in range(n_z)])
        z_comp = np.array([data[field_names[2]][time_idx, z, ::skip, ::skip].flatten() for z in range(n_z)])
        
        # Reshaping for all points
        x_comp = x_comp.flatten()
        y_comp = y_comp.flatten()
        z_comp = z_comp.flatten()
        
        # Calculate magnitudes for coloring
        mag = np.sqrt(x_comp**2 + y_comp**2 + z_comp**2)
        
        # Normalize vectors for better visualization
        u_norm = x_comp / (scale + 1e-10)
        v_norm = y_comp / (scale + 1e-10)
        w_norm = z_comp / (scale + 1e-10)
        
        # For each viewing angle (this creates a rotating view)
        for j, azimuth in enumerate(azimuths):
            # Create figure and 3D axes
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot the vector field
            q = ax.quiver(x_pos, y_pos, z_pos, u_norm, v_norm, w_norm,
                         length=0.5, normalize=False, 
                         colors=plt.cm.viridis(mag/max_mag))
            
            # Add axes labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set the viewing angle
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Set title
            if title:
                ax.set_title(f"{title} (Time={time_idx}, Angle={int(azimuth)}°)")
            else:
                ax.set_title(f"3D Field Visualization (Time={time_idx}, Angle={int(azimuth)}°)")
            
            # Save frame
            frame_file = os.path.join(frames_dir, f"frame_t{time_idx:04d}_a{j:03d}.png")
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            frame_files.append(frame_file)
            
        # Print progress
        print(f"  Created 3D frames for timestep {time_idx} ({i+1}/{(end_time-start_time)//step + 1})")
    
    # Sort frames by time first, then by angle
    frame_files.sort()
    
    return frames_dir, frame_files

def create_3d_all_z_frames(data, field_name, shape, output_dir, 
                         title=None, cmap='viridis',
                         start_time=0, end_time=None, step=1):
    """
    Create 3D visualization showing all Z planes simultaneously.
    """
    # Get field data
    field_data = data[field_name]
    
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # Set end_time if not specified
    if end_time is None:
        end_time = n_timesteps - 1
    else:
        end_time = min(end_time, n_timesteps - 1)
    
    # Create output directory if it doesn't exist
    frames_dir = os.path.join(output_dir, f"3d_slices_{field_name}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Creating 3D slice visualization frames for {field_name}...")
    
    # Find global min/max for consistent colormap
    vmin = np.min(field_data[start_time:end_time+1:step])
    vmax = np.max(field_data[start_time:end_time+1:step])
    
    # Create an array of viewing angles for rotation
    # This creates a 360-degree rotation with 36 frames (10 degrees per frame)
    elevation = 30  # Fixed elevation
    azimuths = np.linspace(0, 360, 36, endpoint=False)
    
    # Create coordinate arrays for plotting
    x = np.arange(n_x)
    y = np.arange(n_y)
    X, Y = np.meshgrid(x, y)
    
    # Create each frame as an image, using different viewing angles
    frame_files = []
    
    # For each timestep
    for i, time_idx in enumerate(range(start_time, end_time+1, step)):
        # For each viewing angle (this creates a rotating view)
        for j, azimuth in enumerate(azimuths):
            # Create figure and 3D axes
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # Plot each Z plane
            for z in range(n_z):
                # Get data for this Z plane
                Z_plane = np.ones_like(X) * z
                
                # Get field data
                C = field_data[time_idx, z]
                
                # Plot as a surface
                ax.plot_surface(X, Y, Z_plane, facecolors=plt.cm.get_cmap(cmap)((C-vmin)/(vmax-vmin)),
                               rstride=1, cstride=1, alpha=0.7, shade=False)
            
            # Add axes labels
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            
            # Set the viewing angle
            ax.view_init(elev=elevation, azim=azimuth)
            
            # Set axis limits
            ax.set_xlim(0, n_x-1)
            ax.set_ylim(0, n_y-1)
            ax.set_zlim(0, n_z-1)
            
            # Set title
            if title:
                ax.set_title(f"{title} (Time={time_idx}, Angle={int(azimuth)}°)")
            else:
                ax.set_title(f"3D {field_name} Visualization (Time={time_idx}, Angle={int(azimuth)}°)")
            
            # Add a colorbar
            m = cm.ScalarMappable(cmap=cmap)
            m.set_array(field_data[time_idx])
            m.set_clim(vmin, vmax)
            cbar = plt.colorbar(m, ax=ax, shrink=0.6)
            cbar.set_label(field_name)
            
            # Save frame
            frame_file = os.path.join(frames_dir, f"frame_t{time_idx:04d}_a{j:03d}.png")
            plt.savefig(frame_file, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            frame_files.append(frame_file)
            
        # Print progress
        print(f"  Created 3D slice frames for timestep {time_idx} ({i+1}/{(end_time-start_time)//step + 1})")
    
    # Sort frames by time first, then by angle
    frame_files.sort()
    
    return frames_dir, frame_files

def frames_to_gif(frame_files, output_filename, fps=10):
    """
    Convert a sequence of image frames to a GIF animation.
    Uses PIL which doesn't require external dependencies.
    """
    print(f"Creating GIF from {len(frame_files)} frames...")
    
    # Load all images
    images = [Image.open(f) for f in frame_files]
    
    # Calculate duration in milliseconds
    duration = int(1000 / fps)
    
    # Save as GIF
    images[0].save(
        output_filename,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=duration,
        loop=0
    )
    
    print(f"GIF saved to {output_filename}")
    
    return output_filename

def try_ffmpeg_mp4(frame_files, output_filename, fps=10):
    """
    Try to create an MP4 video using FFmpeg directly.
    """
    try:
        print("Attempting to create MP4 using FFmpeg directly...")
        
        # Different options to try if one fails
        codec_options = [
            # Option 1: libx264 with yuv420p (most compatible)
            ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '23'],
            
            # Option 2: MPEG-4
            ['-c:v', 'mpeg4', '-q:v', '3'],
            
            # Option 3: VP9 (for WebM)
            ['-c:v', 'libvpx-vp9', '-crf', '30', '-b:v', '0']
        ]
        
        # File extensions for each codec
        extensions = ['.mp4', '.mp4', '.webm']
        
        # Try each option
        for i, codec in enumerate(codec_options):
            try:
                out_file = output_filename.rsplit('.', 1)[0] + extensions[i]
                
                # For frame files with consistent naming pattern
                if len(frame_files) > 1 and os.path.basename(frame_files[0]).startswith("frame_") and \
                   os.path.basename(frame_files[1]).startswith("frame_"):
                    # Standard sequential frames
                    cmd = [
                        'ffmpeg', '-y',
                        '-framerate', str(fps),
                        '-i', os.path.join(os.path.dirname(frame_files[0]), "frame_%04d.png"),
                        *codec,
                        out_file
                    ]
                elif len(frame_files) > 1 and "frame_t" in os.path.basename(frame_files[0]):
                    # 3D rotation frames with time and angle components
                    pattern = os.path.join(os.path.dirname(frame_files[0]), "frame_t%04d_a%03d.png")
                    cmd = [
                        'ffmpeg', '-y',
                        '-framerate', str(fps*2),  # Faster for rotations
                        '-i', pattern,
                        *codec,
                        out_file
                    ]
                else:
                    # Fall back to a file list approach
                    temp_list_file = "temp_file_list.txt"
                    with open(temp_list_file, 'w') as f:
                        for frame in frame_files:
                            f.write(f"file '{frame}'\n")
                    
                    cmd = [
                        'ffmpeg', '-y',
                        '-r', str(fps),
                        '-f', 'concat',
                        '-safe', '0',
                        '-i', temp_list_file,
                        *codec,
                        out_file
                    ]
                
                result = subprocess.run(cmd, 
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True,
                                     check=True)
                
                # Clean up temporary file if it was created
                if os.path.exists("temp_file_list.txt"):
                    os.remove("temp_file_list.txt")
                
                print(f"Video created successfully: {out_file}")
                return out_file
                
            except subprocess.CalledProcessError as e:
                print(f"Failed with codec option {i+1}: {e}")
                print(f"FFmpeg stderr: {e.stderr}")
                
                # Clean up temporary file if it was created
                if os.path.exists("temp_file_list.txt"):
                    os.remove("temp_file_list.txt")
                    
                continue
        
        raise Exception("All codec options failed")
        
    except Exception as e:
        print(f"Failed to create video with FFmpeg: {e}")
        return None

def main():
    """Main function to create animations from HDF5 data."""
    # Path
    path = '/home/stsoukalas/shared/data/LAPD_ALfven_2025-02/'
    
    # File paths
    b_field_file = f"{path}b37-40.hdf5"
    e_field_file = f"{path}e37-40.hdf5"
    
    # Check if the files exist
    if not os.path.exists(b_field_file):
        print(f"ERROR: Magnetic field file '{b_field_file}' not found")
        return
    
    if not os.path.exists(e_field_file):
        print(f"ERROR: Electric field file '{e_field_file}' not found")
        return
    
    # Create output directories
    frames_dir = "frames"
    movies_dir = "movies"
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(movies_dir, exist_ok=True)
    
    # Check if FFmpeg is available
    has_ffmpeg = check_ffmpeg()
    print(f"FFmpeg available: {has_ffmpeg}")
    
    try:
        # Read magnetic field data
        print("\nReading magnetic field data...")
        b_data, b_shape = read_compound_dataset(b_field_file, "B vectors/Bvec")
        print(f"B field shape: {b_shape}")
        
        # Read electric field data
        print("\nReading electric field data...")
        e_data, e_shape = read_compound_dataset(e_field_file, "E vectors/Evec")
        print(f"E field shape: {e_shape}")
        
        # Animation parameters
        fps = 10
        
        # Time parameters - adjust these to control animation length
        start_time = 0
        end_time = min(20, b_shape[0]-1)  # Limit to 20 frames for this example
        step = 2  # Skip every other frame
        
        print(f"\nCreating animations from timestep {start_time} to {end_time} with step {step}")
        
        # 1. Create ExB drift magnitude movie
        print("\nCreating ExB drift magnitude animation...")
        frames_dir_path, frame_files = create_exb_drift_magnitude_frames(
            b_data,
            e_data,
            b_shape,
            z_pos=0,  # First z position
            output_dir=frames_dir,
            title="ExB Drift Magnitude",
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        
        # Create GIF
        gif_file = os.path.join(movies_dir, "exb_drift_magnitude.gif")
        frames_to_gif(frame_files, gif_file, fps=fps)
        
        # Try to create video with FFmpeg if available
        if has_ffmpeg:
            mp4_file = os.path.join(movies_dir, "exb_drift_magnitude.mp4")
            try_ffmpeg_mp4(frame_files, mp4_file, fps=fps)
        
        # 2. Create 3D vector field visualization for magnetic field
        print("\nCreating 3D magnetic field visualization...")
        frames_dir_path, frame_files = create_3d_field_frames(
            b_data,
            ['DATA_X', 'DATA_Y', 'DATA_Z'],
            b_shape,
            output_dir=frames_dir,
            title="3D Magnetic Field",
            start_time=start_time,
            end_time=end_time,
            step=step*2  # Use larger step for 3D to reduce processing time
        )
        
        # Create GIF
        gif_file = os.path.join(movies_dir, "b_field_3d.gif")
        frames_to_gif(frame_files, gif_file, fps=fps*2)  # Higher FPS for smooth rotation
        
        # Try to create video with FFmpeg if available
        if has_ffmpeg:
            mp4_file = os.path.join(movies_dir, "b_field_3d.mp4")
            try_ffmpeg_mp4(frame_files, mp4_file, fps=fps*2)
        
        # 3. Create 3D visualization showing all Z planes for magnetic field (Bz component)
        print("\nCreating 3D slice visualization for Bz...")
        frames_dir_path, frame_files = create_3d_all_z_frames(
            b_data,
            'DATA_Z',
            b_shape,
            output_dir=frames_dir,
            title="Bz Component (Gauss)",
            start_time=start_time,
            end_time=end_time,
            step=step*2  # Use larger step for 3D to reduce processing time
        )
        
        # Create GIF
        gif_file = os.path.join(movies_dir, "bz_3d_slices.gif")
        frames_to_gif(frame_files, gif_file, fps=fps*2)  # Higher FPS for smooth rotation
        
        # Try to create video with FFmpeg if available
        if has_ffmpeg:
            mp4_file = os.path.join(movies_dir, "bz_3d_slices.mp4")
            try_ffmpeg_mp4(frame_files, mp4_file, fps=fps*2)
        
        # 4. Create 3D vector field visualization for electric field
        print("\nCreating 3D electric field visualization...")
        frames_dir_path, frame_files = create_3d_field_frames(
            e_data,
            ['DATA_X', 'DATA_Y', 'DATA_Z'],
            e_shape,
            output_dir=frames_dir,
            title="3D Electric Field (V/cm)",
            start_time=start_time,
            end_time=end_time,
            step=step*2  # Use larger step for 3D to reduce processing time
        )
        
        # Create GIF
        gif_file = os.path.join(movies_dir, "e_field_3d.gif")
        frames_to_gif(frame_files, gif_file, fps=fps*2)  # Higher FPS for smooth rotation
        
        # Try to create video with FFmpeg if available
        if has_ffmpeg:
            mp4_file = os.path.join(movies_dir, "e_field_3d.mp4")
            try_ffmpeg_mp4(frame_files, mp4_file, fps=fps*2)
        
        print("\nAll requested animations created successfully!")
        print(f"GIF animations are in the '{movies_dir}' directory.")
        if has_ffmpeg:
            print(f"MP4 videos are also in the '{movies_dir}' directory if FFmpeg succeeded.")
        
    except Exception as e:
        print(f"\nError creating animations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()