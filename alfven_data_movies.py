import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import cm
import os
import subprocess
import platform
import sys
from PIL import Image  # For alternative GIF creation

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
        # Try to get FFmpeg version
        result = subprocess.run(['ffmpeg', '-version'], 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               text=True)
        
        # If successful, print the version
        if result.returncode == 0:
            print(f"FFmpeg is available: {result.stdout.split('\\n')[0]}")
            return True
        else:
            print("FFmpeg is installed but returned an error.")
            print(f"Error: {result.stderr}")
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

def try_ffmpeg_gif(frame_files, output_filename, fps=10):
    """
    Try to use FFmpeg to create a GIF from the frames.
    FFmpeg generally creates better quality GIFs than PIL.
    """
    try:
        print("Attempting to create GIF using FFmpeg...")
        
        # Create palette for better quality
        palette_file = output_filename + "_palette.png"
        
        # Create palette
        palette_cmd = [
            'ffmpeg', '-y',
            '-i', frame_files[0].replace('0000', '%04d'),
            '-vf', f"fps={fps},scale=320:-1:flags=lanczos,palettegen",
            palette_file
        ]
        
        subprocess.run(palette_cmd, check=True, 
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
        
        # Create GIF using the palette
        gif_cmd = [
            'ffmpeg', '-y',
            '-i', frame_files[0].replace('0000', '%04d'),
            '-i', palette_file,
            '-filter_complex', f"fps={fps},scale=320:-1:flags=lanczos[x];[x][1:v]paletteuse",
            output_filename
        ]
        
        subprocess.run(gif_cmd, check=True, 
                     stdout=subprocess.PIPE, 
                     stderr=subprocess.PIPE)
        
        # Remove the palette file
        os.remove(palette_file)
        
        print(f"GIF created successfully with FFmpeg: {output_filename}")
        return True
        
    except Exception as e:
        print(f"Failed to create GIF with FFmpeg: {e}")
        return False

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
                
                cmd = [
                    'ffmpeg', '-y',
                    '-framerate', str(fps),
                    '-i', frame_files[0].replace('0000', '%04d'),
                    *codec,
                    out_file
                ]
                
                result = subprocess.run(cmd, 
                                     stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE,
                                     text=True,
                                     check=True)
                
                print(f"Video created successfully: {out_file}")
                return out_file
                
            except subprocess.CalledProcessError as e:
                print(f"Failed with codec option {i+1}: {e}")
                print(f"FFmpeg stderr: {e.stderr}")
                continue
        
        raise Exception("All codec options failed")
        
    except Exception as e:
        print(f"Failed to create video with FFmpeg: {e}")
        return None

def create_vector_field_frames(data, field_names, shape, z_pos, output_dir, 
                              title=None, start_time=0, end_time=None, 
                              step=1, skip=3):
    """
    Create frames for a vector field animation.
    """
    if len(field_names) < 2:
        raise ValueError("Need at least 2 components for vector plot")
    
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # Set end_time if not specified
    if end_time is None:
        end_time = n_timesteps - 1
    else:
        end_time = min(end_time, n_timesteps - 1)
    
    # Create coordinate grid for vectors
    Y, X = np.meshgrid(
        np.arange(0, n_y, skip),
        np.arange(0, n_x, skip),
        indexing='ij'
    )
    
    # Find global scale factor for consistent vector scaling
    # We'll sample a few timesteps to estimate the maximum magnitude
    sample_times = np.linspace(start_time, end_time, min(10, (end_time-start_time)//step+1), dtype=int)
    max_mag = 0
    for t in sample_times:
        x_comp = data[field_names[0]][t, z_pos, ::skip, ::skip]
        y_comp = data[field_names[1]][t, z_pos, ::skip, ::skip]
        mag = np.sqrt(x_comp**2 + y_comp**2)
        max_mag = max(max_mag, np.max(mag))
    
    # Add safety factor
    scale = max_mag * 1.5
    
    # Create output directory if it doesn't exist
    frames_dir = os.path.join(output_dir, f"vector_{field_names[0]}_{field_names[1]}_z{z_pos}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Creating vector field frames for {field_names[0]},{field_names[1]} (Z={z_pos})...")
    
    # Create each frame as an image
    frame_files = []
    for i, time_idx in enumerate(range(start_time, end_time+1, step)):
        # Create figure and axes
        fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
        
        # Get vector components for this timestep
        x_comp = data[field_names[0]][time_idx, z_pos, ::skip, ::skip]
        y_comp = data[field_names[1]][time_idx, z_pos, ::skip, ::skip]
        
        # Calculate magnitudes
        magnitudes = np.sqrt(x_comp**2 + y_comp**2)
        
        # Normalize vectors
        x_comp_norm = x_comp / (scale + 1e-10)
        y_comp_norm = y_comp / (scale + 1e-10)
        
        # Create quiver plot
        q = ax.quiver(X, Y, x_comp_norm, y_comp_norm, magnitudes, 
                     cmap='viridis', pivot='mid', scale=1.0)
        
        # Add colorbar
        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label('Magnitude')
        
        # Set labels
        if title:
            ax.set_title(f"{title} (Z={z_pos}, Time={time_idx})")
        else:
            ax.set_title(f"Vector Field (Z={z_pos}, Time={time_idx})")
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

def create_exb_drift_frames(b_data, e_data, shape, z_pos, output_dir,
                          title="ExB Drift Velocity", start_time=0, 
                          end_time=None, step=1, skip=3):
    """
    Create frames for an ExB drift animation.
    """
    # We assume the shape is (timesteps, z_positions, y, x)
    n_timesteps, n_z, n_y, n_x = shape
    
    # Set end_time if not specified
    if end_time is None:
        end_time = n_timesteps - 1
    else:
        end_time = min(end_time, n_timesteps - 1)
    
    # Create coordinate grid for vectors
    Y, X = np.meshgrid(
        np.arange(0, n_y, skip),
        np.arange(0, n_x, skip),
        indexing='ij'
    )
    
    # Find global scale factor for ExB drift
    sample_times = np.linspace(start_time, end_time, min(10, (end_time-start_time)//step+1), dtype=int)
    max_mag = 0
    for t in sample_times:
        # Get field components
        ex = e_data['DATA_X'][t, z_pos, ::skip, ::skip]
        ey = e_data['DATA_Y'][t, z_pos, ::skip, ::skip]
        ez = e_data['DATA_Z'][t, z_pos, ::skip, ::skip]
        
        bx = b_data['DATA_X'][t, z_pos, ::skip, ::skip]
        by = b_data['DATA_Y'][t, z_pos, ::skip, ::skip]
        bz = b_data['DATA_Z'][t, z_pos, ::skip, ::skip]
        
        # Calculate B magnitude
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate ExB drift
        exb_x = (ey * bz - ez * by) / (b_mag**2 + 1e-10)
        exb_y = (ez * bx - ex * bz) / (b_mag**2 + 1e-10)
        
        # Calculate magnitude
        mag = np.sqrt(exb_x**2 + exb_y**2)
        max_mag = max(max_mag, np.max(mag))
    
    # Add safety factor
    scale = max_mag * 1.5
    
    # Create output directory if it doesn't exist
    frames_dir = os.path.join(output_dir, f"exb_drift_z{z_pos}_frames")
    os.makedirs(frames_dir, exist_ok=True)
    
    print(f"Creating ExB drift frames (Z={z_pos})...")
    
    # Create each frame as an image
    frame_files = []
    for i, time_idx in enumerate(range(start_time, end_time+1, step)):
        # Create figure and axes
        fig, ax = plt.figure(figsize=(8, 6)), plt.gca()
        
        # Get field components
        ex = e_data['DATA_X'][time_idx, z_pos, ::skip, ::skip]
        ey = e_data['DATA_Y'][time_idx, z_pos, ::skip, ::skip]
        ez = e_data['DATA_Z'][time_idx, z_pos, ::skip, ::skip]
        
        bx = b_data['DATA_X'][time_idx, z_pos, ::skip, ::skip]
        by = b_data['DATA_Y'][time_idx, z_pos, ::skip, ::skip]
        bz = b_data['DATA_Z'][time_idx, z_pos, ::skip, ::skip]
        
        # Calculate B magnitude
        b_mag = np.sqrt(bx**2 + by**2 + bz**2)
        
        # Calculate ExB drift
        exb_x = (ey * bz - ez * by) / (b_mag**2 + 1e-10)
        exb_y = (ez * bx - ex * bz) / (b_mag**2 + 1e-10)
        
        # Calculate magnitude
        exb_mag = np.sqrt(exb_x**2 + exb_y**2)
        
        # Normalize vectors
        exb_x_norm = exb_x / (scale + 1e-10)
        exb_y_norm = exb_y / (scale + 1e-10)
        
        # Create quiver plot
        q = ax.quiver(X, Y, exb_x_norm, exb_y_norm, exb_mag, 
                     cmap='viridis', pivot='mid', scale=1.0)
        
        # Add colorbar
        cbar = plt.colorbar(q, ax=ax)
        cbar.set_label('ExB Drift Magnitude')
        
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

def main():
    """Main function to create animations from HDF5 data."""
    # File paths
    b_field_file = "b37-40.hdf5"
    e_field_file = "e37-40.hdf5"
    
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
        end_time = min(50, b_shape[0]-1)  # Limit to 50 frames for this example
        step = 2  # Skip every other frame
        
        print(f"\nCreating animations from timestep {start_time} to {end_time} with step {step}")
        
        # Create component animations
        for field, title in [
            ('DATA_Z', 'Bz Component (Gauss)'),  # B field z-component
        ]:
            # Create frames
            frames_dir_path, frame_files = create_frames_for_gif(
                b_data,
                field,
                b_shape,
                z_pos=0,  # First z position
                output_dir=frames_dir,
                title=title,
                start_time=start_time,
                end_time=end_time,
                step=step
            )
            
            # Create GIF using PIL (always works)
            gif_file = os.path.join(movies_dir, f"b_field_{field.lower()}.gif")
            frames_to_gif(frame_files, gif_file, fps=fps)
            
            # Try to create video with FFmpeg if available
            if has_ffmpeg:
                mp4_file = os.path.join(movies_dir, f"b_field_{field.lower()}.mp4")
                video_file = try_ffmpeg_mp4(frame_files, mp4_file, fps=fps)
                
                if not video_file:
                    # Try FFmpeg-based GIF as fallback
                    try_ffmpeg_gif(frame_files, gif_file, fps=fps)
            
            print(f"Animation for B field {field} created successfully!")
            
        # Create vector field animation for magnetic field
        print("\nCreating magnetic field vector animation...")
        frames_dir_path, frame_files = create_vector_field_frames(
            b_data,
            ['DATA_X', 'DATA_Y'],
            b_shape,
            z_pos=0,
            output_dir=frames_dir,
            title="Magnetic Field Vectors (Gauss)",
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        
        # Create GIF using PIL
        gif_file = os.path.join(movies_dir, "b_field_vectors.gif")
        frames_to_gif(frame_files, gif_file, fps=fps)
        
        # Try to create video with FFmpeg if available
        if has_ffmpeg:
            mp4_file = os.path.join(movies_dir, "b_field_vectors.mp4")
            try_ffmpeg_mp4(frame_files, mp4_file, fps=fps)
        
        # Create ExB drift animation
        print("\nCreating ExB drift animation...")
        frames_dir_path, frame_files = create_exb_drift_frames(
            b_data,
            e_data,
            b_shape,
            z_pos=0,
            output_dir=frames_dir,
            title="ExB Drift Velocity",
            start_time=start_time,
            end_time=end_time,
            step=step
        )
        
        # Create GIF using PIL
        gif_file = os.path.join(movies_dir, "exb_drift.gif")
        frames_to_gif(frame_files, gif_file, fps=fps)
        
        # Try to create video with FFmpeg if available
        if has_ffmpeg:
            mp4_file = os.path.join(movies_dir, "exb_drift.mp4")
            try_ffmpeg_mp4(frame_files, mp4_file, fps=fps)
        
        print("\nAll animations created successfully!")
        print(f"GIF animations are in the '{movies_dir}' directory.")
        if has_ffmpeg:
            print(f"MP4 videos are also in the '{movies_dir}' directory if FFmpeg succeeded.")
        print("\nTo view these in VS Code, install an extension like 'Media Preview'.")
        
    except Exception as e:
        print(f"\nError creating animations: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()