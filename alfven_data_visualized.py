import h5py 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

##read the hd5 file####
def read_hdf5_maya_data(file_path, data_group_name, parameter_indices=None, time_index=0):
    """
    Read data from HDF5 file following the Maya visualization structure.
    
    Parameters:
    -----------
    file_path : str
        Path to the HDF5 file
    data_group_name : str
        Name of the data group to read (e.g., 'Bdot vectors', 'density')
    parameter_indices : list or None
        List of parameter indices [p1_idx, p2_idx, p3_idx, p4_idx] or None
    time_index : int
        Index of the timestep to read
        
    Returns:
    --------
    data : ndarray
        3D volume data at specified parameter indices and time
    spatial_grid : ndarray
        Spatial grid coordinates (x, y, z) for each data point
    metadata : dict
        Dictionary containing metadata about the dataset
    """
    with h5py.File(file_path, 'r') as f:
        # Read spatial grid
        spatial_grid = f['Spatial grid'][:]
        
        # Get metadata
        metadata = {}
        
        # Check if data group exists
        if data_group_name not in f:
            raise ValueError(f"Data group '{data_group_name}' not found in file")
            
        # Get data group attributes
        group = f[data_group_name]
        for attr in ['Name', 'Symbol', 'Units', 'Type']:
            if attr in group.attrs:
                metadata[attr.lower()] = group.attrs[attr]
                
        # Get timesteps
        timesteps = group['Timesteps'][:]
        if time_index >= len(timesteps):
            raise ValueError(f"Time index {time_index} out of range (max: {len(timesteps)-1})")
        metadata['timestep_value'] = timesteps[time_index]
        
        # Determine parameter part of the dataset name
        param_part = ""
        if parameter_indices is not None:
            # Check for parameters
            param_names = []
            param_values = []
            
            for i, param_idx in enumerate(parameter_indices):
                param_name = f"Parameter{i+1}"
                if param_name in f and param_idx is not None:
                    param_dataset = f[param_name]
                    if param_idx >= len(param_dataset):
                        raise ValueError(f"Parameter index {param_idx} out of range for {param_name}")
                    param_names.append(param_name)
                    param_values.append(param_dataset[param_idx])
                    param_part += f".{param_idx}"
                    
            metadata['parameters'] = dict(zip(param_names, param_values))
        
        # Construct dataset name
        symbol = metadata.get('symbol', data_group_name)
        dataset_name = f"{symbol}{param_part}"
        
        # Access dataset
        if dataset_name not in group:
            raise ValueError(f"Dataset '{dataset_name}' not found in group '{data_group_name}'")
            
        dataset = group[dataset_name]
        
        # Extract data for the specific time index
        data = dataset[:, :, :, time_index]
        
    return data, spatial_grid, metadata

###Visualize the scalar field ### (Electirc potential)





if __name__ == '__main__':
  main()