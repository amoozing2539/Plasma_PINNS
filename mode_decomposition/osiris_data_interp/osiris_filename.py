import os

def make_osiris_filename(data_name, timestep, data_type = 'field', folder = '', particle='electrons',
                          cylindrical = False, mode=0, mode_type = 're', tavg = False, q3d = False):
    '''
    Creates file name to access data from osiris simulations (cartesian/default and cylindrical)
  
    Parameters:
    data_name (str): name of the data being accessed, for example: 'p1x1', 'b3', 'gammax1', 'charge', etc.
                     must be of corresponding datatype: p1x1 -> phase; b3 -> field; etc.
  
    Returns:
    filename (str): filename of h5 file of data
  
    '''
#     def print_values(**kwargs):
#     for key, value in kwargs.items():
#         print("The value of {} is {}".format(key, value))
    
    
    if q3d:
        cylindrical = True
    if cylindrical:
        suf = '_cyl_m'
        if tavg:
            suf = '_cyl_m-tavg'
        if mode == 0 and mode_type == 'im':
            mode_type = 're'
        try:
            MODE = 'MODE-'+str(mode)+'-'+mode_type.upper()
        except NameError:
            print('issue with MODE name - check that the correct modes are labeled')
        mode_label = str(mode)+'-'+mode_type.lower()
        
    if data_type == 'field':
        fold = 'MS/FLD'
        if cylindrical:
            field_pre = os.path.join(MODE,data_name+suf)
            field_post = data_name+suf+'-'+mode_label
        else:
            field_pre = data_name
            field_post = data_name
        end_of_file = os.path.join(field_pre,field_post+'-')
        
    elif data_type == 'phase':
        fold = 'MS/PHA'
        phase_pre = os.path.join(data_name,particle)
        phase_post = data_name+'-'+particle
        end_of_file = os.path.join(phase_pre,phase_post+'-')
        
    elif data_type == 'raw':
        fold = 'MS/RAW'
        end_of_file = os.path.join(partice,'RAW-'+particle+'-')
        
    elif data_type == 'density':
        fold = 'MS/DENSITY'
        dens_pre = particle
        if cylindrical:
            dens_post = os.path.join(MODE,data_name+suf, data_name+suf+'-'+particle+'-'+mode_label)
        else:
            dens_post = os.path.join(data_name, data_name+'-'+particle)
        end_of_file =  os.path.join(dens_pre, dens_post+'-')

    elif data_type == 'udist':
        fold = 'MS/UDIST'
        dist_pre = particle
        if cylindrical:
            dist_post = os.path.join(MODE,data_name+suf, data_name+suf+'-'+particle+'-'+mode_label)
        else:
            dist_post = os.path.join(data_name, data_name+'-'+particle)
        end_of_file =  os.path.join(dist_pre, dist_post+'-')

    else:
        raise NameError('Must have data_type that is either field, phase, density, or udist data...')
        return(None)
    
    path_to_file = os.path.join(folder, fold, end_of_file)
    filename = path_to_file+str(timestep).zfill(6)+'.h5'
    
    return(filename)

