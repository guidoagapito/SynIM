import os
import yaml
import datetime
import re

def extract_wfs_list(config):
    """Extract all WFS configurations from config"""
    wfs_list = []
    
    # Find Pyramid WFSs
    for key in config:
        if key == 'pyramid' or (isinstance(key, str) and (key.startswith('pyramid') or key == 'pyr')):
            wfs_list.append({
                'name': key,
                'type': 'pyr',
                'index': re.findall(r'\d+', key)[0] if re.findall(r'\d+', key) else '1',
                'config': config[key]
            })
    
    # Find Shack-Hartmann WFSs
    for key in config:
        if key == 'sh' or key.startswith('sh_') or key.startswith('sh'):
            wfs_list.append({
                'name': key,
                'type': 'sh', 
                'index': re.findall(r'\d+', key)[0] if re.findall(r'\d+', key) else '1',
                'config': config[key]
            })
    
    return wfs_list

def extract_dm_list(config):
    """Extract all DM configurations from config"""
    dm_list = []
    
    for key in config:
        if key == 'dm' or (isinstance(key, str) and key.startswith('dm')):
            dm_list.append({
                'name': key,
                'index': re.findall(r'\d+', key)[0] if re.findall(r'\d+', key) else '1',
                'config': config[key]
            })
    
    # If no DMs found, create a default one
    if len(dm_list) == 0 and 'dm' in config:
        dm_list.append({
            'name': 'dm',
            'index': '1',
            'config': config['dm']
        })
    
    return dm_list

def extract_source_info(config, wfs_name):
    """Extract source information related to a specific WFS"""
    source_info = {}
    
    # Check both formats: direct reference or connection through prop
    if 'inputs' in config[wfs_name] and 'in_ef' in config[wfs_name]['inputs']:
        source_ref = config[wfs_name]['inputs']['in_ef']
        
        # Format might be 'prop.out_source_lgs1_ef' or direct 'out_source_lgs1_ef'
        match = re.search(r'out_(\w+)_ef', source_ref)
        if match:
            source_name = match.group(1)
            
            # Find the source in the config
            if source_name in config:
                source = config[source_name]
                source_info['type'] = 'lgs' if 'lgs' in source_name else 'ngs'
                source_info['name'] = source_name
                if 'polar_coordinates' in source:
                    source_info['pol_coords'] = source['polar_coordinates']
                if 'height' in source:
                    source_info['height'] = source['height']
                if 'wavelengthInNm' in source:
                    source_info['wavelength'] = source['wavelengthInNm']
    
    # Direct reference within source objects (for simple YAML)
    if not source_info and 'on_axis_source' in config:
        source_info['type'] = 'ngs'
        source_info['name'] = 'on_axis_source'
        source_info['pol_coords'] = config['on_axis_source'].get('polar_coordinates', [0, 0])
        source_info['wavelength'] = config['on_axis_source'].get('wavelengthInNm', 750)
    
    return source_info

def is_simple_config(config):
    """Detect if this is a simple SCAO config or a complex MCAO config"""
    # Check for multiple DMs
    dm_count = sum(1 for key in config if key.startswith('dm') and key != 'dm')
    
    # Check for multiple WFSs
    wfs_count = sum(1 for key in config if key.startswith('sh_') or key.startswith('pyramid') and key != 'pyramid')
    
    return dm_count == 0 and wfs_count == 0

def generate_im_filenames(config_file, timestamp=False):
    """Generate interaction matrix filenames for all WFS-DM combinations"""
    
    # Load YAML configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    # Detect if simple or complex configuration
    simple_config = is_simple_config(config)
    
    # Basic system info
    instrument = os.path.basename(config_file).split('.')[0].split('_')[0].upper()
    
    # Pupil parameters
    pupil_params = {}
    if 'main' in config:
        pupil_params['pixel_pupil'] = config['main'].get('pixel_pupil', 0)
        pupil_params['pixel_pitch'] = config['main'].get('pixel_pitch', 0)
    
    if 'pupilstop' in config:
        pupstop = config['pupilstop']
        if isinstance(pupstop, dict):
            pupil_params['obsratio'] = pupstop.get('obsratio', 0.0)
            pupil_params['tag'] = pupstop.get('tag', '')
    
    if simple_config:
        # Simple SCAO configuration
        filenames = []
        
        # Determine WFS type (pyramid or SH)
        wfs_type = None
        wfs_params = {}
        
        if 'pyramid' in config:
            wfs_type = 'pyr'
            wfs_params = {
                'pup_diam': config['pyramid'].get('pup_diam', 0),
                'mod_amp': config['pyramid'].get('mod_amp', 0),
                'wavelength': config['pyramid'].get('wavelengthInNm', 0),
                'fov': config['pyramid'].get('fov', 0)
            }
        elif 'sh' in config:
            wfs_type = 'sh'
            wfs_params = {
                'nsubaps': config['sh'].get('subap_on_diameter', 0),
                'wavelength': config['sh'].get('wavelengthInNm', 0),
                'fov': config['sh'].get('subap_wanted_fov', 0),
                'npx': config['sh'].get('subap_npx', 0)
            }
        
        # Source info
        source_info = {}
        if 'on_axis_source' in config:
            source_info = {
                'type': 'ngs',
                'pol_coords': config['on_axis_source'].get('polar_coordinates', [0, 0]),
                'wavelength': config['on_axis_source'].get('wavelengthInNm', 0)
            }
        
        # DM info
        dm_params = {}
        if 'dm' in config:
            dm_params = {
                'height': config['dm'].get('height', 0),
                'nmodes': config['dm'].get('nmodes', 0),
                'type': config['dm'].get('type_str', 'zernike')
            }
        
        # Build filename parts
        parts = []
        parts.append(instrument)
        
        if source_info:
            if 'pol_coords' in source_info:
                dist, angle = source_info['pol_coords']
                parts.append(f"pd{dist:.1f}a{angle:.0f}")
        
        if pupil_params:
            ps = pupil_params.get('pixel_pupil', 0)
            pp = pupil_params.get('pixel_pitch', 0)
            parts.append(f"ps{ps}p{pp:.4f}")
            
            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                parts.append(f"o{pupil_params['obsratio']:.3f}")
        
        if wfs_type == 'sh':
            nsubaps = wfs_params.get('nsubaps', 0)
            wl = wfs_params.get('wavelength', 0)
            fov = wfs_params.get('fov', 0)
            npx = wfs_params.get('npx', 0)
            parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
        
        elif wfs_type == 'pyr':
            pup_diam = wfs_params.get('pup_diam', 0)
            wl = wfs_params.get('wavelength', 0)
            mod_amp = wfs_params.get('mod_amp', 0)
            fov = wfs_params.get('fov', 0)
            parts.append(f"pyr{pup_diam:.1f}_wl{wl}_fv{fov:.1f}_ma{mod_amp:.1f}")
        
        if dm_params:
            height = dm_params.get('height', 0)
            nmodes = dm_params.get('nmodes', 0)
            dm_type = dm_params.get('type', 'zernike')
            parts.append(f"dmH{height}_nm{nmodes}_{dm_type}")
        
        # Add timestamp if requested
        if timestamp:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(ts)
        
        # Join all parts with underscores and add extension
        filename = "_".join(parts) + ".fits"
        filenames.append(filename)
    
    else:
        # Complex MCAO configuration
        filenames = []
        
        # Extract WFS and DM lists
        wfs_list = extract_wfs_list(config)
        dm_list = extract_dm_list(config)
        
        # Generate filenames for each WFS-DM combination
        for wfs in wfs_list:
            for dm in dm_list:
                parts = []
                # 1. System identifier
                parts.append(instrument)
                
                # 2. Source information
                source_info = extract_source_info(config, wfs['name'])
                if source_info:
                    if 'pol_coords' in source_info:
                        dist, angle = source_info['pol_coords']
                        parts.append(f"pd{dist:.1f}a{angle:.0f}")
                    
                    if source_info.get('type') == 'lgs' and 'height' in source_info:
                        parts.append(f"h{source_info['height']:.0f}")
                
                # 3. Pupil parameters
                if pupil_params:
                    ps = pupil_params.get('pixel_pupil', 0)
                    pp = pupil_params.get('pixel_pitch', 0)
                    parts.append(f"ps{ps}p{pp:.4f}")
                    
                    if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                        parts.append(f"o{pupil_params['obsratio']:.3f}")
                
                # 4. WFS parameters
                wfs_config = wfs['config']
                if wfs['type'] == 'sh':
                    nsubaps = wfs_config.get('subap_on_diameter', 0)
                    wl = wfs_config.get('wavelengthInNm', 0)
                    fov = wfs_config.get('subap_wanted_fov', 0)
                    npx = wfs_config.get('subap_npx', 0)
                    parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
                
                elif wfs['type'] == 'pyr':
                    pup_diam = wfs_config.get('pup_diam', 0)
                    wl = wfs_config.get('wavelengthInNm', 0)
                    mod_amp = wfs_config.get('mod_amp', 0)
                    fov = wfs_config.get('fov', 0)
                    parts.append(f"pyr{pup_diam:.1f}_wl{wl}_fv{fov:.1f}_ma{mod_amp:.1f}")
                
                # 5. DM parameters
                dm_config = dm['config']
                height = dm_config.get('height', 0)
                nmodes = dm_config.get('nmodes', 0)
                dm_type = dm_config.get('type_str', 'zernike')
                parts.append(f"dmH{height}_nm{nmodes}_{dm_type}")
                
                # Add WFS and DM indices to make filename unique
                parts.append(f"wfs{wfs['index']}_dm{dm['index']}")
                
                # Add timestamp if requested
                if timestamp:
                    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    parts.append(ts)
                
                # Join all parts with underscores and add extension
                filename = "_".join(parts) + ".fits"
                filenames.append(filename)
    
    return filenames

# For backward compatibility
def generate_im_filename(config_file, timestamp=False):
    """Generate a single interaction matrix filename (for backward compatibility)"""
    filenames = generate_im_filenames(config_file, timestamp)
    return filenames

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        yaml_file = sys.argv[1]
        filenames = generate_im_filenames(yaml_file)
        print(f"Generated filenames:")
        for filename in filenames:
            print(f"  {filename}")
    else:
        print("Usage: python filename_generator.py path/to/config.yml")

# import os
# import yaml
# import datetime

# def extract_wfs_params(config):
#     """Extract WFS parameters from config"""
#     # Try to determine WFS type (pyramid, SH, etc.)
#     wfs_type = None
#     wfs_params = {}
    
#     # Check for pyramid WFS
#     if any(k for k in config.keys() if 'pyramid' in k.lower()):
#         wfs_type = 'pyr'
#         for key in config:
#             if 'pyramid' in key.lower() and isinstance(config[key], dict):
#                 pyramid_config = config[key]
#                 wfs_params.update({
#                     'pup_diam': pyramid_config.get('pup_diam', 0),
#                     'mod_amp': pyramid_config.get('mod_amp', 0),
#                     'wavelength': pyramid_config.get('wavelengthInNm', 0),
#                     'fov': pyramid_config.get('fov', 0)
#                 })
#                 break
    
#     # Check for Shack-Hartmann WFS
#     elif any(k for k in config.keys() if k.startswith('sh_')):
#         wfs_type = 'sh'
#         for key in config:
#             if key.startswith('sh_') and isinstance(config[key], dict):
#                 sh_config = config[key]
#                 wfs_params.update({
#                     'nsubaps': sh_config.get('subap_on_diameter', 0),
#                     'wavelength': sh_config.get('wavelengthInNm', 0),
#                     'fov': sh_config.get('subap_wanted_fov', 0),
#                     'npx': sh_config.get('subap_npx', 0)
#                 })
#                 break
    
#     return wfs_type, wfs_params

# def extract_source_params(config):
#     """Extract source parameters from config"""
#     source_params = {}
    
#     # Look for LGS sources
#     lgs_sources = [k for k in config.keys() if k.startswith('source_lgs')]
#     if lgs_sources:
#         for src_key in lgs_sources:
#             source = config[src_key]
#             if isinstance(source, dict) and 'polar_coordinates' in source:
#                 source_params['type'] = 'lgs'
#                 source_params['pol_coords'] = source.get('polar_coordinates', [0, 0])
#                 source_params['height'] = source.get('height', 90000)
#                 source_params['wavelength'] = source.get('wavelengthInNm', 589)
#                 break
    
#     # Look for NGS sources
#     ngs_sources = [k for k in config.keys() if k.startswith('source_') and not k.startswith('source_lgs')]
#     if ngs_sources:
#         for src_key in ngs_sources:
#             source = config[src_key]
#             if isinstance(source, dict) and 'polar_coordinates' in source:
#                 # Only override if no LGS was found or if this is specifically an on-axis source
#                 if 'type' not in source_params or 'on_axis' in src_key:
#                     source_params['type'] = 'ngs'
#                     source_params['pol_coords'] = source.get('polar_coordinates', [0, 0])
#                     source_params['wavelength'] = source.get('wavelengthInNm', 750)
#                 break
    
#     return source_params

# def extract_dm_params(config):
#     """Extract DM parameters from config"""
#     dm_params = {}
    
#     # Look for DM configuration
#     dm_keys = [k for k in config.keys() if k.startswith('dm')]
#     if dm_keys:
#         for dm_key in dm_keys:
#             dm_config = config[dm_key]
#             if isinstance(dm_config, dict):
#                 dm_params['height'] = dm_config.get('height', 0)
#                 dm_params['nmodes'] = dm_config.get('nmodes', 0)
#                 dm_params['type'] = dm_config.get('type_str', 'zernike')
#                 break
    
#     return dm_params

# def extract_pupil_params(config):
#     """Extract pupil parameters from config"""
#     pupil_params = {}
    
#     if 'main' in config:
#         pupil_params['pixel_pupil'] = config['main'].get('pixel_pupil', 0)
#         pupil_params['pixel_pitch'] = config['main'].get('pixel_pitch', 0)
    
#     # Look for pupil configuration
#     if 'pupilstop' in config:
#         pupstop = config['pupilstop']
#         if isinstance(pupstop, dict):
#             pupil_params['obsratio'] = pupstop.get('obsratio', 0.0)
#             pupil_params['tag'] = pupstop.get('tag', '')
    
#     return pupil_params

# def generate_im_filename(config_file, timestamp=False):
#     """Generate an interaction matrix filename based on a YAML config file"""
    
#     # Load YAML configuration
#     with open(config_file, 'r') as f:
#         config = yaml.safe_load(f)
    
#     # Extract parameters
#     pupil_params = extract_pupil_params(config)
#     dm_params = extract_dm_params(config)
#     source_params = extract_source_params(config)
#     wfs_type, wfs_params = extract_wfs_params(config)
    
#     # Start building the filename
#     parts = []
    
#     # 1. Instrument/system identifier
#     instrument = os.path.basename(config_file).split('.')[0].split('_')[0].upper()
#     parts.append(instrument)
    
#     # 2. Source parameters
#     if source_params:
#         if 'pol_coords' in source_params:
#             dist, angle = source_params['pol_coords']
#             parts.append(f"pd{dist:.1f}a{angle:.0f}")
        
#         if source_params.get('type') == 'lgs' and 'height' in source_params:
#             parts.append(f"h{source_params['height']:.0f}")
    
#     # 3. Pupil parameters
#     if pupil_params:
#         ps = pupil_params.get('pixel_pupil', 0)
#         pp = pupil_params.get('pixel_pitch', 0)
#         parts.append(f"ps{ps}p{pp:.4f}")
        
#         if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
#             parts.append(f"o{pupil_params['obsratio']:.3f}")
    
#     # 4. WFS parameters
#     if wfs_type == 'sh':
#         nsubaps = wfs_params.get('nsubaps', 0)
#         wl = wfs_params.get('wavelength', 0)
#         fov = wfs_params.get('fov', 0)
#         npx = wfs_params.get('npx', 0)
#         parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
    
#     elif wfs_type == 'pyr':
#         pup_diam = wfs_params.get('pup_diam', 0)
#         wl = wfs_params.get('wavelength', 0)
#         mod_amp = wfs_params.get('mod_amp', 0)
#         fov = wfs_params.get('fov', 0)
#         parts.append(f"pyr{pup_diam:.1f}_wl{wl}_fv{fov:.1f}_ma{mod_amp:.1f}")
    
#     # 5. DM parameters
#     if dm_params:
#         height = dm_params.get('height', 0)
#         nmodes = dm_params.get('nmodes', 0)
#         dm_type = dm_params.get('type', 'zernike')
#         parts.append(f"dmH{height}_nm{nmodes}_{dm_type}")
    
#     # Add timestamp if requested
#     if timestamp:
#         ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#         parts.append(ts)
    
#     # Join all parts with underscores and add extension
#     filename = "_".join(parts) + ".fits"
    
#     return filename

# if __name__ == "__main__":
#     # Example usage
#     import sys
#     if len(sys.argv) > 1:
#         yaml_file = sys.argv[1]
#         filename = generate_im_filename(yaml_file)
#         print(f"Generated filename: {filename}")
#     else:
#         print("Usage: python filename_generator.py path/to/config.yml")