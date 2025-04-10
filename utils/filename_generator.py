import os
import yaml
import datetime
import re
from utils.params_utils import parse_params_file

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
    """Generate interaction matrix filenames for all WFS-DM combinations, grouped by star type"""
    
    # Load YAML or PRO configuration
    config = parse_params_file(config_file)
    
    # Detect if simple or complex configuration
    simple_config = is_simple_config(config)
    
    # Basic system info
    base_name = 'IM_syn'
    
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
    
    # Output dictionary: key=star type, value=list of filenames
    filenames_by_type = {
        'lgs': [],
        'ngs': [],
        'ref': []
    }
    
    # Extract all DM configurations
    dm_list = extract_dm_list(config)
    
    # For simple configurations with on-axis source
    if simple_config:
        # Simple SCAO configuration
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
        parts.append(base_name)
        
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
            parts.append(f"dmH{height}")
            
            # Check for custom influence functions
            if 'ifunc_tag' in config['dm']:
                parts.append(f"ifunc_{config['dm']['ifunc_tag']}")
            elif 'ifunc_object' in config['dm']:
                parts.append(f"ifunc_{config['dm']['ifunc_object']}")
            elif 'type_str' in config['dm']:
                nmodes = dm_params.get('nmodes', 0)
                parts.append(f"nm{nmodes}_{dm_params['type']}")
            else:
                # Default case
                nmodes = dm_params.get('nmodes', 0)
                parts.append(f"nm{nmodes}")

        # Add timestamp if requested
        if timestamp:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            parts.append(ts)
        
        # Join all parts with underscores and add extension
        filename = "_".join(parts) + ".fits"
        filenames_by_type['ngs'].append(filename)  # Default to NGS for simple config
    else:
        # Complex MCAO configuration: find all sources and related WFSs
        
        # Find all LGS sources and WFSs
        lgs_sources = [k for k in config.keys() if k.startswith('source_lgs')]
        for source_key in lgs_sources:
            source_idx = re.search(r'(\d+)$', source_key)
            if source_idx:
                idx = source_idx.group(1)
                wfs_key = f'sh_lgs{idx}'
                
                if wfs_key in config:
                    # Process each DM with this WFS
                    for dm in dm_list:
                        parts = []
                        parts.append(base_name)
                        
                        # Source parameters
                        source = config[source_key]
                        if 'polar_coordinates' in source:
                            dist, angle = source['polar_coordinates']
                            parts.append(f"pd{dist:.1f}a{angle:.0f}")
                        
                        if 'height' in source:
                            parts.append(f"h{source['height']:.0f}")
                        
                        # Pupil parameters
                        if pupil_params:
                            ps = pupil_params.get('pixel_pupil', 0)
                            pp = pupil_params.get('pixel_pitch', 0)
                            parts.append(f"ps{ps}p{pp:.4f}")
                            
                            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                                parts.append(f"o{pupil_params['obsratio']:.3f}")
                        
                        # WFS parameters
                        wfs_config = config[wfs_key]
                        nsubaps = wfs_config.get('subap_on_diameter', 0)
                        wl = wfs_config.get('wavelengthInNm', 0)
                        fov = wfs_config.get('subap_wanted_fov', 0)
                        npx = wfs_config.get('subap_npx', 0)
                        parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")

                        # DM parameters
                        dm_config = dm['config']
                        height = dm_config.get('height', 0)
                        parts.append(f"dmH{height}")

                        # Check for custom influence functions
                        if 'ifunc_tag' in dm_config:
                            parts.append(f"ifunc_{dm_config['ifunc_tag']}")
                        elif 'ifunc_object' in dm_config:
                            parts.append(f"ifunc_{dm_config['ifunc_object']}")
                        elif 'type_str' in dm_config:
                            nmodes = dm_config.get('nmodes', 0)
                            parts.append(f"nm{nmodes}_{dm_config['type_str']}")
                        else:
                            # Default case
                            nmodes = dm_config.get('nmodes', 0)
                            parts.append(f"nm{nmodes}")

                        # Add WFS and DM indices to make filename unique
                        #parts.append(f"lgs{idx}_dm{dm['index']}")
                        
                        # Add timestamp if requested
                        if timestamp:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            parts.append(ts)
                        
                        # Join all parts with underscores and add extension
                        filename = "_".join(parts) + ".fits"
                        filenames_by_type['lgs'].append(filename)
        
        # Find all NGS sources and WFSs
        ngs_sources = [k for k in config.keys() if k.startswith('source_ngs')]
        for source_key in ngs_sources:
            source_idx = re.search(r'(\d+)$', source_key)
            if source_idx:
                idx = source_idx.group(1)
                wfs_key = f'sh_ngs{idx}'
                
                if wfs_key in config:
                    # Process each DM with this WFS
                    for dm in dm_list:
                        parts = []
                        parts.append(base_name)
                        
                        # Source parameters
                        source = config[source_key]
                        if 'polar_coordinates' in source:
                            dist, angle = source['polar_coordinates']
                            parts.append(f"pd{dist:.1f}a{angle:.0f}")
                        
                        # Pupil parameters
                        if pupil_params:
                            ps = pupil_params.get('pixel_pupil', 0)
                            pp = pupil_params.get('pixel_pitch', 0)
                            parts.append(f"ps{ps}p{pp:.4f}")
                            
                            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                                parts.append(f"o{pupil_params['obsratio']:.3f}")
                        
                        # WFS parameters
                        wfs_config = config[wfs_key]
                        nsubaps = wfs_config.get('subap_on_diameter', 0)
                        wl = wfs_config.get('wavelengthInNm', 0)
                        fov = wfs_config.get('subap_wanted_fov', 0)
                        npx = wfs_config.get('subap_npx', 0)
                        parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
                        
                        # DM parameters
                        dm_config = dm['config']
                        height = dm_config.get('height', 0)
                        nmodes = dm_config.get('nmodes', 0)
                        parts.append(f"dmH{height}_nm{nmodes}")
                        
                        # Add WFS and DM indices to make filename unique
                        #parts.append(f"ngs{idx}_dm{dm['index']}")
                        
                        # Add timestamp if requested
                        if timestamp:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            parts.append(ts)
                        
                        # Join all parts with underscores and add extension
                        filename = "_".join(parts) + ".fits"
                        filenames_by_type['ngs'].append(filename)
        
        # Find all REF sources and WFSs
        ref_sources = [k for k in config.keys() if k.startswith('source_ref')]
        for source_key in ref_sources:
            source_idx = re.search(r'(\d+)$', source_key)
            if source_idx:
                idx = source_idx.group(1)
                wfs_key = f'sh_ref{idx}'
                
                if wfs_key in config:
                    # Process each DM with this WFS
                    for dm in dm_list:
                        parts = []
                        parts.append(base_name)
                        
                        # Source parameters
                        source = config[source_key]
                        if 'polar_coordinates' in source:
                            dist, angle = source['polar_coordinates']
                            parts.append(f"pd{dist:.1f}a{angle:.0f}")
                        
                        # Pupil parameters
                        if pupil_params:
                            ps = pupil_params.get('pixel_pupil', 0)
                            pp = pupil_params.get('pixel_pitch', 0)
                            parts.append(f"ps{ps}p{pp:.4f}")
                            
                            if 'obsratio' in pupil_params and pupil_params['obsratio'] > 0:
                                parts.append(f"o{pupil_params['obsratio']:.3f}")
                        
                        # WFS parameters
                        wfs_config = config[wfs_key]
                        nsubaps = wfs_config.get('subap_on_diameter', 0)
                        wl = wfs_config.get('wavelengthInNm', 0)
                        fov = wfs_config.get('subap_wanted_fov', 0)
                        npx = wfs_config.get('subap_npx', 0)
                        parts.append(f"sh{nsubaps}x{nsubaps}_wl{wl}_fv{fov:.1f}_np{npx}")
                        
                        # DM parameters
                        dm_config = dm['config']
                        height = dm_config.get('height', 0)
                        nmodes = dm_config.get('nmodes', 0)
                        parts.append(f"dmH{height}_nm{nmodes}")
                        
                        # Add WFS and DM indices to make filename unique
                        #parts.append(f"ref{idx}_dm{dm['index']}")
                        
                        # Add timestamp if requested
                        if timestamp:
                            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            parts.append(ts)
                        
                        # Join all parts with underscores and add extension
                        filename = "_".join(parts) + ".fits"
                        filenames_by_type['ref'].append(filename)
    
    return filenames_by_type

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