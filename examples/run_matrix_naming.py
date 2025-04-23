import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.params_common_utils import generate_im_filenames

def main():
    # Percorso del file di configurazione YAML
    import specula
    specula.init(device_idx=-1, precision=1)
    # Get the path to the specula package's __init__.py file
    specula_init_path = specula.__file__
    # Navigate up to repository root
    specula_package_dir = os.path.dirname(specula_init_path)
    specula_repo_path = os.path.dirname(specula_package_dir)

    # Path to the YAML configuration file and output directory
    # The path to the YAML file is detemrined by the specula module
    yaml_file = os.path.join(specula_repo_path, "main", "scao", "params_morfeo_full.yml")
    
    # Genera i nomi dei file
    try:
        filenames_by_type = generate_im_filenames(yaml_file)
        
        print("Nomi file generati per tipo di stella guida:")
        
        # Mostra i files generati per LGS
        if filenames_by_type['lgs']:
            print("\nLaser Guide Star (LGS) files:")
            for i, filename in enumerate(filenames_by_type['lgs'], 1):
                print(f"{i}. {filename}")
                
        # Mostra i files generati per NGS
        if filenames_by_type['ngs']:
            print("\nNatural Guide Star (NGS) files:")
            for i, filename in enumerate(filenames_by_type['ngs'], 1):
                print(f"{i}. {filename}")
                
        # Mostra i files generati per REF
        if filenames_by_type['ref']:
            print("\nReference Star (REF) files:")
            for i, filename in enumerate(filenames_by_type['ref'], 1):
                print(f"{i}. {filename}")
                
    except Exception as e:
        print(f"Errore nella generazione dei nomi dei file: {e}")

if __name__ == "__main__":
    main()