import os
import sys
import site

def setup_gpu_libraries():
    """
    Automatically detects pip-installed NVIDIA libraries (e.g., from 
    tensorflow[and-cuda]) and adds their lib directories to LD_LIBRARY_PATH.
    This is necessary when libraries are installed in user site-packages 
    and not automatically picked up by the system linker.
    """
    # Check both system and user site-packages
    try:
        site_packages = site.getsitepackages()
    except AttributeError:
        site_packages = []
        
    if hasattr(site, 'getusersitepackages'):
        site_packages.append(site.getusersitepackages())
    
    cuda_libs = []
    for sp in site_packages:
        nvidia_base = os.path.join(sp, 'nvidia')
        if os.path.exists(nvidia_base):
            for d in os.listdir(nvidia_base):
                lib_path = os.path.join(nvidia_base, d, 'lib')
                if os.path.isdir(lib_path):
                    cuda_libs.append(lib_path)
    
    if cuda_libs:
        unique_libs = []
        for lib in cuda_libs:
            if lib not in unique_libs:
                unique_libs.append(lib)
        
        current_ld = os.environ.get('LD_LIBRARY_PATH', '')
        # Only add if not already there to avoid bloating the env
        new_libs = [l for l in unique_libs if l not in current_ld]
        if new_libs:
            os.environ['LD_LIBRARY_PATH'] = ':'.join(new_libs) + (':' + current_ld if current_ld else '')
            return True
    return False

# Self-executing if imported at the top
setup_gpu_libraries()
