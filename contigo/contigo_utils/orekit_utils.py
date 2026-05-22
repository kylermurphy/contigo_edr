import os

from pathlib import Path
from typing import List, Union, Optional

import requests
import jdk4py
import jpype
import orekit_jpype as orekit

import contigo.config as config

def start_orekit(vmargs: Union[str, None] = None,
           additional_classpaths: Union[List, None] = None,
           jvmpath: Optional[Union[str, os.PathLike]] = None):

    if jpype.isJVMStarted() is False:
        print('Starting Orekit JVM')
        # setup the JAVA_HOME environment in order to use orekit
        # some users might not need this but we want to make sure 
        # it works for everyone so we set it to the jdk4py JAVA_HOME 
        # which is compatible with orekit
        os.environ["JAVA_HOME"] = str(jdk4py.JAVA_HOME)
        # get the base directory so we can find files
        f_path = Path(__file__).resolve()
        base_dir = f_path / '..' / '..' / '..'
        if additional_classpaths is None:

            d_dir = base_dir / 'java_src' / 'target' / 'orekit_utils-1.0.0.jar'
            d_dir = d_dir.resolve()

            additional_classpaths = [d_dir]

        #start the orekit JVM with the required
        #contigo class path
        orekit.initVM(jvmpath=jvmpath,
              additional_classpaths=additional_classpaths)

        # finish seeting up the orekit data
        from orekit_jpype.pyhelpers import download_orekit_data_curdir
        from orekit_jpype.pyhelpers import setup_orekit_data

        # get the SHA of the latest commit to the orekit data repository
        # append the first 8 characters of the SHA to the data fine name
        # this ensures we can check that we are always using the latest
        # and won't download it if we have the file already
        orekit_sha = get_gitlab_sha(project_id=18, branch='main')
        data_file = f'orekit_data_{orekit_sha[0:8]}.zip'
        # check for the orekit data file
        orekit_data = Path(config.DATA_DIR).resolve() / data_file

        orekit_data = orekit_data.resolve()
        if not orekit_data.exists():
            # check for old orekit data file and remove them if they exist
            old_files = list(Path(config.DATA_DIR).glob('orekit_data_*.zip'))
            for old_file in old_files:
                print(f'Removing old Orekit data file {old_file}')
                old_file.unlink()

            print(f'Downloading Orekit data to {orekit_data}')
            download_orekit_data_curdir(orekit_data._str)

        # setup the orekit data
        print(f'Loading Orekit data from {orekit_data}')
        setup_orekit_data(filenames=orekit_data._str, from_pip_library=False)

        # set the state variable to true so we know orekit has been loaded
        config.state['orekit_loaded'] = True

def get_gitlab_sha(project_id=18, branch="main"):
    # Note: the branch was renamed from 'master' to 'main'
    url_base = "https://gitlab.orekit.org/api/v4/projects/"
    url = f"{url_base}{project_id}/repository/commits/{branch}"
    response = requests.get(url)
    
    if response.status_code == 404:
        return "Error 404: Check if project ID or branch name is correct."
    
    response.raise_for_status()
    return response.json().get("id")