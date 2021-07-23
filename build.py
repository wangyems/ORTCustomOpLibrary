import os
from pathlib import Path
import requests
import shutil
import sys

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def resolve_executable_path(command_or_path):
    """Returns the absolute path of an executable."""
    if command_or_path and command_or_path.strip():
        executable_path = shutil.which(command_or_path)
        if executable_path is None:
            raise ("Failed to resolve executable path for "
                             "'{}'.".format(command_or_path))
        return os.path.abspath(executable_path)
    else:
        return None

clean = True

if __name__ == "__main__":

    if clean:
        print(resolve_executable_path('cmake'))
        #shutil.rmtree(SCRIPT_DIR + "/package")
        #shutil.rmtree(SCRIPT_DIR + "/build")
        sys.exit(0)

    # different os?
    Path(SCRIPT_DIR + "/package").mkdir(parents=True, exist_ok=True)
    Path(SCRIPT_DIR + "/build").mkdir(parents=True, exist_ok=True)

    url = 'https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.8.1'
    r = requests.get(url, allow_redirects=True)

    pkg_path = Path(SCRIPT_DIR + '/package' + '/ort.zip')
    open(pkg_path, 'wb').write(r.content)

    shutil.unpack_archive(pkg_path, pkg_path.parent.absolute())

    #cmake_path = resolve_executable_path(args.cmake_path)
    #ctest_path = None if args.use_vstest else resolve_executable_path(args.ctest_path)
    #build_dir = args.build_dir
    #script_dir = os.path.realpath(os.path.dirname(__file__))
    #source_dir = os.path.normpath(os.path.join(script_dir, "..", ".."))