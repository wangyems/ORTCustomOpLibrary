# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import argparse
import os
from pathlib import Path
import requests
import shutil
import subprocess


class BuildHelper():
    def __init__(self, build_path, package_url):
        self.SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
        self.BUILD_DIR = os.path.join(self.SCRIPT_DIR, build_path)
        self.PACKAGE_DIR = os.path.join(self.BUILD_DIR, "ort_package")
        self.PACKAGE_PATH = os.path.join(self.PACKAGE_DIR, "ort.zip")

        self.make_directory(self.BUILD_DIR, self.PACKAGE_DIR)
        self.download_package_from_url(package_url)
        self.execute_cmake()

    def resolve_executable_path(self, command_or_path):
        if command_or_path and command_or_path.strip():
            executable_path = shutil.which(command_or_path)
            if executable_path is None:
                raise ("Failed to resolve executable path for "
                       "'{}'.".format(command_or_path))
            return os.path.abspath(executable_path)
        else:
            return None

    def execute_cmake(self):
        cmake_path = self.resolve_executable_path("cmake")

        subprocess.run([cmake_path, ".."], cwd=self.BUILD_DIR)
        subprocess.run([cmake_path, "--build", "."], cwd=self.BUILD_DIR)

    def make_directory(self, *paths):
        [Path(path).mkdir(parents=True, exist_ok=True) for path in paths]

    def download_package_from_url(self, url):
        try:
            print("downloading onnxruntime package from", url)
            r = requests.get(url, allow_redirects=True)
        except requests.exceptions.RequestException as e:
            raise SystemExit(e)

        open(self.PACKAGE_PATH, 'wb').write(r.content)
        shutil.unpack_archive(self.PACKAGE_PATH, self.PACKAGE_DIR)


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build_path",
        required=False,
        nargs=1,
        type=str,
        default='build',
        help="Specify build path.")

    parser.add_argument(
        "--package_url",
        required=False,
        nargs=1,
        type=str,
        default=
        'https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.10.0',
        help="Onnxruntime release nuget package download link.")

    args = parser.parse_args()
    return args


def main():
    args = parse_arguments()
    BuildHelper(args.build_path, args.package_url)


if __name__ == "__main__":
    main()
