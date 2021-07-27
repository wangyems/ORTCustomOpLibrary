# OrtCustomOpDynamicLibrary

python build.py --help
usage: build.py [-h] [--build_path BUILD_PATH] [--package_url PACKAGE_URL]

optional arguments:
  -h, --help            show this help message and exit
  --build_path BUILD_PATH
                        Specify build path.
  --package_url PACKAGE_URL
                        Onnxruntime release nuget package download link.

python build.py
downloading onnxruntime package from https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.Gpu/1.8.1
-- Configuring done
-- Generating done
-- Build files have been written to: /bert_ort/wy/Turing/OrtCustomOpDynamicLibrary/build
Consolidate compiler generated dependencies of target custom_op_library_fp16
[ 12%] Building CXX object CMakeFiles/custom_op_library_fp16.dir/custom_op_library_fp16.cc.o
[ 25%] Linking CUDA device code CMakeFiles/custom_op_library_fp16.dir/cmake_device_link.o
[ 37%] Linking CXX shared library libcustom_op_library_fp16.so
[ 50%] Built target custom_op_library_fp16
Consolidate compiler generated dependencies of target custom_op_library_fp32
[ 62%] Building CXX object CMakeFiles/custom_op_library_fp32.dir/custom_op_library_fp32.cc.o
[ 75%] Linking CUDA device code CMakeFiles/custom_op_library_fp32.dir/cmake_device_link.o
[ 87%] Linking CXX shared library libcustom_op_library_fp32.so
[100%] Built target custom_op_library_fp32
