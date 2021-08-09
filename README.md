# OrtCustomOpDynamicLibrary

**A standalone code repository to generate dynamic library files for custom OnnxRuntime operators**.

**Build**: python build.py --help

**Consume in OnnxRuntime Session**: 
- Windows: [consume dll files in C# API](https://github.com/microsoft/onnxruntime/blob/430e80e7b6e5e6222b2d90ca5e43609d62082722/csharp/test/Microsoft.ML.OnnxRuntime.Tests/InferenceTest.cs#L1181)
- Linux: [consume so files in python API](https://github.com/microsoft/onnxruntime/blob/430e80e7b6e5e6222b2d90ca5e43609d62082722/onnxruntime/test/python/onnxruntime_test_python.py#L810)
