// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

using namespace std;
#include <cuda_fp16.h>
#include <cuda_runtime.h>

void SequencePoolingCuda(
  cudaStream_t stream,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const float* input,
  const int64_t* sentence_lengthes,
  float* output);

void SequencePoolingCuda(
  cudaStream_t stream,
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const half* input,
  const int64_t* sentence_lengthes,
  half* output);

void SequencePoolingCPU(
  const int batch_size,
  const int hidden_size,
  const int num_sequences,
  const int sequence_length_for_split,
  const float* input,
  const int64_t* sentence_lengthes,
  float* output);
