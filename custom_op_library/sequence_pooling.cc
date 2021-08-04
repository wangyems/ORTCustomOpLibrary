// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include "sequence_pooling.h"

using namespace std;

// An example
// In: input: [1, 4096, 768]
// In: sen_lens: [1, 47]     contains like [30, 40, 20, ....., 96] and sum up to 4096
// Out: output: [1, 256, 768]
//      where [0, 0:46, 768] is the max pooling result of input along axis=1 by sen_lens
//      and [0, 47:256, 768] part is all zeros

namespace {
template <typename T>
inline void MaxPoolingByRowImpl(T* start_dst, const T* start_src, const T* end_src) {
  while (start_src != end_src) {
    if (*start_src > *start_dst) {    
      *start_dst = *start_src;
    }
    ++start_src;
    ++start_dst;
  }
}

template <typename T>
void MaxPoolingByRow(T* start_dst, const T* start_src, int64_t sentence_length, int hidden_size) {
  if (sentence_length == 0) {
    memset(start_dst, 0, hidden_size * sizeof(T));
    return;
  }
  memcpy(start_dst, start_src, hidden_size * sizeof(T));
  for (int offset = 1; offset < sentence_length; ++offset) {
    start_src += hidden_size;
    MaxPoolingByRowImpl<T>(start_dst, start_src, start_src + hidden_size);
  }
}
} //namespace

void SequencePoolingCPU(
  const int batch_size,
  const int hidden_size,
  const int num_sequences, // 256
  const int sequence_length_for_split,
  const float* input,
  const int64_t* sentence_lengthes,
  float* output) {

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    int64_t past_sentence_length_sum = 0;
    for (int sequence_id = 0; sequence_id < num_sequences; ++sequence_id) {
        const std::ptrdiff_t input_offset(batch_id * sequence_length_for_split * hidden_size + past_sentence_length_sum * hidden_size);
        const std::ptrdiff_t output_offset(batch_id * num_sequences * hidden_size + sequence_id * hidden_size);

        int64_t sentence_length = sentence_lengthes[batch_id * num_sequences + sequence_id];
        MaxPoolingByRow<float>(output + output_offset, input + input_offset, sentence_length, hidden_size);

        past_sentence_length_sum += sentence_length;
    }
  }
      
}
