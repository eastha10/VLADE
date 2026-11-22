// VLADE fused W0/W2/W3/W4/W8/BF16 A16/FP32 reference CUDA kernel.
// Packed codes are read directly; no dense dequantized weight tensor is allocated.
extern "C" __global__ void vlade_packed_linear(
 const float* x, const unsigned char* packed, const long long* offsets,
 const unsigned char* bits, const float* scales, const float* bias, float* y,
 int M, int N, int K, int group_size, int groups) {
 int index = blockIdx.x, row = index % N, sample = index / N;
 int b = bits[row];
 float total = 0.0f;
 if (b) for (int k = threadIdx.x; k < K; k += blockDim.x) {
   long long position = (long long)k*b;
   unsigned int word = packed[offsets[row]+position/8];
   if (position%8+b > 8) word |= (unsigned int)packed[offsets[row]+position/8+1] << 8;
   unsigned int code = (word >> (position%8)) & ((1u << b)-1);
   float w = b == 16 ? __uint_as_float(code << 16) :
      ((int)code - ((1 << (b-1))-1)) * scales[(long long)row*groups+k/group_size];
   total += x[(long long)sample*K+k]*w;
 }
 __shared__ float sums[128];
 sums[threadIdx.x] = total;
 __syncthreads();
 for (int stride=64; stride; stride/=2) {
   if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x+stride];
   __syncthreads();
 }
 if (threadIdx.x == 0) y[index] = sums[0] + bias[row];
}
