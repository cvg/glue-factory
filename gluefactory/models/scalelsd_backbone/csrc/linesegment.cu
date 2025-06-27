#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

// #include <THC/THC.h>
// #include <THC/THCDeviceUtils.cuh>
#include <torch/torch.h>
#include <torch/extension.h>

#include <vector>
#include <iostream>

int const CUDA_NUM_THREADS = 1024;

inline int CUDA_GET_BLOCKS(const int N) {
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += blockDim.x * gridDim.x)


__global__ void encode_kernel(const int nthreads, const float* lines,
                           const int input_height, const int input_width, const int num,
                           const int height, const int width, float* map,
                           bool* label, float* tmap)
{
    CUDA_1D_KERNEL_LOOP(index, nthreads){
        int w = index % width;
        int h = (index / width) % height;
        int x_index = h*width + w;
        int y_index = height*width + h*width + w;
        int ux_index = 2*height*width + h*width + w;
        int uy_index = 3*height*width + h*width + w;
        int vx_index = 4*height*width + h*width + w;
        int vy_index = 5*height*width + h*width + w;
        int label_index = h*width + w;

        float px = (float) w;
        float py = (float) h;
        float min_dis = 1e30;
        int minp = -1;
        bool flagp = true;
        for(int i = 0; i < num; ++i) {
            float xs = (float)width  /(float)input_width;
            float ys = (float)height /(float)input_height;
            float x1 = lines[4*i  ]*xs;
            float y1 = lines[4*i+1]*ys;
            float x2 = lines[4*i+2]*xs;
            float y2 = lines[4*i+3]*ys;

            float dx = x2 - x1;
            float dy = y2 - y1;
            float ux = x1 - px;
            float uy = y1 - py;
            float vx = x2 - px;
            float vy = y2 - py;
            float norm2 = dx*dx + dy*dy;
            bool flag = false;
            float t = ((px-x1)*dx + (py-y1)*dy)/(norm2+1e-6);
            if (t<=1 && t>=0.0)
                flag = true;

            t = t<0.0? 0.0:t;
            t = t>1.0? 1.0:t;

            float ax = x1   + t*(x2-x1) - px;
            float ay = y1   + t*(y2-y1) - py;

            float dis = ax*ax + ay*ay;
            if (dis < min_dis) {
                min_dis = dis;
                map[x_index] = ax;
                map[y_index] = ay;
                float norm_u2 = ux*ux+uy*uy;
                float norm_v2 = vx*vx+vy*vy;

                if (norm_u2 < norm_v2){
                    map[ux_index] = ux;
                    map[uy_index] = uy;
                    map[vx_index] = vx;
                    map[vy_index] = vy;
                }
                else{
                    map[ux_index] = vx;
                    map[uy_index] = vy;
                    map[vx_index] = ux;
                    map[vy_index] = uy;
                }

                minp = i;
                if (flag)
                    flagp = true;
                else
                    flagp = false;

                tmap[index] = t;
            }
        }
        // label[label_index+minp*height*width] = flagp;

    }
}


std::tuple<at::Tensor, at::Tensor, at::Tensor> lsencode_cuda(
    const at::Tensor& lines,
    const int input_height,
    const int input_width,
    const int height,
    const int width,
    const int num_lines)

{
    auto map = at::zeros({6,height,width}, lines.options());
    auto tmap = at::zeros({1,height,width}, lines.options());
    auto label = at::zeros({1,height,width}, lines.options().dtype(at::kBool));
    auto nthreads = height*width;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    float* map_data = map.data<float>();
    float* tmap_data = tmap.data<float>();
    bool*  label_data = label.data<bool>();

    encode_kernel<<<CUDA_GET_BLOCKS(nthreads), CUDA_NUM_THREADS >>>(
            nthreads,
            lines.contiguous().data<float>(),
            input_height, input_width,
            num_lines,
            height, width,
            map_data,
            label_data,
            tmap_data);

    // THCudaCheck(cudaGetLastError());

    return std::make_tuple(map, label, tmap);
}