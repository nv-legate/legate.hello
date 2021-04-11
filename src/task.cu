/* Copyright 2021 NVIDIA Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include "task.h"
#include <cstdio>

#define THREADS_PER_BLOCK 128

using namespace Legion;

namespace legate {
  namespace hello {

    struct HelloArgs {
    public:
      char name[64];
      int count;
    };

    __global__ void gpu_hello_world(const HelloArgs args) {
      const int cnt = blockIdx.x * blockDim.x + threadIdx.x;
      if (cnt >= args.count)
        return;
      printf("Hello %s from thread %d of block %d in GPU variant\n", args.name,
             threadIdx.x, blockIdx.x);
    }

    /*static*/ int
    HelloTask::gpu_variant(const Task *task,
                           const std::vector<PhysicalRegion> &regions, Context ctx,
                           Runtime *runtime) {
      LegateDeserializer derez(task->args, task->arglen);
      HelloArgs args;
      args.count = derez.unpack_32bit_int();
      std::string name = derez.unpack_string();
      assert((name.size() + 1) < sizeof(args.name));
      strncpy(args.name, name.c_str(), sizeof(args.name));

      // You can launch CUDA kernels inside GPU variants like normal
      const int blocks = (args.count + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
      gpu_hello_world<<<blocks, THREADS_PER_BLOCK>>>(args);
      // No need to synchronize as Legion handles all this for you

      return task->is_index_space ? task->index_point[0] : 1;
    }

  } // namespace hello
} // namespace legate

