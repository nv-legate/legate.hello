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

#ifndef __HELLO_TASK_H__
#define __HELLO_TASK_H__

#include "legate.h"

namespace legate {
  namespace hello {

    // You don't have to extend the base LegateTask
    // but it will make your life easier as the base
    // class will help figure out variants and help
    // you make variant registrations with the runtime
    class HelloTask : public LegateTask<HelloTask> {
    public:
      // Every LegateTask has to have a library-relative task ID
      // This is the first task in this library so it gets ID 0
      // To see how we get assigned a global TaskID then see hello.cc
      static const int TASK_ID = 0;
      // Describe an upper bound on how many regions this class uses for
      // the purposes of setting up layout constraints on those regions.
      // This particular task doesn't have any regions.
      static const int REGIONS = 0;

    public:
      // LegateTasks can have up to three variants: one for CPUs, one
      // for OpenMP processors, and one for GPUs. They must have this
      // type and they must be called this in order for the LegateTask
      // base class to be able to recognize them
      static int cpu_variant(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context ctx, Legion::Runtime *runtime);
#ifdef LEGATE_USE_OPENMP
      static int omp_variant(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context ctx, Legion::Runtime *runtime);
#endif
#ifdef LEGATE_USE_CUDA
      static int gpu_variant(const Legion::Task *task,
                             const std::vector<Legion::PhysicalRegion> &regions,
                             Legion::Context ctx, Legion::Runtime *runtime);
#endif
    public:
      // Every LegateTask also needs to support one of these methods that is the
      // call back from LegateTask for each variant on the task
      static void record_variant(Legion::TaskID tid, const Legion::CodeDescriptor &desc,
                                 Legion::ExecutionConstraintSet &execution_constraints,
                                 Legion::TaskLayoutConstraintSet &layout_constraints,
                                 LegateVariant var, Legion::Processor::Kind kind, 
                                 bool leaf, bool inner, bool idempotent,
                                 bool has_non_void_retrun_type);
    };

  } // namespace hello
} // namespace legate

#endif // __HELLO_TASK_H__
