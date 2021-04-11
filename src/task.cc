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
#include "hello.h"
#include <cstdio>
// Include OpenMP if necessary
#ifdef LEGATE_USE_OPENMP
#include <omp.h>
#endif

using namespace Legion;

namespace legate {
  namespace hello {

    // We can make custom loggers for our library
    Logger log_hello("hello");

    /*static*/ int
    HelloTask::cpu_variant(const Task *task,
                           const std::vector<PhysicalRegion> &regions, Context ctx,
                           Runtime *runtime) {
      LegateDeserializer derez(task->args, task->arglen);
      const int count = derez.unpack_32bit_int();
      std::string name = derez.unpack_string();

      for (int i = 0; i < count; i++)
        log_hello.print("Hello %s from CPU variant", name.c_str());

      return task->is_index_space ? task->index_point[0] : 1;
    }

#ifdef LEGATE_USE_OPENMP
    /*static*/ int
    HelloTask::omp_variant(const Task *task,
                           const std::vector<PhysicalRegion> &regions, Context ctx,
                           Runtime *runtime) {
      LegateDeserializer derez(task->args, task->arglen);
      const int count = derez.unpack_32bit_int();
      std::string name = derez.unpack_string();

#pragma omp parallel for
      for (int i = 0; i < count; i++)
        log_hello.print("Hello %s from thread %d in OpenMP variant", name.c_str(),
                        omp_get_thread_num());

      return task->is_index_space ? task->index_point[0] : 1;
    }
#endif

    /*static*/ void
    HelloTask::record_variant(TaskID tid, const CodeDescriptor &desc,
                              ExecutionConstraintSet &execution_constraints,
                              TaskLayoutConstraintSet &layout_constraints,
                              LegateVariant var, Processor::Kind kind, bool leaf,
                              bool inner, bool idempotent, bool ret_type) {
      // For this just turn around and call this on the base LegateNumPy
      // type so it will deduplicate across all task kinds
      LegateHello::record_variant(tid, task_name(), desc, execution_constraints,
                                  layout_constraints, var, kind, leaf, inner,
                                  idempotent, ret_type);
    }

  } // namespace hello
} // namespace legate

namespace // unnammed
{
static void __attribute__((constructor)) register_tasks(void) {
  // Use this to hook registration of the variants, ask LegateTask to
  // register all the variants which is what will trigger the callback
  // for recording all variants
  legate::hello::HelloTask::register_variants_with_return<int, int>();
}
}; // namespace
