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

#include "hello.h"
#include "mapper.h"

using namespace Legion;

namespace legate {
  namespace hello {

    // This is the unique string name for our library which can be used
    // from both C++ and Python to generate IDs
    static const char *const hello_library_name = "legate.hello";

    /*static*/ void
    LegateHello::record_variant(TaskID tid, const char *task_name,
                                const CodeDescriptor &descriptor,
                                ExecutionConstraintSet &execution_constraints,
                                TaskLayoutConstraintSet &layout_constraints,
                                LegateVariant var, Processor::Kind kind, bool leaf,
                                bool inner, bool idempotent, bool ret_type) {
      // Buffer up our task variants in the pending_task-variants registrar until
      // the runtime has started and we can register them with our library IDs
      assert((kind == Processor::LOC_PROC) || (kind == Processor::TOC_PROC) ||
             (kind == Processor::OMP_PROC));
      std::deque<PendingTaskVariant> &pending_task_variants =
          get_pending_task_variants();
      pending_task_variants.push_back(
          PendingTaskVariant(tid, false /*global*/,
                             (kind == Processor::LOC_PROC)
                                 ? "CPU"
                                 : (kind == Processor::TOC_PROC) ? "GPU" : "OpenMP",
                             task_name, descriptor, var, ret_type));
      TaskVariantRegistrar &registrar = pending_task_variants.back();
      registrar.execution_constraints.swap(execution_constraints);
      registrar.layout_constraints.swap(layout_constraints);
      registrar.add_constraint(ProcessorConstraint(kind));
      registrar.set_leaf(leaf);
      registrar.set_inner(inner);
      registrar.set_idempotent(idempotent);
      // Everyone is doing registration on their own nodes
      registrar.global_registration = false;
    }

    /*static*/ std::deque<LegateHello::PendingTaskVariant> &
    LegateHello::get_pending_task_variants(void) {
      static std::deque<PendingTaskVariant> pending_task_variants;
      return pending_task_variants;
    }

    // This is the function that we register with Legion to get a callback 
    // after the runtime is started but before the application begins which 
    // is when we'll get to do all our registrations
    void registration_callback(Machine machine, Runtime *runtime,
                               const std::set<Processor> &local_procs) {
      // This is the callback that we get from the runtime after it has started
      // but before the actual application starts running so we can now do all
      // our registrations.
      // Generate unique task IDs for this library, only need one task ID in this
      // case
      const TaskID hello_tid = runtime->generate_library_task_ids(
          hello_library_name, 1 /*only one task ID*/);
      std::deque<LegateHello::PendingTaskVariant> &pending_task_variants =
          LegateHello::get_pending_task_variants();
      // Do all our registrations
      for (std::deque<LegateHello::PendingTaskVariant>::iterator it =
               pending_task_variants.begin();
           it != pending_task_variants.end(); it++) {
        // Our library local task ID should be 0 in this case (see task.h)
        assert(it->task_id == 0);
        // Set the task ID to be the runtime generated one
        it->task_id = hello_tid;
        // Attach the task name too for debugging
        runtime->attach_name(hello_tid, it->task_name, false /*mutable*/,
                             true /*local only*/);
        // Perform the registration with the runtime
        runtime->register_task_variant(*it, it->descriptor, NULL, 0, it->ret_type,
                                       it->var);
      }
      pending_task_variants.clear();

      // Now we can generate a mapper ID for our library and register it with the
      // runtime
      const MapperID hello_mapper_id =
          runtime->generate_library_mapper_ids(hello_library_name, 1);
      // Register the mapper with all the local processors
      for (std::set<Processor>::const_iterator it = local_procs.begin();
           it != local_procs.end(); it++)
        runtime->add_mapper(
            hello_mapper_id,
            new HelloMapper(runtime->get_mapper_runtime(), machine, *it), *it);
    }

  } // namespace hello
} // namespace legate

#ifdef __cplusplus
extern "C" {
#endif

void legate_hello_perform_registration(void) {
  // Tell the runtime about our registration callback so we can register
  // ourselves Make sure it is global so this shared object always gets loaded
  // on all nodes
  Runtime::perform_registration_callback(
      legate::hello::registration_callback, true /*global*/);
}

#ifdef __cplusplus
}
#endif
