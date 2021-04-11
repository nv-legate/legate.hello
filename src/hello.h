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

#ifndef __LEGATE_HELLO_H__
#define __LEGATE_HELLO_H__

#include "legate.h"
#include <deque>

namespace legate {
  namespace hello {

    // We'll have a general class with static members to do our library start-up
    class LegateHello {
    public:
      // We'll call this method from of our tasks to build up the
      // list of pending task variants to register after the runtime
      // is started and we can get our library IDs
      static void record_variant(Legion::TaskID tid, const char *task_name,
                                 const Legion::CodeDescriptor &desc,
                                 Legion::ExecutionConstraintSet &execution_constraints,
                                 Legion::TaskLayoutConstraintSet &layout_constraints,
                                 LegateVariant var, Legion::Processor::Kind kind, 
                                 bool leaf, bool inner, bool idempotent, bool ret_type);
    public:
      // This is a class for storing our task variants until they are ready
      // to be registered once the runtime has started and we have IDs for them
      struct PendingTaskVariant : public Legion::TaskVariantRegistrar {
      public:
        PendingTaskVariant(void)
            : TaskVariantRegistrar(), task_name(NULL), var(LEGATE_NO_VARIANT) {}
        PendingTaskVariant(Legion::TaskID tid, bool global, const char *var_name,
                           const char *t_name, const Legion::CodeDescriptor &desc,
                           LegateVariant v, bool ret)
            : TaskVariantRegistrar(tid, global, var_name), task_name(t_name),
              descriptor(desc), var(v), ret_type(ret) {}

      public:
        const char *task_name;
        Legion::CodeDescriptor descriptor;
        LegateVariant var;
        bool ret_type;
      };
      // Make sure you make the singleton like this with a function or bad things
      // happen at initialization time
      static std::deque<PendingTaskVariant> &get_pending_task_variants(void);
    };

  } // namespace hello
} // namespace legate

#endif // __LEGATE_HELLO__
