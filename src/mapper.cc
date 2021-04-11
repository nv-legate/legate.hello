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

#include "mapper.h"

using namespace Legion;
using namespace Legion::Mapping;

namespace legate {
  namespace hello {

    HelloMapper::HelloMapper(MapperRuntime *rt, Machine m, Processor p)
        : Legion::Mapping::DefaultMapper(rt, m, p, create_hello_mapper_name(p),
                                         true /*take ownership of the name*/) {}

    /*static*/ const char *HelloMapper::create_hello_mapper_name(Processor p) {
      const size_t buffer_size = 64;
      char *result = (char *)malloc(buffer_size * sizeof(char));
      snprintf(result, buffer_size - 1, "Hello Mapper on Processor " IDFMT "",
               p.id);
      return result;
    }

  } // namespace hello
} // namespace legate
