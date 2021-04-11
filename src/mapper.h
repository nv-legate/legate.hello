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

#ifndef __HELLO_MAPPER_H__
#define __HELLO_MAPPER_H__

#include "hello.h"
#include "mappers/default_mapper.h"

namespace legate {
  namespace hello {

    // Our hello mapper is just the default mapper in disguise, but we
    // could later easily override any of the virtual methods for mapping
    class HelloMapper : public Legion::Mapping::DefaultMapper {
    public:
      HelloMapper(Legion::Mapping::MapperRuntime *runtime, 
          Legion::Machine machine, Legion::Processor local);
    public:
      static const char *create_hello_mapper_name(Legion::Processor p);
    };

  } // namespace hello
} // namespace legate

#endif // __HELLO_MAPPER_H__
