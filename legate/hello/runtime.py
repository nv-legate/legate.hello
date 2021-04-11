# Copyright 2021 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import, division, print_function

import os

import numpy as np
import pyarrow

from legate.core import (
    BufferBuilder,
    Future,
    IndexTask,
    LegateArray,
    LegateLibrary,
    LegateStore,
    Rect,
    Task,
    get_legion_context,
    get_legion_runtime,
    legate_add_library,
    legion,
)


class HelloFuture(LegateStore):
    def __init__(self, future):
        self._future = future
        self._legate_data = None

    @property
    def __legate_data_interface__(self):
        # By implementing the Legate data interface these objects can be
        # passed into other Legate libraries and have the future data be
        # consumed without requiring synchronization
        if self._legate_data is None:
            arrow_type = pyarrow.from_numpy_dtype(np.dtype(np.int32))
            # Return ourselves as the LegateStore for this LegateArray
            array = LegateArray(arrow_type, [None, self])
            self._legate_data = dict()
            self._legate_data["version"] = 1
            data = dict()
            field = pyarrow.field(
                "Legate NumPy Array", arrow_type, nullable=False
            )
            data[field] = array
            self._legate_data["data"] = data
        return self._legate_data

    @property
    def type(self):
        # Return the type for this LegateStore
        return pyarrow.from_numpy_dtype(np.dtype(np.int32))

    @property
    def kind(self):
        # Return the kind of the Legion data for this LegateStore
        return Future

    @property
    def storage(self):
        # Return our future as the storage for this LegateStore
        return self._future

    def __int__(self):
        # If users actually need the data then we block on the
        # future and get its data to return
        return int(
            np.frombuffer(self._future.get_buffer(4), dtype=np.int32, count=1)[
                0
            ]
        )

    def __str__(self):
        return str(int(self))


class Runtime(LegateLibrary):
    def __init__(self, name, runtime):
        self._name = name
        self._context = None
        self._runtime = runtime
        self._task_id = legion.legion_runtime_generate_library_task_ids(
            self._runtime, self._name.encode("utf-8"), 1
        )
        self._mapper_id = legion.legion_runtime_generate_library_mapper_ids(
            self._runtime, self._name.encode("utf-8"), 1
        )
        legate_add_library(self)

    def get_name(self):
        return self._name

    def get_shared_library(self):
        from legate.hello.install_info import libpath

        return os.path.join(
            libpath, "liblghello" + self.get_library_extension()
        )

    def get_c_header(self):
        from legate.hello.install_info import header

        return header

    def get_registration_callback(self):
        return "legate_hello_perform_registration"

    def destroy(self):
        print(self._name + " is destroyed")

    def initialize(self, hello_lib):
        self._hello_lib = hello_lib

    def launch_hello_task(self, count):
        argbuf = BufferBuilder()
        argbuf.pack_32bit_int(1)
        argbuf.pack_string("World")
        if self._context is None:
            self._context = get_legion_context()
        if count == 1:
            task = Task(
                self._task_id,
                data=argbuf.get_string(),
                size=argbuf.get_size(),
                mapper=self._mapper_id,
            )
            return HelloFuture(task.launch(self._runtime, self._context))
        else:
            rect = Rect([count])
            task = IndexTask(
                self._task_id,
                rect,
                data=argbuf.get_string(),
                size=argbuf.get_size(),
                mapper=self._mapper_id,
            )
            future_map = task.launch(self._runtime, self._context)
            return HelloFuture(
                future_map.reduce(
                    self._context,
                    self._runtime,
                    legion.LEGION_REDOP_BASE
                    + legion.LEGION_REDOP_KIND_SUM * legion.LEGION_TYPE_TOTAL
                    + legion.LEGION_TYPE_INT32,
                )
            )


runtime = Runtime("legate.hello", get_legion_runtime())


def launch_hello_task(count=1):
    return runtime.launch_hello_task(count)
