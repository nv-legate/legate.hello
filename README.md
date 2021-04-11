<!--
Copyright 2021 NVIDIA Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-->

# Legate Hello World

This repository contains a "Hello World" introduction to building a small Legate library.

Users can begin by investigating the `legate/hello/runtime.py` file which demonstrates
the necessary code for setting up a `LegateLibrary` object, launching Legion tasks to
perform computations, and returning data objects that implement the `__legate_data_interface__`.

From there, users can investigate the files in the `src/` directory. The `src/hello*` files
demonstrate how to register tasks with the Legion runtime using Legate APIs. Implementations of
various different task variants for CPUs, OpenMP, and GPU processors can be found in
`src/task.cc` and `src/task.cu`. The `src/mapper.cc` file shows a small customization to
a Legion mapper for mapping the tasks produced by the Legate Hello world example.

There is a test program in `tests/hello.py` that will demonstrate running the Hello 
World library code in practice.

The `install.py` and `setup.py` scripts in this directory coupled with `src/Makefile` will
provide a good framework for setting up a build system for any Legate library.

