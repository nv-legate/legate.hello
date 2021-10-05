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

ifndef LEGATE_DIR
$(error LEGATE_DIR variable is not defined, aborting build)
endif

# The name of this library being built
LIBNAME = liblghello

# General source files
GEN_CPU_SRC	= task.cc \
		  mapper.cc \
		  hello.cc
# No initial GPU files for core Legate but libraries have them
GEN_GPU_SRC	= task.cu
# Any source headers paths and headers that need to be installed
INSTALL_HEADERS =

include $(LEGATE_DIR)/share/legate/legate.mk
