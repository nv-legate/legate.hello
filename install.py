#!/usr/bin/env python

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

from __future__ import print_function

import argparse
import json
import multiprocessing
import os
import subprocess
import sys

_version = sys.version_info.major


def git_clone(repo_dir, url, branch=None):
    subprocess.check_call(
        ["git", "clone"] + (["-b", branch] if branch else []) + [url, repo_dir]
    )


def git_reset(repo_dir, refspec):
    subprocess.check_call(["git", "reset", "--hard", refspec], cwd=repo_dir)


def git_update(repo_dir, branch=None):
    subprocess.check_call(["git", "pull", "--ff-only"], cwd=repo_dir)
    if branch is not None:
        subprocess.check_call(["git", "checkout", branch], cwd=repo_dir)


def load_json_config(filename):
    try:
        with open(filename, "r") as f:
            return json.load(f)
    except IOError:
        return None


def dump_json_config(filename, value):
    with open(filename, "w") as f:
        return json.dump(value, f)


def get_cmake_config(cmake, legate_dir, default=None):
    config_filename = os.path.join(legate_dir, ".cmake.json")
    if cmake is None:
        cmake = load_json_config(config_filename)
        if cmake is None:
            cmake = default
    assert cmake in [True, False]
    dump_json_config(config_filename, cmake)
    return cmake


def find_c_define(define, header):
    with open(header, "r") as f:
        line = f.readline()
        while line:
            line = line.rstrip()
            if line.startswith("#define") and define in line.split(" "):
                return True
            line = f.readline()
    return False


def build_legate_hello(
    legate_hello_dir,
    install_dir,
    cmake,
    cmake_exe,
    debug,
    clean_first,
    thread_count,
    verbose,
    unknown,
):
    src_dir = os.path.join(legate_hello_dir, "src")
    if cmake:
        print("Warning: CMake is currently not supported for Legate build.")
        print("Using GNU Make for now.")
    make_flags = [
        "LEGATE_DIR=%s" % install_dir,
        "DEBUG=%s" % (1 if debug else 0),
        "PREFIX=%s" % install_dir,
    ] + (["GCC=%s" % os.environ["CXX"]] if "CXX" in os.environ else [])
    if clean_first:
        subprocess.check_call(["make"] + make_flags + ["clean"], cwd=src_dir)
    subprocess.check_call(
        ["make"] + make_flags + ["-j", str(thread_count), "install"],
        cwd=src_dir,
    )
    cmd = ["python", "setup.py", "install", "--recurse"]
    if unknown is not None:
        cmd += unknown
        if "--prefix" not in unknown:
            cmd += ["--prefix", str(install_dir)]
    else:
        cmd += ["--prefix", str(install_dir)]
    subprocess.check_call(cmd, cwd=legate_hello_dir)


def install(
    cmake=None,
    cmake_exe=None,
    legate_dir=None,
    legion_dir=None,
    debug=False,
    clean_first=False,
    extra_flags=[],
    thread_count=None,
    verbose=False,
    unknown=None,
):
    legate_hello_dir = os.path.dirname(os.path.realpath(__file__))

    cmake = get_cmake_config(cmake, legate_hello_dir, default=False)

    if clean_first is None:
        clean_first = not cmake

    thread_count = thread_count
    if thread_count is None:
        thread_count = multiprocessing.cpu_count()

    # Check to see if we installed Legate Core
    legate_config = os.path.join(legate_hello_dir, ".legate.core.json")
    if "LEGATE_DIR" in os.environ:
        legate_dir = os.environ["LEGATE_DIR"]
    elif legate_dir is None:
        legate_dir = load_json_config(legate_config)
    if legate_dir is None or not os.path.exists(legate_dir):
        raise Exception("You need to provide a Legate Core installation")
    legate_dir = os.path.realpath(legate_dir)
    dump_json_config(legate_config, legate_dir)

    build_legate_hello(
        legate_hello_dir,
        legate_dir,
        cmake,
        cmake_exe,
        debug,
        clean_first,
        thread_count,
        verbose,
        unknown,
    )


def driver():
    parser = argparse.ArgumentParser(description="Install Legate Hello")
    parser.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        required=False,
        default=os.environ.get("DEBUG") == "1",
        help="Build Legate with debugging enabled.",
    )
    parser.add_argument(
        "--with-core",
        dest="legate_dir",
        metavar="DIR",
        required=False,
        help="Path to Legate Core installation directory.",
    )
    parser.add_argument(
        "--cmake",
        dest="cmake",
        action="store_true",
        required=False,
        default=os.environ["USE_CMAKE"] == "1"
        if "USE_CMAKE" in os.environ
        else None,
        help="Build Legate with CMake.",
    )
    parser.add_argument(
        "--no-cmake",
        dest="cmake",
        action="store_false",
        required=False,
        help="Don't build Legate with CMake (instead use GNU Make).",
    )
    parser.add_argument(
        "--with-cmake",
        dest="cmake_exe",
        metavar="EXE",
        required=False,
        default="cmake",
        help="Path to CMake executable (if not on PATH).",
    )
    parser.add_argument(
        "--clean",
        dest="clean_first",
        action="store_true",
        required=False,
        default=None,
        help="Clean before build.",
    )
    parser.add_argument(
        "--no-clean",
        "--noclean",
        dest="clean_first",
        action="store_false",
        required=None,
        help="Skip clean before build.",
    )
    parser.add_argument(
        "--extra",
        dest="extra_flags",
        action="append",
        required=False,
        default=[],
        help="Extra flags for make command.",
    )
    parser.add_argument(
        "-j",
        dest="thread_count",
        nargs="?",
        type=int,
        help="Number threads used to compile.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        required=False,
        help="Enable verbose build output.",
    )
    args, unknown = parser.parse_known_args()

    install(unknown=unknown, **vars(args))


if __name__ == "__main__":
    driver()
