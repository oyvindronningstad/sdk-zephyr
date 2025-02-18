#!/usr/bin/env python3
#
# Copyright (c) 2017 Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0

"""
Generation script for syscall_macros.h

The generation of macros for invoking system calls of various number
of arguments, in different execution types (supervisor only, user only,
mixed supervisor/user code) is tedious and repetitive. Rather than writing
by hand, this script generates it.

This script has no inputs, and emits the generated header to stdout.
"""

import sys
from enum import Enum


class Retval(Enum):
    VOID = 0
    U32 = 1
    U64 = 2


def gen_macro(ret, argc):
    if ret == Retval.VOID:
        suffix = "_VOID"
    elif ret == Retval.U64:
        suffix = "_RET64"
    else:
        suffix = ""

    sys.stdout.write("K_SYSCALL_DECLARE%d%s(id, name" % (argc, suffix))
    if (ret != Retval.VOID):
        sys.stdout.write(", ret")
    for i in range(argc):
        sys.stdout.write(", t%d, p%d" % (i, i))
    sys.stdout.write(")")


def gen_fn(ret, argc, name, extern=False):
    sys.stdout.write("\t%s %s %s(" %
                     (("extern" if extern else "static inline"),
                      ("ret" if ret != Retval.VOID else "void"), name))
    if argc == 0:
        sys.stdout.write("void")
    else:
        for i in range(argc):
            sys.stdout.write("t%d p%d" % (i, i))
            if i != (argc - 1):
                sys.stdout.write(", ")
    sys.stdout.write(")")


def tabs(count):
    sys.stdout.write("\t" * count)


def gen_make_syscall(ret, argc, tabcount):
    tabs(tabcount)

    # The core kernel is built with the --no-whole-archive linker option.
    # For all the individual .o files which make up the kernel, if there
    # are no external references to symbols within these object files,
    # everything in the object file is dropped.
    #
    # This has a subtle interaction with system call handlers. If an object
    # file has system call handler inside it, and nothing else in the
    # object file is referenced, then the linker will prefer the weak
    # version of the handler in the generated syscall_dispatch.c. The
    # user will get an "unimplemented system call" error if the associated
    # system call for that handler is made.
    #
    # Fix this by making a fake reference to the handler function at the
    # system call site. The address gets stored inside a special section
    # "hndlr_ref".  This is enough to prevent the handlers from being
    # dropped, and the hndlr_ref section is itself dropped from the binary
    # from gc-sections; these references will not consume space.

    sys.stdout.write(
        "static Z_GENERIC_SECTION(hndlr_ref) __used void *href = (void *)&z_hdlr_##name; \\\n")
    tabs(tabcount)
    if (ret != Retval.VOID):
        sys.stdout.write("return (ret)")
    else:
        sys.stdout.write("return (void)")
    if (argc <= 6 and ret != Retval.U64):
        sys.stdout.write("z_arch_syscall%s_invoke%d(" %
                     (("_ret64" if ret == Retval.U64 else ""), argc))
    else:
        sys.stdout.write("z_syscall%s_invoke%d(" %
                     (("_ret64" if ret == Retval.U64 else ""), argc))
    for i in range(argc):
        sys.stdout.write("(u32_t)p%d, " % (i))
    sys.stdout.write("id); \\\n")


def gen_call_impl(ret, argc):
    if (ret != Retval.VOID):
        sys.stdout.write("return ")
    sys.stdout.write("z_impl_##name(")
    for i in range(argc):
        sys.stdout.write("p%d" % (i))
        if i != (argc - 1):
            sys.stdout.write(", ")
    sys.stdout.write("); \\\n")


def newline():
    sys.stdout.write(" \\\n")


def gen_defines_inner(ret, argc, kernel_only=False, user_only=False):
    sys.stdout.write("#define ")
    gen_macro(ret, argc)
    newline()

    if not user_only:
        gen_fn(ret, argc, "z_impl_##name", extern=True)
        sys.stdout.write(";")
        newline()

    gen_fn(ret, argc, "name")
    newline()
    sys.stdout.write("\t{")
    newline()

    if kernel_only:
        sys.stdout.write("\t\t")
        gen_call_impl(ret, argc)
    elif user_only:
        gen_make_syscall(ret, argc, 2)
    else:
        sys.stdout.write("\t\tif (_is_user_context()) {")
        newline()

        gen_make_syscall(ret, argc, 3)

        sys.stdout.write("\t\t} else {")
        newline()

        # Prevent memory access issues if the implementation function gets
        # inlined
        sys.stdout.write("\t\t\tcompiler_barrier();")
        newline()

        sys.stdout.write("\t\t\t")
        gen_call_impl(ret, argc)
        sys.stdout.write("\t\t}")
        newline()

    sys.stdout.write("\t}\n\n")


def gen_defines(argc, kernel_only=False, user_only=False):
    gen_defines_inner(Retval.VOID, argc, kernel_only, user_only)
    gen_defines_inner(Retval.U32, argc, kernel_only, user_only)
    gen_defines_inner(Retval.U64, argc, kernel_only, user_only)


sys.stdout.write(
    "/* Auto-generated by gen_syscall_header.py, do not edit! */\n\n")
sys.stdout.write("#ifndef GEN_SYSCALL_H\n#define GEN_SYSCALL_H\n\n")
sys.stdout.write("#include <syscall.h>\n")

for i in range(11):
    sys.stdout.write(
        "#if !defined(CONFIG_USERSPACE) || defined(__ZEPHYR_SUPERVISOR__)\n")
    gen_defines(i, kernel_only=True)
    sys.stdout.write("#elif defined(__ZEPHYR_USER__)\n")
    gen_defines(i, user_only=True)
    sys.stdout.write("#else /* mixed kernel/user macros */\n")
    gen_defines(i)
    sys.stdout.write("#endif /* mixed kernel/user macros */\n\n")

sys.stdout.write("#endif /* GEN_SYSCALL_H */\n")
