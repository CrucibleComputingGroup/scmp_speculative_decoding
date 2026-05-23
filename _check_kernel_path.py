"""Sanity-check that scmp_kernels is importable and which file backs sc_matmul.

Run after activating the environment:  python _check_kernel_path.py
Mirrors scmp_llm_llama/_check_kernel_path.py.
"""
import scmp_kernels
import scmp_kernels.sc.matmul as mm

print("scmp_kernels.__file__ :", scmp_kernels.__file__)
print("sc.matmul.__file__    :", mm.__file__)
print("sc_matmul source      :", mm.sc_matmul.__code__.co_filename)
