"""Compile the packaged kernel with NVRTC and launch on PyTorch's current stream."""
from __future__ import annotations
import ctypes as C
import ctypes.util
import os
from pathlib import Path
import threading

import torch

_lock = threading.Lock()
_kernels = {}


def _libraries():
    directory = Path(torch.__file__).parent / 'lib'
    if os.name == 'nt':
        # Retain the handle for NVRTC's adjacent builtins DLL for the process lifetime.
        handle = os.add_dll_directory(str(directory))
        candidates = sorted(directory.glob('nvrtc64_*.dll'))
        candidates = [p for p in candidates if '.alt.' not in p.name]
        if not candidates:
            raise RuntimeError('CUDA backend requires NVRTC from the CUDA-enabled PyTorch distribution')
        nvrtc = C.CDLL(str(candidates[0]))
        nvrtc._vlade_dll_directory = handle
        driver = C.WinDLL(str(Path(os.environ['SystemRoot']) / 'System32/nvcuda.dll'))
    else:
        name = ctypes.util.find_library('nvrtc')
        candidates = list(directory.glob('libnvrtc.so*'))
        candidates += list((Path(torch.__file__).parents[1] / 'nvidia/cuda_nvrtc/lib').glob('libnvrtc.so*'))
        if not name and not candidates:
            raise RuntimeError('CUDA backend requires libnvrtc')
        nvrtc = C.CDLL(name or str(candidates[0]))
        driver = C.CDLL(ctypes.util.find_library('cuda') or 'libcuda.so.1')
    return nvrtc, driver


def _check(code, operation):
    if code:
        raise RuntimeError(f'{operation} failed with CUDA/NVRTC status {code}')


def _kernel(device):
    key = (device.index, torch.cuda.get_device_capability(device))
    with _lock:
        if key in _kernels:
            return _kernels[key]
        nvrtc, driver = _libraries()
        program = C.c_void_p()
        source = Path(__file__).with_name('kernel.cu').read_bytes()
        nvrtc.nvrtcCreateProgram.argtypes = [C.POINTER(C.c_void_p), C.c_char_p, C.c_char_p, C.c_int, C.c_void_p, C.c_void_p]
        _check(nvrtc.nvrtcCreateProgram(C.byref(program), source, b'vlade_packed.cu', 0, None, None), 'nvrtcCreateProgram')
        major, minor = key[1]
        options = (C.c_char_p * 2)(b'--std=c++14', f'--gpu-architecture=compute_{major}{minor}'.encode())
        nvrtc.nvrtcCompileProgram.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_char_p)]
        try:
            status = nvrtc.nvrtcCompileProgram(program, 2, options)
            if status:
                size = C.c_size_t()
                nvrtc.nvrtcGetProgramLogSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
                nvrtc.nvrtcGetProgramLog.argtypes = [C.c_void_p, C.c_void_p]
                nvrtc.nvrtcGetProgramLogSize(program, C.byref(size))
                log = C.create_string_buffer(size.value)
                nvrtc.nvrtcGetProgramLog(program, log)
                raise RuntimeError('NVRTC packed kernel compilation failed: ' + log.value.decode())
            size = C.c_size_t()
            nvrtc.nvrtcGetPTXSize.argtypes = [C.c_void_p, C.POINTER(C.c_size_t)]
            nvrtc.nvrtcGetPTX.argtypes = [C.c_void_p, C.c_void_p]
            _check(nvrtc.nvrtcGetPTXSize(program, C.byref(size)), 'nvrtcGetPTXSize')
            ptx = C.create_string_buffer(size.value)
            _check(nvrtc.nvrtcGetPTX(program, ptx), 'nvrtcGetPTX')
        finally:
            nvrtc.nvrtcDestroyProgram.argtypes = [C.POINTER(C.c_void_p)]
            nvrtc.nvrtcDestroyProgram(C.byref(program))
        driver.cuInit.argtypes = [C.c_uint]
        _check(driver.cuInit(0), 'cuInit')
        module, function = C.c_void_p(), C.c_void_p()
        driver.cuModuleLoadData.argtypes = [C.POINTER(C.c_void_p), C.c_void_p]
        driver.cuModuleGetFunction.argtypes = [C.POINTER(C.c_void_p), C.c_void_p, C.c_char_p]
        _check(driver.cuModuleLoadData(C.byref(module), ptx), 'cuModuleLoadData')
        _check(driver.cuModuleGetFunction(C.byref(function), module, b'vlade_packed_linear'), 'cuModuleGetFunction')
        driver.cuLaunchKernel.argtypes = [C.c_void_p, *([C.c_uint] * 7), C.c_void_p, C.POINTER(C.c_void_p), C.c_void_p]
        result = (driver, nvrtc, module, function)
        _kernels[key] = result  # Keep module/library handles alive; one module per device/context.
        return result


def linear(x, packed):
    if not x.is_cuda or any(t.device != x.device for t in (packed.codes, packed.bits, packed.offsets, packed.scales, packed.bias)):
        raise ValueError('CUDA packed operands must be on the same CUDA device')
    expected = ((packed.codes,torch.uint8),(packed.bits,torch.uint8),(packed.offsets,torch.int64),
                (packed.scales,torch.float32),(packed.bias,torch.float32))
    if any(t.dtype != dtype or not t.is_contiguous() for t,dtype in expected):
        raise ValueError('CUDA packed buffers must retain their validated dtype and contiguous layout')
    x = x.contiguous()
    if x.dtype != torch.float32 or x.ndim != 2:
        raise ValueError('CUDA kernel receives contiguous float32 activations')
    if x.shape[0] * packed.out_features > 2**31-1 or max(x.shape[0], packed.in_features, packed.out_features) > 2**31-1:
        raise ValueError('Packed CUDA launch dimensions exceed supported range')
    with torch.cuda.device(x.device):
        driver, _nvrtc, _module, function = _kernel(x.device)
        output = torch.empty((len(x), packed.out_features), device=x.device, dtype=torch.float32)
        values = [C.c_void_p(t.data_ptr()) for t in (x, packed.codes, packed.offsets, packed.bits, packed.scales, packed.bias, output)]
        values += [C.c_int(v) for v in (len(x), packed.out_features, packed.in_features, packed.group_size, packed.scales.shape[1])]
        arguments = (C.c_void_p * len(values))(*(C.cast(C.pointer(v), C.c_void_p) for v in values))
        stream = torch.cuda.current_stream(x.device)
        _check(driver.cuLaunchKernel(function, len(x) * packed.out_features, 1, 1, 128, 1, 1, 0,
                                     C.c_void_p(stream.cuda_stream), arguments, None), 'cuLaunchKernel')
        for tensor in (x, packed.codes, packed.offsets, packed.bits, packed.scales, packed.bias, output):
            tensor.record_stream(stream)
    return output
