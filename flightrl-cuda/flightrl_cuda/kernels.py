"""Fixed-storage bindings for the two optional RL kernels, on Torch's stream."""
import ctypes as ct
import operator
from pathlib import Path
import torch

_library = None


def storage_signature(t):
    return (t.data_ptr(),tuple(t.shape),t.stride(),t.dtype,t.device,t.requires_grad)


def library():
    global _library
    if _library is None:
        paths=list(Path(__file__).parent.glob('_kernels*.so'))
        if len(paths)!=1: raise RuntimeError('install/build flightrl-cuda native kernels first')
        lib=ct.CDLL(str(paths[0]))
        if lib.rl_kernel_abi()!=1: raise RuntimeError('RL kernel ABI mismatch')
        lib.rl_stats.argtypes=[ct.c_void_p]*7+[ct.c_int,ct.c_bool,ct.c_void_p]
        lib.rl_gae.argtypes=[ct.c_void_p]*7+[ct.c_int,ct.c_int,ct.c_float,ct.c_float,ct.c_void_p]
        _library=lib
    return _library


class BoundKernel:
    def __init__(self, name, tensors, dtypes, shapes):
        if name not in ('rl_stats','rl_gae'): raise ValueError('unknown RL kernel')
        if len(tensors)!=7 or not len(tensors)==len(dtypes)==len(shapes):
            raise ValueError('incomplete kernel schema')
        self.name=name
        self.capacity=tuple(shapes[0])
        if any(not 0 < size <= 2147483647 for size in self.capacity):
            raise ValueError('RL dimensions must fit positive int32')
        if name=='rl_stats':
            expected_types=(torch.float32,torch.uint8,torch.uint8,torch.float32,torch.int64,torch.float32,torch.int64)
            expected_shapes=[self.capacity]*7
            rank=1
        else:
            expected_types=(torch.float32,torch.float32,torch.uint8,torch.uint8,torch.float32,torch.float32,torch.float32)
            expected_shapes=[self.capacity]*4+[self.capacity[-1:]]+[self.capacity]*2
            rank=2
        if len(self.capacity)!=rank or tuple(dtypes)!=expected_types or list(map(tuple,shapes))!=expected_shapes:
            raise ValueError('invalid RL kernel schema')
        self.fn=getattr(library(),name)
        self.tensors=tuple(tensors); self.device=tensors[0].device
        self.stream=torch.cuda.current_stream(self.device)
        ranges=[]
        for t,dtype,shape in zip(tensors,dtypes,shapes):
            if t.device.type!='cuda' or t.device!=self.device or t.dtype!=dtype or tuple(t.shape)!=shape or not t.is_contiguous() or t.requires_grad:
                raise ValueError('invalid RL kernel tensor binding')
            begin,end=t.data_ptr(),t.data_ptr()+t.numel()*t.element_size()
            if any(begin<b and a<end for a,b in ranges): raise ValueError('aliased RL kernel binding')
            ranges.append((begin,end))
        self.pointers=tuple(t.data_ptr() for t in tensors)
        self.shapes=tuple(tuple(t.shape) for t in tensors)
        self.strides=tuple(t.stride() for t in tensors)
        self.dtypes=tuple(t.dtype for t in tensors)

    def run(self, *args):
        # ctypes silently truncates Python integers to C int. Validate dimensions
        # against the actual binding before pointer submission, including stride.
        try:
            n=operator.index(args[0])
            if self.name=='rl_stats':
                valid=len(args)==2 and 0<n<=self.capacity[0]
            else:
                valid=(len(args)==4 and n==self.capacity[1] and
                       operator.index(args[1])==self.capacity[0] and
                       0<=args[2]<=1 and 0<=args[3]<=1)
        except (IndexError,TypeError):
            valid=False
        if not valid: raise ValueError('RL launch dimensions or parameters exceed binding')
        # Reject reallocation or metadata mutation rather than using stale pointers.
        for t,p,shape,stride,dtype in zip(self.tensors,self.pointers,self.shapes,self.strides,self.dtypes):
            if (t.data_ptr()!=p or tuple(t.shape)!=shape or t.stride()!=stride or
                t.dtype!=dtype or t.device!=self.device or t.requires_grad):
                raise ValueError('RL bound storage changed')
        if torch.cuda.current_stream(self.device)!=self.stream:
            raise ValueError('submit on the RL binding construction stream')
        with torch.cuda.device(self.device):
            error=self.fn(*self.pointers,*args,torch.cuda.current_stream(self.device).cuda_stream)
        if error: raise RuntimeError('RL CUDA launch failed: '+str(error))
