import numpy as np
from aie.ir import Module
from aie.iron import ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2

N = 1024
TILE_SIZE = 32


def addone_design() -> Module:
    vec_ty = np.ndarray[(N,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in0")
    of_out = ObjectFifo(tile_ty, name="out0")

    def core_fn(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(TILE_SIZE):
            elem_out[i] = elem_in[i] + 1
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_fn, fn_args=[of_in.cons(), of_out.prod()])

    in_handle = of_in.prod()
    out_handle = of_out.cons()

    def seq_fn(inp, out, _):
        tg = TaskGroup()
        in_handle.fill(inp, group=tg)
        out_handle.drain(out, wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq_fn, [vec_ty, vec_ty, [in_handle, out_handle]])
    return Program(NPU2(), rt, workers=[worker]).resolve_program()


if __name__ == "__main__":
    print(addone_design())
