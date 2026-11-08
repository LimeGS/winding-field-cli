"""Private entrypoint executed by the pinned Python 3.14 Lasagna runtime."""

import torch


def run_tile_inference(self,model,tile,*,device):
    # Pre-Ampere CUDA (including GTX1660) cannot run native BF16 kernels.
    bf16=device.type=='cuda' and torch.cuda.get_device_capability(device)[0]>=8
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=bf16):
        output=model(tile)
    return output['output'] if isinstance(output,dict) else output


def output_size(size,factor):
    # Gaussian decimation keeps index zero, including the last odd-grid sample.
    return max(1,(int(size)+int(factor)-1)//int(factor))


def bounded_main(checkpoint,device_name):
    import contextlib
    import sys
    import numpy as np
    from bounded_lasagna import read_frame,write_frame,finalize
    from pathlib import Path
    # The script directory contains lasagna.py, which shadows the installed package.
    sys.path[:]=[p for p in sys.path if Path(p).resolve()!=Path(__file__).resolve().parent]
    wire=sys.stdout.buffer
    # Upstream model-loading diagnostics must not enter the binary protocol.
    with contextlib.redirect_stdout(sys.stderr):
        from lasagna.preprocess_cos_omezarr import (
            LasagnaCosPredict3DAdapter,OutputProductSpec,OutputChannelSpec,
            PYRAMID_POLICY_SCALAR,PYRAMID_POLICY_CUSTOM)
        from lasagna.tiled_predict3d import _input_tile_to_device
        cos=OutputProductSpec(name='cos',level=0,scaledown=1,
            channels=(OutputChannelSpec('cos',relative_path='unused-cos'),),chunk_size=32,
            pyramid_policy=PYRAMID_POLICY_SCALAR,inference_scaledown=1)
        normal=OutputProductSpec(name='lasagna_normal',level=0,scaledown=1,
            channels=tuple(OutputChannelSpec(n,relative_path='unused-'+n) for n in ('grad_mag','nx','ny')),
            chunk_size=32,pyramid_policy=PYRAMID_POLICY_CUSTOM,accumulator_channel_count=7,inference_scaledown=1)
        adapter=LasagnaCosPredict3DAdapter(checkpoint=checkpoint,tile_size=64,device_name=device_name,
                                         cos_product=cos,normal_product=normal)
        device=torch.device(device_name);model=None
        while operation:=sys.stdin.buffer.read(1):
            value=read_frame(sys.stdin.buffer)
            if operation==b'P':
                if value.shape!=(64,)*3 or value.dtype!=np.uint8:raise ValueError('invalid canonical CT tile')
                if model is None:model=adapter.load_model(device=device)
                with torch.inference_mode():
                    raw=run_tile_inference(adapter,model,_input_tile_to_device(value,device),device=device)
                    products=adapter.product_tensors_from_output(raw)
                    result=torch.cat([products['cos'],products['lasagna_normal']],dim=1)[0].cpu().numpy()
            elif operation==b'F':result=finalize(adapter,value)
            else:raise ValueError('unknown bounded Lasagna operation')
            write_frame(wire,result)


if __name__=='__main__':
    import sys
    if len(sys.argv)==4 and sys.argv[1]=='--bounded':
        bounded_main(sys.argv[2],sys.argv[3])
        raise SystemExit(0)
    from contextlib import nullcontext
    import gpu_pause
    import numpy as np
    import zarr
    import lasagna.preprocess_cos_omezarr as upstream
    from lasagna.preprocess_cos_omezarr import LasagnaCosPredict3DAdapter,cli_main
    tiled=sys.modules[upstream.run_tiled_inference_3d.__module__]
    upstream._ds_size=output_size
    tiled._ds_size=output_size
    # Upstream reduced weight views have strided X, rejected by its native adder.
    accumulate=tiled._add_accumulator_view
    tiled._add_accumulator_view=lambda destination,source:accumulate(destination,np.ascontiguousarray(source))
    # Persist zero chunks so absent files still mean incomplete/corrupt output.
    zarr.config.set({'array.write_empty_chunks': True})
    # The CLI owns its process; upstream's global training-service scheduler does not.
    gpu_pause.gpu_pause_context=nullcontext
    LasagnaCosPredict3DAdapter.run_tile_inference=run_tile_inference
    raise SystemExit(cli_main())
