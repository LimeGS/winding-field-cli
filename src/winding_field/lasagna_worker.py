"""Private entrypoint executed by the pinned Python 3.14 Lasagna runtime."""

import torch


def run_tile_inference(self,model,tile,*,device):
    # CPU BF16 is emulated on some backends; retain FP32 there, BF16 on CUDA.
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type!='cpu'):
        output=model(tile)
    return output['output'] if isinstance(output,dict) else output


if __name__=='__main__':
    from contextlib import nullcontext
    import gpu_pause
    from lasagna.preprocess_cos_omezarr import LasagnaCosPredict3DAdapter,cli_main
    # The CLI owns its process; upstream's global training-service scheduler does not.
    gpu_pause.gpu_pause_context=nullcontext
    LasagnaCosPredict3DAdapter.run_tile_inference=run_tile_inference
    raise SystemExit(cli_main())
