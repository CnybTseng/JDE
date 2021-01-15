import torch

def move_tensors_to_gpu(tensors, device=None, non_blocking=False):
    '''Move tensors to GPU.
    
    Param
    -----
    tensors     : List of torch.Tensor tensors.
    device      : The destination GPU device. Defaults to the current CUDA device.
    non_blocking: If True and the source is in pinned memory,
                  the copy will be asynchronous with respect to the host.
                  Otherwise, the argument has no effect. Default: False.
    '''
    if not isinstance(tensors, list):
        raise TypeError('expect list, but got {}'.format(type(tensors)))
    gpu_tensors = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError('expect torch.Tensor, but got {}'.format(type(tensor)))
        gpu_tensors.append(tensor.cuda(device=device, non_blocking=non_blocking))
    return gpu_tensors