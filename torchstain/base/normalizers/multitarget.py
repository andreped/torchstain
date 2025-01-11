def MultiMacenkoNormalizer(backend="torch", **kwargs):
    if backend == "numpy":
        from torchstain.numpy.normalizers import NumpyMultiMacenkoNormalizer
        return NumpyMultiMacenkoNormalizer(**kwargs)
    elif backend == "torch":
        from torchstain.torch.normalizers import TorchMultiMacenkoNormalizer
        return TorchMultiMacenkoNormalizer(**kwargs)
    else:
        raise Exception(f"Unsupported backend {backend}")
