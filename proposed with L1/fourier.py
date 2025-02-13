import cupy as cp


class Transform:

    def __init__(self, N):

       pass

    def fft(self, v):

        return cp.fft.fft2(v)

    def ifft(self, v):

        return cp.fft.ifft2(v).real