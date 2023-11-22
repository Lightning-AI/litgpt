# this file is just to validate on the CI logs that these tests were run
from conftest import RunIf


@RunIf(min_cuda_gpus=1)
def test_runif_min_cuda_gpus():
    assert True


@RunIf(min_cuda_gpus=1, standalone=True)
def test_runif_min_cuda_gpus_standalone():
    assert True


@RunIf(standalone=True)
def test_runif_standalone():
    assert True
