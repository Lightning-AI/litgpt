import pytest
import torch
from conftest import RunIf


@RunIf(min_cuda_gpus=1, thunder=True)
@pytest.mark.parametrize("reduction", ["none", "mean"])
def test_unsloth_cross_entropy(reduction):
    import thunder
    from thunder.core.transforms import grad

    from lightning_thunder.unsloth.executor import unsloth_ex

    logits = torch.randn(64, 128, device="cuda")
    labels = torch.randint(128, (64,), device="cuda")

    def foo(logits, labels):
        # this is the variant supported by unsloth.
        # if different arguments are used, the implementation would no be lowered to unsloth and instead would get
        # decomposed
        return torch.nn.functional.cross_entropy(logits, labels, reduction=reduction, ignore_index=-100)

    cfoo = thunder.jit(foo, executors=[unsloth_ex])
    actual = cfoo(logits, labels)
    trace_str = str(thunder.last_traces(cfoo)[-1])
    assert "unsloth_cross_entropy" in trace_str

    expected = foo(logits, labels)
    torch.testing.assert_close(actual, expected)

    logits.requires_grad_()
    cfoo_grad = grad(cfoo)
    actual = cfoo_grad(logits, labels)[0]
    trace_str = str(thunder.last_traces(cfoo_grad)[-1])
    assert "unsloth_cross_entropy_backward" in trace_str
    out = cfoo(logits, labels)
    out.sum().backward()
    expected = logits.grad
    torch.testing.assert_close(actual, expected)


@RunIf(min_cuda_gpus=1, thunder=True)
def test_unsloth_gpt():
    import thunder
    from thunder.core.transforms import grad

    from lightning_thunder.unsloth.executor import unsloth_ex
    from litgpt import GPT, Config
    from litgpt.utils import chunked_cross_entropy

    def forward_and_loss(model, input_ids, targets):
        logits = model(input_ids)
        return chunked_cross_entropy(logits, targets, chunk_size=0)

    cfn = thunder.jit(forward_and_loss, executors=[unsloth_ex])

    device = torch.device("cuda")
    config = Config(
        vocab_size=320,
        padding_multiple=64,
        n_layer=2,
        n_head=4,
        n_embd=64,
        rotary_percentage=1.0,
        parallel_residual=False,
        bias=False,
        norm_class_name="RMSNorm",
        mlp_class_name="LLaMAMLP",
        intermediate_size=1376,
    )
    with device:
        model = GPT(config)
        input_ids = torch.randint(1, 10, (2, 3))
        targets = torch.randint(0, 10, (2, 3))

    loss = cfn(model, input_ids, targets)
    assert isinstance(loss, torch.Tensor)

    fwd = thunder.last_traces(cfn)
    bwd = thunder.last_backward_traces(cfn)
    fwd_str, bwd_str = fwd[-1].python(), bwd[-1].python()
    print(fwd_str)
    print(bwd_str)

    assert "unsloth_cross_entropy" in fwd_str
    assert "unsloth_cross_entropy_backward" in bwd_str

    cfn_grad = grad(cfn)
    _ = cfn_grad(model, input_ids, targets)
    bwd = thunder.last_traces(cfn_grad)
    bwd_str = bwd[-1].python()
    assert "unsloth_cross_entropy_backward" in bwd_str


@RunIf(min_cuda_gpus=1, thunder=True)
def test_unsloth_rms_norm():
    import thunder
    from thunder.core.transforms import grad

    from lightning_thunder.unsloth.executor import ThunderRMSNorm, unsloth_ex
    from litgpt.model import RMSNorm

    with torch.device("cuda"):
        x = torch.randn(2, 16)
        rmsnorm = RMSNorm(16, eps=0.001)
    # monkeypatching was successful
    assert isinstance(rmsnorm, ThunderRMSNorm)

    if True:
        crmsnorm = thunder.jit(rmsnorm, executors=[unsloth_ex])
        actual = crmsnorm(x)
        fwd = thunder.last_traces(crmsnorm)
        bwd = thunder.last_backward_traces(crmsnorm)
        fwd_str, bwd_str = fwd[-1].python(), bwd[-1].python()
        print(fwd_str)
        print(bwd_str)
        # assert "litgpt_rmsnorm" in fwd_str
    else:
        rmsnorm.requires_grad_(False)
        crmsnorm = thunder.jit(rmsnorm, executors=[unsloth_ex])
        actual = crmsnorm(x)
        fwd = thunder.last_traces(crmsnorm)
        fwd_str = fwd[-1].python()
        print(fwd_str)
        # assert "litgpt_rmsnorm" in fwd_str

    expected = rmsnorm(x)
    torch.testing.assert_close(actual, expected)

    x.requires_grad_()

    return

    cfoo_grad = grad(cfoo)
    actual = cfoo_grad(logits, labels)[0]
    trace_str = str(thunder.last_traces(cfoo_grad)[-1])
    assert "unsloth_cross_entropy_backward" in trace_str
    out = cfoo(logits, labels)
    out.sum().backward()
    expected = logits.grad
    torch.testing.assert_close(actual, expected)
