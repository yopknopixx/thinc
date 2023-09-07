import numpy
import pytest

from thinc.api import (
    SGD,
    ArgsKwargs,
    CupyOps,
    Linear,
    MPSOps,
    NumpyOps,
    IvyWrapper,
    Relu,
    chain,
    get_current_ops,
    ivy2xp,
    use_ops,
    xp2ivy,
)

from thinc.backends import context_pools
from thinc.compat import has_cupy_gpu, has_ivy

from thinc.layers.ivywrapper import IvyWrapper
from thinc.shims.ivy import (
    default_deserialize_ivy_model,
    default_serialize_ivy_model,
)

from thinc.util import get_ivy_default_device

from ..util import check_input_converters, make_tempdir


XP_OPS = [NumpyOps()]
if has_cupy_gpu:
    XP_OPS.append(CupyOps())


def check_learns_zero_output(model, sgd, X, Y):
    """Check we can learn to output a zero vector"""
    Yh, get_dX = model.begin_update(X)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh)
    model.finish_update(sgd)
    prev = numpy.abs(Yh.sum())
    for i in range(100):
        Yh, get_dX = model.begin_update(X)
        total = numpy.abs(Yh.sum())
        dX = get_dX(Yh - Y)  # noqa: F841
        model.finish_update(sgd)
    assert total < prev


@pytest.mark.skipif(not has_ivy, reason="needs Ivy")
@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_ivy_unwrapped(nN, nI, nO):
    model = Linear(nO, nI).initialize()
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    sgd = SGD(0.01)
    Y = numpy.zeros((nN, nO), dtype="f")
    check_learns_zero_output(model, sgd, X, Y)


@pytest.mark.skipif(not has_ivy, reason="needs Ivy")
@pytest.mark.parametrize("nN,nI,nO", [(2, 3, 4)])
def test_ivy_wrapper(nN, nI, nO):
    import ivy

    model = IvyWrapper(ivy.Linear(nI, nO)).initialize()
    sgd = SGD(0.001)
    X = numpy.zeros((nN, nI), dtype="f")
    X += numpy.random.uniform(size=X.size).reshape(X.shape)
    Y = numpy.zeros((nN, nO), dtype="f")
    Yh, get_dX = model.begin_update(X)
    assert isinstance(Yh, numpy.ndarray)
    assert Yh.shape == (nN, nO)
    dYh = (Yh - Y) / Yh.shape[0]
    dX = get_dX(dYh)
    model.finish_update(sgd)
    assert dX.shape == (nN, nI)
    check_learns_zero_output(model, sgd, X, Y)
    assert isinstance(model.predict(X), numpy.ndarray)


@pytest.mark.skipif(not has_ivy, reason="needs Ivy")
def test_ivy_roundtrip_conversion():
    import ivy

    xp_tensor = numpy.zeros((2, 3), dtype="f")
    ivy_tensor = xp2ivy(xp_tensor)
    assert isinstance(ivy_tensor, ivy.Array)
    new_xp_tensor = ivy2xp(ivy_tensor)
    assert numpy.array_equal(xp_tensor, new_xp_tensor)


@pytest.mark.skipif(not has_ivy, reason="needs Ivy")
def test_ivy_wrapper_roundtrip():
    import ivy

    model = IvyWrapper(ivy.Linear(2, 3))
    model_bytes = model.to_bytes()
    IvyWrapper(ivy.Linear(2, 3)).from_bytes(model_bytes)
    with make_tempdir() as path:
        model_path = path / "model"
        model.to_disk(model_path)
        new_model = IvyWrapper(ivy.Linear(2, 3)).from_bytes(model_bytes)
        new_model.from_disk(model_path)


@pytest.mark.skipif(not has_ivy, reason="needs Ivy")
@pytest.mark.parametrize(
    "data,n_args,kwargs_keys",
    [
        # fmt: off
        # (numpy.zeros((2, 3), dtype="f"), 1, []),
        ([numpy.zeros((2, 3), dtype="f"), numpy.zeros((2, 3), dtype="f")], 2, []),
        ((numpy.zeros((2, 3), dtype="f"), numpy.zeros((2, 3), dtype="f")), 2, []),
        ({"a": numpy.zeros((2, 3), dtype="f"), "b": numpy.zeros((2, 3), dtype="f")}, 0, ["a", "b"]),
        (ArgsKwargs((numpy.zeros((2, 3), dtype="f"), numpy.zeros((2, 3), dtype="f")), {"c": numpy.zeros((2, 3), dtype="f")}), 2, ["c"]),
        # fmt: on
    ],
)
def test_ivy_convert_inputs(data, n_args, kwargs_keys):
    import ivy

    model = IvyWrapper(ivy.Linear(3, 4))
    convert_inputs = model.attrs["convert_inputs"]
    Y, backprop = convert_inputs(model, data, is_train=True)
    check_input_converters(Y, backprop, data, n_args, kwargs_keys, ivy.Array)


@pytest.mark.skipif(not has_ivy, reason="needs Ivy")
def test_ivy_wrapper_custom_serde():
    import ivy

    def serialize(model):
        return default_serialize_ivy_model(model)

    def deserialize(model, state_bytes, device):
        return default_deserialize_ivy_model(model, state_bytes, device)

    def get_model():
        return IvyWrapper(
            ivy.Linear(2, 3),
            serialize_model=serialize,
            deserialize_model=deserialize,
        )

    model = get_model()
    model_bytes = model.to_bytes()
    get_model().from_bytes(model_bytes)
    with make_tempdir() as path:
        model_path = path / "model"
        model.to_disk(model_path)
        new_model = get_model().from_bytes(model_bytes)
        new_model.from_disk(model_path)
