from typing import Any, Callable, Dict, Optional, Tuple, cast, Type, TypeVar

from ..compat import ivy
from ..config import registry
from ..model import Model
from ..shims import IvyShim
from ..types import ArgsKwargs, Floats3d, Padded, ArrayXd
from ..util import (
    convert_recursive,
    ivy2xp,
    xp2ivy,
    partial,
    is_xp_array,
    is_ivy_array,
)

InT = TypeVar("InT")
OutT = TypeVar("OutT")
InFunc = TypeVar("InFunc")
XType = TypeVar("XType", bound=ArrayXd)
YType = TypeVar("YType", bound=ArrayXd)


@registry.layers("IvyWrapper.v1")
def IvyWrapper(
    ivy_model: Any,
    loss_fn: Optional[Callable] = None,
    convert_inputs: Optional[Callable] = None,
    convert_outputs: Optional[Callable] = None,
    serialize_model: Optional[Callable[[Any], bytes]] = None,
    deserialize_model: Optional[Callable[[Any, bytes, "str"], Any]] = None,
) -> Model[Any, Any]:
    if convert_inputs is None:
        convert_inputs = convert_ivy_default_inputs
    if convert_outputs is None:
        convert_outputs = convert_ivy_default_outputs
    return Model(
        "ivy",
        forward,
        attrs={"convert_inputs": convert_inputs, "convert_outputs": convert_outputs},
        shims=[
            IvyShim(
                ivy_model,
                loss_fn=loss_fn,
                serialize_model=serialize_model,
                deserialize_model=deserialize_model,
            )
        ],
        dims={"nI": None, "nO": None},
    )


def forward(model: Model, X: Any, is_train: bool) -> Tuple[Any, Callable]:
    convert_inputs = model.attrs["convert_inputs"]
    convert_outputs = model.attrs["convert_outputs"]

    X_ivy, get_dX = convert_inputs(model, X, is_train)
    if is_train:
        Y_ivy, ivy_backprop = model.shims[0](X_ivy, is_train)
    else:
        Y_ivy = model.shims[0](X_ivy, is_train)
    Y_pred, get_dY_ivy = convert_outputs(model, Y_ivy, is_train)

    def backprop(dY: Any) -> Any:
        dYivy = get_dY_ivy(dY)
        dX_ivy = ivy_backprop(dYivy)
        dX = get_dX(dX_ivy)
        return dX

    return Y_pred, backprop


def convert_ivy_default_inputs(
    model: Model, X: Any, is_train: bool
) -> Tuple[ArgsKwargs, Callable[[ArgsKwargs], Any]]:
    shim = cast(IvyShim, model.shims[0])
    xp2ivy_ = lambda x: xp2ivy(x, device=shim.device)
    converted = convert_recursive(is_xp_array, xp2ivy_, X)

    if isinstance(converted, ArgsKwargs):

        def reverse_conversion(dXtorch):
            return convert_recursive(is_ivy_array, ivy2xp, dXtorch)

        return converted, reverse_conversion

    elif isinstance(converted, dict):

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_ivy_array, ivy2xp, dXtorch)
            return dX.kwargs

        return ArgsKwargs(args=tuple(), kwargs=converted), reverse_conversion

    elif isinstance(converted, (tuple, list)):

        def reverse_conversion(dXtorch):
            dX = convert_recursive(is_ivy_array, ivy2xp, dXtorch)
            return dX.args

        return ArgsKwargs(args=converted, kwargs={}), reverse_conversion

    else:

        def reverse_conversion(dXtorch):
            return convert_recursive(is_ivy_array, ivy2xp, dXtorch)

        return ArgsKwargs(args=(converted,), kwargs={}), reverse_conversion


def convert_ivy_default_outputs(model: Model, Yivy: Any, is_train: bool):
    shim = cast(IvyShim, model.shims[0])

    Y = convert_recursive(is_ivy_array, ivy2xp, Yivy)

    def reverse_conversion(dY: Any) -> ArgsKwargs:
        dYivy = convert_recursive(is_xp_array, partial(xp2ivy, device=shim.device), dY)
        return ArgsKwargs(args=((Yivy,),), kwargs={"grad_tensors": dYivy})

    return Y, reverse_conversion
