import logging

from trulens.nn.attribution import InputAttribution, IntegratedGradients
from trulens.nn.models import get_model_wrapper
from trulens.utils.tru_logger import get_logger

get_logger().setLevel(logging.ERROR)  # disable trulens logging


def compute_saliency_map(net, x):
    return InputAttribution(
        get_model_wrapper(
            net,
            input_shape=x.shape[1:],
        )
    ).attributions(x)


def compute_integrated_gradients(net, x, baseline):
    return IntegratedGradients(
        get_model_wrapper(
            net,
            input_shape=x.shape[1:],
        ),
        baseline,
        resolution=30,
    ).attributions(x)
