import numpy as np


def initialize_weight(layer):
    classname = layer.__class__.__name__
    if 'Conv' in classname:
        weight_shape = list(layer.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        layer.weight.data.uniform_(-w_bound, w_bound)
        if layer.bias is not None:
            layer.bias.data.fill_(0)
    elif 'Linear' in classname:
        weight_shape = list(layer.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        layer.weight.data.uniform_(-w_bound, w_bound)
        if layer.bias is not None:
            layer.bias.data.fill_(0)


def initialize_model_weight(layers):
    for layer in layers:
        if list(layer.children()) == []:
            initialize_weight(layer)
        else:
            for sub_layer in list(layer.children()):
                initialize_weight(sub_layer)
