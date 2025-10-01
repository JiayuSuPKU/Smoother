import warnings

import torch
from scvi import REGISTRY_KEYS, settings
from scvi.nn import FCLayers

# SCVI.module.base._utils.py
def iterate(obj, func):
    """Iterates over an object and applies a function to each element."""
    t = type(obj)
    if t is list or t is tuple:
        return t([iterate(o, func) for o in obj])
    else:
        return func(obj) if obj is not None else None

# SCVI.module.base._utils.py
def broadcast_labels(o, n_broadcast=-1):
    """Utility for the semi-supervised setting.

    If y is defined(labelled batch) then one-hot encode the labels (no broadcasting needed)
    If y is undefined (unlabelled batch) then generate all possible labels (and broadcast other
    arguments if not None)
    """
    ys_ = torch.nn.functional.one_hot(
        torch.arange(n_broadcast, device=o.device, dtype=torch.long), n_broadcast
    )
    ys = ys_.repeat_interleave(o.size(-2), dim=0)
    if o.ndim == 2:
        new_o = o.repeat(n_broadcast, 1)
    elif o.ndim == 3:
        new_o = o.repeat(1, n_broadcast, 1)
    return ys, new_o

# SCVI.module.base._utils.py
def get_max_epochs_heuristic(
    n_obs: int, epochs_cap: int = 400, decay_at_n_obs: int = 20000
) -> int:
    """Compute a heuristic for the default number of maximum epochs.

    If `n_obs <= decay_at_n_obs`, the number of maximum epochs is set to
    `epochs_cap`. Otherwise, the number of maximum epochs decays according to
    `(decay_at_n_obs / n_obs) * epochs_cap`, with a minimum of 1.

    Parameters
    ----------
    n_obs
        The number of observations in the dataset.
    epochs_cap
        The maximum number of epochs for the heuristic.
    decay_at_n_obs
        The number of observations at which the heuristic starts decaying.

    Returns
    -------
    `int`
        A heuristic for the default number of maximum epochs.
    """
    max_epochs = min(round((decay_at_n_obs / n_obs) * epochs_cap), epochs_cap)
    max_epochs = max(max_epochs, 1)

    if max_epochs == 1:
        warnings.warn(
            "The default number of maximum epochs has been set to 1 due to the large"
            "number of observations. Pass in `max_epochs` to the `train` function in "
            "order to override this behavior.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )

    return max_epochs

# SCVI.model.base._archesmixins.py
def set_params_online_update(
    module,
    unfrozen,
    freeze_decoder_first_layer,
    freeze_batchnorm_encoder,
    freeze_batchnorm_decoder,
    freeze_dropout,
    freeze_expression,
    freeze_classifier,
):
    """Freeze parts of network for scArches."""
    # do nothing if unfrozen
    if unfrozen:
        return

    mod_inference_mode = {"encoder_z2_z1", "decoder_z1_z2"}
    mod_no_hooks_yes_grad = {"l_encoder"}
    if not freeze_classifier:
        mod_no_hooks_yes_grad.add("classifier")
    parameters_yes_grad = {"background_pro_alpha", "background_pro_log_beta"}

    def no_hook_cond(key):
        one = (not freeze_expression) and "encoder" in key
        two = (not freeze_decoder_first_layer) and "px_decoder" in key
        return one or two

    def requires_grad(key):
        mod_name = key.split(".")[0]
        # linear weights and bias that need grad
        one = "fc_layers" in key and ".0." in key and mod_name not in mod_inference_mode
        # modules that need grad
        two = mod_name in mod_no_hooks_yes_grad
        three = sum([p in key for p in parameters_yes_grad]) > 0
        # batch norm option
        four = (
            "fc_layers" in key
            and ".1." in key
            and "encoder" in key
            and (not freeze_batchnorm_encoder)
        )
        five = (
            "fc_layers" in key
            and ".1." in key
            and "decoder" in key
            and (not freeze_batchnorm_decoder)
        )
        if one or two or three or four or five:
            return True
        else:
            return False

    for key, mod in module.named_modules():
        # skip over protected modules
        if key.split(".")[0] in mod_no_hooks_yes_grad:
            continue
        if isinstance(mod, FCLayers):
            hook_first_layer = False if no_hook_cond(key) else True
            mod.set_online_update_hooks(hook_first_layer)
        if isinstance(mod, torch.nn.Dropout):
            if freeze_dropout:
                mod.p = 0
        # momentum freezes the running stats of batchnorm
        freeze_batchnorm = ("decoder" in key and freeze_batchnorm_decoder) or (
            "encoder" in key and freeze_batchnorm_encoder
        )
        if isinstance(mod, torch.nn.BatchNorm1d) and freeze_batchnorm:
            mod.momentum = 0

    for key, par in module.named_parameters():
        if requires_grad(key):
            par.requires_grad = True
        else:
            par.requires_grad = False
