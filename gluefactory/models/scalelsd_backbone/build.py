from .dpt.models import DPTFieldModel

def build_dpt(
    basemodel = "vitb_rn50_384",
    features=256,
    readout = "project",
    channels_last = False,
    use_bn = True,
    enable_attention_hooks = False,
    head_size = [[3],[1],[1],[2],[2]],
    use_layer_scale = False,
    **kwargs):
    
    model = DPTFieldModel(
        features=features,
        backbone=basemodel,
        readout=readout,
        channels_last=channels_last,
        use_bn=use_bn,
        enable_attention_hooks=enable_attention_hooks,
        head_size=head_size,
        use_layer_scale=use_layer_scale
    )

    return model
    
def build_backbone(**kwargs):
    return build_dpt(**kwargs)
