#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import re


# In[ ]:


def param_groups_lrd(model, weight_decay=0.05, no_weight_decay_list=[], layer_decay=.85):
    """
    Parameter groups for layer-wise lr decay
    Following BEiT: https://github.com/microsoft/unilm/blob/master/beit/optim_factory.py#L58
    """
    param_group_names = {}
    param_groups = {}
        
    ids=0
    module_names = dict()
    for module_name, module in model.named_parameters():
        if module.requires_grad:
            module_name = re.sub(r".[0-9].(bias|weight)", "", module_name)
            if module_name not in module_names:
                module_names[module_name]=ids
                ids += 1
            
    # 创建10挡lr_scales
    layer_scales = list(layer_decay ** (9 - i) for i in range(10))
    bn = len(module_names)/10.0

    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        # no decay: all 1D parameters and model specific ones
        # ndim返回的是数组的维度,就是shape的len().shape是(2,4,7)ndim就是3
        # bias的ndim就是1
        # 维度为一或者指定的参数不进行权重衰减
        if p.ndim == 1 or n in no_weight_decay_list:
            g_decay = "no_decay"
            this_decay = 0.
        else:
            g_decay = "decay"
            this_decay = weight_decay
         
        
        layer_id = int(module_names[re.sub(r".[0-9].(bias|weight)", "", n)] // bn)
        
        # 用layer_id来区分lr_scale,用g_decay来区分是否有weight_decay
        group_name = "layer_%d_%s" % (layer_id, g_decay)
        
        # 如果是一组新的lr_scale，weight_decay，就创建一个新的字典保存param，lr_scale，weight_decay
        if group_name not in param_group_names:
            this_scale = layer_scales[layer_id]

            param_group_names[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "params": [],
            }
        # 否则就直接在已有的字典中的param里添加上就可以，不需要重新创建一个字典了。
        param_group_names[group_name]["params"].append(n)
        param_groups[group_name]["params"].append(p)

    # print("parameter groups: \n%s" % json.dumps(param_group_names, indent=2))
    # 返回的是param_groups.values(),不需要名字，实际上param_group_names就收集了个寂寞，无用。
    # 后期optimizer.param_groups()取出的数据，就是这个返回值
    return list(param_groups.values())




