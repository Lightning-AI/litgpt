from collections import defaultdict


def find_forward_hooks(module):
    mapping = defaultdict(list)
    for name, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            hook_data = ("forward_pre_hook", hook.func.__name__, hook.args, hook.keywords)
            mapping[name].append(hook_data)
        for hook in submodule._forward_hooks.values():
            hook_data = ("forward_hook", hook.func.__name__, hook.args, hook.keywords)
            mapping[name].append(hook_data)
    return dict(mapping)
