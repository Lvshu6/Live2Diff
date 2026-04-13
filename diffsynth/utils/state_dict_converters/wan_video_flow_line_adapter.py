def WanFlowLineAdapterStateDictConverter(state_dict):
    state_dict_ = {}
    for name in state_dict:
        if name.startswith("flow_line_blocks") or name.startswith("flow_line_patch_embedding"):
            state_dict_[name] = state_dict[name]
    return state_dict_