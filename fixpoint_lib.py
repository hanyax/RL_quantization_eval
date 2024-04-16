from fxpmath import Fxp

FXP_CONFIGS = {
    "FXP32": {"signed": True, "n_int": 15, "n_frac": 16, "overflow": "saturate", "n_word": 32},
    "FXP64": {"signed": True, "n_int": 31, "n_frac": 32, "overflow": "saturate", "n_word": 64}
}

def to_fp(input_integer):
    temp = Fxp(0, **FXP_CONFIGS["FXP32"])
    input_fp = temp.set_val(input_integer, raw=True)
    return input_fp

def to_fix(input_fp):
    result = Fxp(input_fp, **FXP_CONFIGS["FXP32"])
    return result
    
def to_fix_val(input_fp):
    local_in = 0+input_fp
    result = Fxp(local_in, **FXP_CONFIGS["FXP32"])
    #print("Fixed number integer Binary Rep: ", result.bin(frac_dot=True))
    return result.val