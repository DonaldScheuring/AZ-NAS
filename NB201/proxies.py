from enum import Enum


class Proxy(Enum):
    EXPRESSIVITY_AZ = "expressivity_az"
    PROGRESSIVITY_AZ = "progressivity_az"
    TRAINABILITY_AZ = "trainability_az"
    ZEN = "zen"
    GRADNORM = "grad_norm"
    NASWOT = "naswot"
    SYNFLOW = "synflow"
    SNIP = "snip"
    GRASP = "grasp"
    GRADSIGN_REV = "gradsign_rev"   # NOTE: changed from gradsign to gradsign_rev because first is incorrectly implemented?
    GRADSIGN = "gradsign"   # NOTE: This one also is kinda bad
    NTK_TENAS = "ntk_tenas"
    LR_TENAS = "lr_tenas"
    ZICO = "zico"
    FLOPS = "FLOPs"
    PARAMS = "params"

# Ensembles

EnsembleProxies = {
                    #"tenas" : [Proxy.NTK_TENAS, Proxy.LR_TENAS],
                    "aznas": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.TRAINABILITY_AZ, Proxy.FLOPS],
                    "one": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.ZICO, Proxy.FLOPS],
                    "two": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.TRAINABILITY_AZ, Proxy.ZICO, Proxy.FLOPS],
                    "three": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.FLOPS],
                    "four": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.TRAINABILITY_AZ, Proxy.FLOPS],
                    "five": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.TRAINABILITY_AZ, Proxy.FLOPS, Proxy.ZICO],
                    "six": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.FLOPS, Proxy.NTK_TENAS], 
                    "seven": [
                        Proxy.EXPRESSIVITY_AZ,
                        Proxy.PROGRESSIVITY_AZ,
                        Proxy.TRAINABILITY_AZ,
                        Proxy.ZEN,
                        Proxy.GRADNORM,
                        Proxy.NASWOT,
                        Proxy.SYNFLOW,
                        Proxy.SNIP,
                        Proxy.GRASP,
                        Proxy.NTK_TENAS,
                        Proxy.LR_TENAS,
                        Proxy.ZICO,
                        Proxy.FLOPS,
                        Proxy.PARAMS
                        ]
                    }









