from enum import Enum


class Proxy(Enum):
    EXPRESSIVITY_AZ = "expressivity_az"
    PROGRESSIVITY_AZ = "progressivity_az"
    TRAINABILITY_AZ = "trainability"    # TODO: should rename to trainability_az
    ZEN = "zen"
    GRADNORM = "gradnorm"
    NASWOT = "naswot"
    SYNFLOW = "synflow"
    SNIP = "snip"
    GRASP = "grasp"
    GRADSIGN_REV = "gradsign_rev"   # NOTE: changed from gradsign to gradsign_rev because first is incorrectly implemented?
    GRADSIGN = "gradsign"
    NTK_TENAS = "ntk_tenas"
    LR_TENAS = "lr_tenas"
    ZICO = "zico"
    FLOPS = "FLOPs"
    ACCURACY = "accuracy"

# Ensembles

EnsembleProxies = {"zero": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.TRAINABILITY_AZ, Proxy.FLOPS],
                    "one": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.ZICO, Proxy.FLOPS],
                    "two": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.TRAINABILITY_AZ, Proxy.ZICO, Proxy.FLOPS],
                    "three": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.FLOPS],
                    "four": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.TRAINABILITY_AZ, Proxy.FLOPS],
                    # "five": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.FLOPS, Proxy.GRADSIGN], gradsign doesn't work
                    "six": [Proxy.EXPRESSIVITY_AZ, Proxy.PROGRESSIVITY_AZ, Proxy.SYNFLOW, Proxy.FLOPS, Proxy.NTK_TENAS]
                    }









