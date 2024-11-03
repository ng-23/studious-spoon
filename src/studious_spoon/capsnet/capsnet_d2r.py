'''
    Implementation of the Capsule Network with
    Dynamic Deep Routing (D2R), as described by
    Peer and his team in their 2018 research paper.

    Architecture is largely the same as the original CapsNet architecture, with
    the addition of a variable number of HiddenCapsules (HiddenCaps) layers between the
    PrimaryCapsules (PrimaryCaps) and DigitCaps layers. 
    Each HiddenCaps layer added has 4 less capsules than previous HiddenCaps layers.

    See https://www.researchgate.net/publication/329945773_Training_Deep_Capsule_Networks for research paper.
'''