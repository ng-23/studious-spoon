'''
    Implementation of the Capsule Network with
    Expectation-Maximization (EM) Routing, as
    described by Hinton and his team in their 2018 research paper.

    Composed of a convolutional layer followed by a PrimaryCapsules (PrimaryCaps) layer,
    then 2 ConvolutionalCapsules (ConvCaps) layers and finally a ClassCaps layer for classification.

    See https://openreview.net/pdf?id=HJWLfGWRb for research paper.
'''