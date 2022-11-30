# Falez CSNN Simulator 2019 with whitening

Forked to implement new experiments.

## Description

Simulator of Convolutional Spiking Neural Network

Provide implementation of experiments described in:
* __Unsupervised Visual Feature Learning with Spike-timing-dependent Plasticity: How Far are we from Traditional Feature Learning Approaches?__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, Pattern Recognition.
* __Multi-layered Spiking Neural Network with Target Timestamp Threshold Adaptation and STDP__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, IJCNN 2019.

## Requirement

* C++ compiler (version >= 14)
* Cmake (version >= 3.1)
* Qt4 (version >= 4.4.3)
* BLAS
* LAPACKE

## Installation

    mkdir csnn-simulator-build
    cd csnn-simulator-build
    cmake ../csnn-simulator -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO
    make

## Usage

Run MNIST Example:

    export INPUT_PATH=/path/to/mnist/
    ./Mnist
