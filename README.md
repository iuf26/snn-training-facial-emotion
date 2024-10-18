# CSNN-simulator

This is a new version of the CSNN simulator that contains 2D and 3D convolution, along with two-stream methods for video analysis.

In order to run a 2D experiment, use the Convolution class in the layer, or Convolution3D while setting the temporal depth to 1.

In order to run a 3D experiment, use the Convolution3D class and set the temporal depth > 1.

In order to run a two-stream experiment, chech the TwoStream.cpp class where two experiments are created, after that, the results of these experiments are fused and evaluated using the SVM.

The SVM.cpp class can be used to test the classification rate of the SVM alone without an SNN. This is useful to make sure that the SNN is indeed adding a benefit.

The FeatureEvaluation.cpp class can be used to re-evaluate extracted and saved features by the SVM alone without re-training and re-running an SNN.

For execution policies, SparseIntermediateExecutionNew can be used for saving certain things like output features adn output timestamps (features but as spikes). If there is no need to save anything use SparseIntermediateExecution instead, it's faster. DenseIntermediateExecution is even faster.


Simulator of Convolutional Spiking Neural Network

Provide implementation of experiments described in:
* __Unsupervised Visual Feature Learning with Spike-timing-dependent Plasticity: How Far are we from Traditional Feature Learning Approaches?__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, Pattern Recognition.
* __Multi-layered Spiking Neural Network with Target Timestamp Threshold Adaptation and STDP__, P Falez, P Tirilly, IM Bilasco, P Devienne, P Boulet, IJCNN 2019.

* C++ compiler (version >= 17)
* Cmake (version >= 3.1)
* Qt4 (version >= 4.4.3)
* BLAS
* LAPACKE
* OpenCV (version >= 4.2.0)

## Installation

### Dependencies
```
sudo apt update
sudo apt install --yes gcc g++ make cmake libatlas-base-dev libblas-dev libopenblas-dev liblapack-dev liblapacke-dev libopencv-dev python3-opencv
sudo add-apt-repository ppa:rock-core/qt4 && sudo apt install qt4-default
```
### Compile
```
mkdir csnn-simulator-build
cd csnn-simulator-build
cmake ../csnn-simulator -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DUSE_GUI=NO
make
```

## Usage
Run MNIST Example:
```
export INPUT_PATH=/path/to/mnist/
./Mnist
