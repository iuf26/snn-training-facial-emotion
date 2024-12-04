#include "Experiment.h"
#include "dataset/STL.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/Whiten.h"
#include "process/WhitenPatches.h"
#include "process/SeparateSign.h"

int main(int argc, char** argv) {
	Experiment<SparseIntermediateExecution> experiment(argc, argv, "gen-cifar10");

	const char* input_path_ptr = std::getenv("INPUT_PATH");

	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	auto& whitening = experiment.push<process::WhiteningPatches>(9, 0.01, 1.0, 2, 1000000);

	experiment.add_train<dataset::STL>(input_path+"train_X.bin", input_path+"train_y.bin");

	experiment.run(10000);

	whitening.save("whiten-stl10");
}
