#include "Experiment.h"
#include "dataset/Mnist.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "stdp/Proportional.h"
#include "layer/Convolution.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "layer/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "stdp/Linear.h"
#include "stdp/BiologicalMultiplicative.h"
#include "analysis/SaveOutputNumpy.h"

int main(int argc, char** argv) {
	Experiment<DenseIntermediateExecution> experiment(argc, argv, "mnist");

	experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
	experiment.push<process::FeatureScaling>();
	experiment.push<LatencyCoding>();

	const char* input_path_ptr = std::getenv("INPUT_PATH");

	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.add_train<dataset::Mnist>(input_path+"train-images.idx3-ubyte", input_path+"train-labels.idx1-ubyte");
	experiment.add_test<dataset::Mnist>(input_path+"t10k-images.idx3-ubyte", input_path+"t10k-labels.idx1-ubyte");

	float w_lr = 0.1f;
	float th_lr = 1.0f;
	float t_obj = 0.75f;

	auto& conv1 = experiment.push<layer::Convolution>(5, 5, 64);
	conv1.set_name("conv1");
	conv1.parameter<uint32_t>("epoch").set(100);
	conv1.parameter<float>("annealing").set(0.95f);
	conv1.parameter<float>("min_th").set(1.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<bool>("wta_infer").set(false);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
	conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

	auto& pool1 = experiment.push<layer::Pooling>(2, 2, 2, 2);
	pool1.set_name("pool1");

	auto& conv2 = experiment.push<layer::Convolution>(5, 5, 128);
	conv2.set_name("conv2");
	conv2.parameter<uint32_t>("epoch").set(100);
	conv2.parameter<float>("annealing").set(0.95f);
	conv2.parameter<float>("min_th").set(1.0f);
	conv2.parameter<float>("t_obj").set(t_obj);
	conv2.parameter<float>("lr_th").set(th_lr);
	conv2.parameter<bool>("wta_infer").set(false);
	conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
	conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

	auto& pool2 = experiment.push<layer::Pooling>(2, 2, 2, 2);
	pool2.set_name("pool2");

	
	///////////////////////////////
	/////////// OUTPUTS ///////////
	///////////////////////////////

	// pool1 : Save features
	//auto& pool1_save = experiment.output<SpikeTiming>(pool1);
	//pool1_save.add_analysis<analysis::SaveOutputNumpy>("pool1");

	// pool1 : Activity
	auto& pool1_activity = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();

	// pool1 : SVM evaluation
	auto& pool1_out = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_out.add_postprocessing<process::SumPooling>(4, 4);
	pool1_out.add_postprocessing<process::FeatureScaling>();
	pool1_out.add_analysis<analysis::Svm>();

	// pool2 : Save features
	//auto& pool2_save = experiment.output<SpikeTiming>(pool2);
	//pool2_save.add_analysis<analysis::SaveOutputNumpy>("pool2");

	// pool2 : Activity
	auto& pool2_activity = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_activity.add_analysis<analysis::Activity>();

	// pool2 : SVM evaluation
	auto& pool2_out = experiment.output<DefaultOutput>(pool2, 0.0, 1.0);
	pool2_out.add_postprocessing<process::SumPooling>(4, 4);
	pool2_out.add_postprocessing<process::FeatureScaling>();
	pool2_out.add_analysis<analysis::Svm>();
	
	
	experiment.run(10000);
}
