#include "Experiment.h"
#include "dataset/STL.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "analysis/Svm.h"
#include "analysis/SaveOutputNumpy.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/GrayScale.h"
#include "process/OnOffFilter.h"

int main(int argc, char** argv) {

	Experiment<DenseIntermediateExecution> experiment(argc, argv, "stl_10gs");

	const char* input_path_ptr = std::getenv("INPUT_PATH");

	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}

	std::string input_path(input_path_ptr);

	experiment.push<process::GrayScale>();
	experiment.push<process::DefaultOnOffFilter>(7, 1.0, 2.0);
	experiment.push<process::FeatureScaling>();
	experiment.push<LatencyCoding>();

	experiment.add_train<dataset::STL>(input_path+"train_X.bin", input_path+"train_y.bin");
	experiment.add_test<dataset::STL>(input_path+"test_X.bin", input_path+"test_y.bin");

	float t_obj = 0.85f;
	float th_lr = 1.0f;
	float w_lr = 0.1f;

	auto& conv1 = experiment.push<layer::Convolution>("conv1", 5, 5, 64);
	conv1.parameter<float>("annealing").set(0.95f);
	conv1.parameter<float>("min_th").set(4.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(5.0, 0.1);
	conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(w_lr, 0.1);

	experiment.add_train_step(conv1, 100);

	auto& pool1 = experiment.push<layer::Pooling>(4, 4, 4, 4);
	pool1.set_name("pool1");

	///////////////////////////////
	/////////// OUTPUTS ///////////
	///////////////////////////////

	// pool1 : Save features
	auto& pool1_save = experiment.output<SpikeTiming>(pool1);
	pool1_save.add_analysis<analysis::SaveOutputNumpy>("pool1_4x4");

	// pool1 : Activity
	auto& pool1_activity = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();

	// pool1 : SVM evaluation
	auto& pool1_out = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_out.add_postprocessing<process::SumPooling>(2, 2);
	pool1_out.add_postprocessing<process::FeatureScaling>();
	pool1_out.add_analysis<analysis::Svm>();

	auto& pool1_out2 = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_out2.add_analysis<analysis::Svm>();

	experiment.run(10000);
}