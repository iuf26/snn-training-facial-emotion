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

	// Load CSNN config
	const char* config_path_ptr = std::getenv("CONFIG_PATH");
	if(config_path_ptr == nullptr) {
		throw std::runtime_error("Require to define CONFIG_PATH variable");
	}
	std::string config_path(config_path_ptr);
	std::ifstream _jsonTextFile(config_path);
	if (!_jsonTextFile.good()) {
		throw std::runtime_error("Failed to open JSON config");
	}
	std::stringstream buffer;
	buffer << _jsonTextFile.rdbuf();
	std::string _jsonText = buffer.str();
	_jsonTextFile.close();
	DynamicJsonDocument config(JSON_ARRAY_SIZE(_jsonText.length()));
	DeserializationError error = deserializeJson(config, _jsonText.c_str());
	if (error) {
		throw std::runtime_error("Failed to parse JSON config");
	}

	// Initialize experiment
	Experiment<DenseIntermediateExecution> experiment(argc, argv, config["output_path"], config["app_name"], config["seed"]);

	// Load dataset
	const char* input_path_ptr = std::getenv("INPUT_PATH");
	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}
	std::string input_path(input_path_ptr);
	experiment.add_train<dataset::STL>(input_path+"train_X.bin", input_path+"train_y.bin");
	experiment.add_test<dataset::STL>(input_path+"test_X.bin", input_path+"test_y.bin");

	// Preprocessing
	experiment.push<process::DefaultOnOffFilter>(config["dog_k"], config["dog_std1"], config["dog_std2"]);
	experiment.push<process::FeatureScaling>();
	experiment.push<LatencyCoding>();

	// Convolutional layer
	auto& conv1 = experiment.push<layer::Convolution>(config["conv1_k_w"], config["conv1_k_h"], config["conv1_c"]);
	conv1.set_name("conv1");
	conv1.parameter<uint32_t>("epoch").set(config["conv1_epochs"]);
	conv1.parameter<float>("annealing").set(config["conv1_annealing"]);
	conv1.parameter<float>("min_th").set(config["conv1_min_th"]);
	conv1.parameter<float>("t_obj").set(config["conv1_t_obj"]);
	conv1.parameter<float>("lr_th").set(config["conv1_lr_th"]);
	conv1.parameter<bool>("wta_infer").set(config["conv1_wta_infer"]);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Gaussian>(config["conv1_w_init_mean"], config["conv1_w_init_std"]);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Constant>(config["conv1_th"].as<float>());
	std::string stdp_type(config["conv1_stdp"].as<const char*>());
	if (stdp_type == "multiplicative") {
		conv1.parameter<STDP>("stdp").set<stdp::Multiplicative>(config["conv1_stdp_lr"], config["conv1_stdp_b"]);
	}
	else if (stdp_type == "biological") {
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(config["conv1_stdp_lr"], config["conv1_stdp_t"]);
	}
	else {
		throw std::runtime_error("STDP type " + stdp_type + " is not implemented");
	}

	// Pooling layer
	auto& pool1 = experiment.push<layer::Pooling>(config["pool1_k_w"], config["pool1_k_h"], config["pool1_s_w"], config["pool1_s_h"]);
	pool1.set_name("pool1");

	// Activity analysis
	auto& conv1_activity = experiment.output<DefaultOutput>(conv1, 0.0, 1.0);
	conv1_activity.add_analysis<analysis::Coherence>();
	auto& pool1_activity = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
	pool1_activity.add_analysis<analysis::Activity>();

	// Save features
	auto& pool1_save = experiment.output<SpikeTiming>(pool1);
	pool1_save.add_analysis<analysis::SaveOutputNumpy>(config["output_path"].as<const char*>());

	// SVM evaluation
	if (config["svm_eval"]) {
		auto& pool1_out = experiment.output<DefaultOutput>(pool1, 0.0, 1.0);
		pool1_out.add_analysis<analysis::Svm>();
	}

	experiment.run(10000);
}
