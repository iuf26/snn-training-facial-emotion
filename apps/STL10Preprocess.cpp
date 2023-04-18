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
#include "analysis/SaveOutputNumpy.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/GrayScale.h"
#include "process/OnOffFilter.h"

int main(int argc, char** argv) {

	// Random seed
	int seed = 0;
	const char* seed_ptr = std::getenv("SEED");
	if(seed_ptr != nullptr) {
		seed = std::stoi(seed_ptr);
	}

	// Output path
	const char* output_path_ptr = std::getenv("OUTPUT_PATH");
	if(output_path_ptr == nullptr) {
		output_path_ptr = "./";
	}
	std::string output_path(output_path_ptr);

	// Load config
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
	// Add seed to config
	config["seed"] = seed;

	// Initialize experiment
	AbstractExperiment* experiment;
	experiment = new Experiment<DenseIntermediateExecution>(argc, argv, output_path, config["exp_name"], seed);

	// Load dataset
	const char* input_path_ptr = std::getenv("INPUT_PATH");
	if(input_path_ptr == nullptr) {
		throw std::runtime_error("Require to define INPUT_PATH variable");
	}
	std::string input_path(input_path_ptr);
	experiment->add_train<dataset::STL>(input_path+"train_X.bin", input_path+"train_y.bin");
	experiment->add_test<dataset::STL>(input_path+"test_X.bin", input_path+"test_y.bin");

	// Preprocessing
	experiment->push<process::DefaultOnOffFilter>(7, config["dog_stds"][0], config["dog_stds"][1]);
	if (config["feature_scaling"]) {
		experiment->push<process::FeatureScaling>();
	}
	experiment->push<LatencyCoding>();

	// Pooling layer
	auto& pool1 = experiment->push<layer::Pooling>(1, 1, 1, 1);
	pool1.set_name("pool1");

	// Save features
	auto& pool1_save = experiment->output<SpikeTiming>(pool1);
	pool1_save.add_analysis<analysis::SaveOutputNumpy>(output_path);

	experiment->run(10000);
	delete experiment;
}
