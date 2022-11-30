#include "analysis/SaveOutputJson.h"

using namespace analysis;

static RegisterClassParameter<SaveOutputJson, AnalysisFactory> _register("SaveOutputJson");

SaveOutputJson::SaveOutputJson() : UniquePassAnalysis(_register),
	_train_filename(), _test_filename(), _json_train_file(), _json_test_file(), _first_train_sample(true), _first_test_sample(true) {
	throw std::runtime_error("Unimplemented");
}

SaveOutputJson::SaveOutputJson(const std::string& train_filename, const std::string& test_filename) : UniquePassAnalysis(_register),
	_train_filename(train_filename), _test_filename(test_filename), _json_train_file(), _json_test_file(), _first_train_sample(true), _first_test_sample(true) {

}

void SaveOutputJson::resize(const Shape&) {

}

void SaveOutputJson::before_train() {
	_json_train_file.open(_train_filename);
	_json_train_file << "[";
	_first_train_sample = true;
}

void SaveOutputJson::process_train(const std::string& label, const Tensor<float>& sample) {
	// Add comma before writting the next sample
	// Except when we process the first sample
	if (!_first_train_sample) {
		_json_train_file << ",";
	}
	else {
		_first_train_sample = false;
	}
	// Convert sample to JSON string
	std::string JSON_output = _to_json_string(label, sample);
	// Append to the file
	_json_train_file << JSON_output;
}

void SaveOutputJson::after_train() {
	_json_train_file << "]";
    _json_train_file.close();
}

void SaveOutputJson::before_test() {
	_json_test_file.open(_test_filename);
	_json_test_file << "[";
	_first_test_sample = true;
}

void SaveOutputJson::process_test(const std::string& label, const Tensor<float>& sample) {
	// Add comma before writting the next sample
	// Except when we process the first sample
	if (!_first_test_sample) {
		_json_test_file << ",";
	}
	else {
		_first_test_sample = false;
	}
	// Convert sample to JSON string
	std::string JSON_output = _to_json_string(label, sample);
	// Append to the file
	_json_test_file << JSON_output;
}

void SaveOutputJson::after_test() {
	_json_test_file << "]";
    _json_test_file.close();
}

std::string SaveOutputJson::_to_json_string(const std::string& label, const Tensor<float>& sample) {
    std::string JSON_output = "";

	// Create a JSON document that holds the memory of the sample to serialize
	// 4 bytes for the label + 4 bytes * (n_neurons * 4) // 4 for x,y,z,time variables
	DynamicJsonDocument doc(JSON_OBJECT_SIZE(4 + 4 * 4 * sample.shape().product()));

	// Create the main object
	JsonObject root = doc.to<JsonObject>();

	// Fill label info
	root["label"] = label;

	// Array containing spikes
	JsonArray spks_arr = root.createNestedArray("data");

	// Iterate over the sample to fill spike array
	size_t width = sample.shape().dim(0);
	size_t height = sample.shape().dim(1);
	size_t depth = sample.shape().dim(2);
	for(size_t x=0; x<width; x++) {
		for(size_t y=0; y<height; y++) {
			for(size_t z=0; z<depth; z++) {
				float t = sample.at(x, y, z);
				// Save spikes with t in [0,1[
				if (t != INFINITE_TIME && t >= 0.0f && t < 1.0f) {
					JsonObject spk_obj = spks_arr.createNestedObject();
					spk_obj["x"] = x;
					spk_obj["y"] = y;
					spk_obj["z"] = z;
					spk_obj["time"] = t;
				}
			}
		}
	}

	// Transform to string
	serializeJson(doc, JSON_output);

	return JSON_output;
}