#include <fstream>
#include "Experiment.h"
#include "dataset/Video.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "process/ResidualConnection.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecution.h"
#include "execution/SparseIntermediateExecutionNew.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/Acceleration.h"
#include "process/ResizeInput.h"
#include "process/SimplePreprocessing.h"
#include "process/CompositeChannels.h"
#include "process/CompositeChannels2.h"
#include "process/DXDY.h"
#include "process/OrientationAmplitude.h"
#include "process/MaxScaling.h"
#include "process/SaveFeatures.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/Flatten.h"
#include "process/SeparateSign.h"
#include "tool/AutoFrameNumberSelector.h"
#include "process/SpikingBackgroundSubtraction.h"
#include "process/MotionGrid.h"
#include "process/SpikingMotionGrid.h"
#include "process/Amplification.h"
#include "process/AddSaltPepperNoise.h"

/**
 *  use this loop to find the ideal t_obj, for (float tobj = 0.10f; tobj <= 1.01f; tobj += 0.05f) float rounded_down = floorf(tobj * 100) / 100;
 */
int main(int argc, char **argv)
{
	for (int _repeat = 1; _repeat < 4; _repeat++)
	{
		size_t _filter_size = 5;
		std::string _dataset = "KTH-j22-e-" + std::to_string(_filter_size) + "-L1-3D";
		std::string _file_path = std::filesystem::current_path(); // path to the build folder where the params are saved
		std::string _file_name = _file_path + "/Param_config/" + _dataset;

		Experiment<SparseIntermediateExecutionNew> experiment(argc, argv, _dataset, false, true, false);
		// Experiment<SparseIntermediateExecution> experiment(argc, argv, _dataset); //, false, true, false, false);

		// --------------setting variables--------------
		std::ifstream _jsonTextFile(_file_name);
		std::string _jsonText;
		std::getline(_jsonTextFile, _jsonText);
		_jsonTextFile.close();
		DynamicJsonDocument doc(JSON_ARRAY_SIZE(_jsonText.length()));
		DeserializationError error = deserializeJson(doc, _jsonText.c_str());
		// Test if parsing succeeds.
		if (error)
			ASSERT_DEBUG("JSON PARSE FAILED");

		const char *input_path_ptr = std::getenv("INPUT_PATH");
		if (input_path_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");
		std::string input_path_var(input_path_ptr);

		_jsonText = "";
		std::string input_path = _jsonText == "" || doc["input_path"] == "" ? input_path_var : doc["input_path"];

		size_t _frame_size_width = _jsonText == "" ? 80 : doc["_frame_width"],
			   _frame_size_height = _jsonText == "" ? 60 : doc["_frame_height"],
			   _video_frames = _jsonText == "" ? 10 : doc["_video_frames"],
			   _train_sample_per_video = _jsonText == "" ? 0 : doc["_train_sample_per_video"], _test_sample_per_video = _jsonText == "" ? 0 : doc["_test_sample_per_video"],
			   _th_mv = _jsonText == "" ? 0 : doc["_th_mv"], _frame_gap = _jsonText == "" ? 3 : doc["_frame_gap"], _grey = _jsonText == "" ? 1 : doc["_grey"], _draw = _jsonText == "" ? 0 : doc["_draw"],
			   filter_number = _jsonText == "" ? 64 : doc["filter_number"],
			   filter_size = _jsonText == "" ? _filter_size : doc["filter_size"],
			   tmp_filter_size = _jsonText == "" ? _filter_size : doc["tmp_filter_size"],
			   temp_stride = _jsonText == "" ? 2 : doc["temp_stride"], sampling_size = _jsonText == "" ? 1200 : doc["sampling_size"];
		size_t _sumPooling = 20;

		float t_obj = _jsonText == "" ? 0.65 : doc["t_obj"];
		float th_lr = _jsonText == "" ? 0.09f : doc["th_lr"];
		float w_lr = _jsonText == "" ? 0.009f : doc["w_lr"];
		// --------------end of setting variables--------------

		experiment.add_train<dataset::Video>(input_path + "/train/", _video_frames, _frame_gap, _th_mv, _train_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);
		experiment.add_test<dataset::Video>(input_path + "/test/", _video_frames, _frame_gap, _th_mv, _test_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);

		//experiment.push<process::SimplePreprocessing>(experiment.name(), 1, 1);
		experiment.push<process::DefaultOnOffFilter>(7, 0.1, 4.0);
		experiment.push<process::MaxScaling>();
		experiment.push<LatencyCoding>();

		// This function takes the following(Layer Name, Kernel width, kernel height, number of kernels, and a flag to draw the weights if 1 or not if 0)
		auto &conv1 = experiment.push<layer::Convolution3D>(filter_size, filter_size, filter_size, filter_number, "", 1, 1, temp_stride);
		conv1.set_name("conv1");
		conv1.parameter<bool>("draw").set(false);
		conv1.parameter<bool>("save_weights").set(true);
		conv1.parameter<bool>("save_random_start").set(false);
		conv1.parameter<bool>("log_spiking_neuron").set(true);
		conv1.parameter<bool>("inhibition").set(true);
		conv1.parameter<uint32_t>("epoch").set(sampling_size);
		conv1.parameter<float>("annealing").set(0.95f);
		conv1.parameter<float>("min_th").set(1.0f);
		conv1.parameter<float>("t_obj").set(t_obj);
		conv1.parameter<float>("lr_th").set(th_lr);
		conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
		conv1_out.add_postprocessing<process::SumPooling>(_sumPooling, _sumPooling);
		conv1_out.add_postprocessing<process::TemporalPooling>(2);
		conv1_out.add_postprocessing<process::FeatureScaling>();
		conv1_out.add_analysis<analysis::Activity>();
		conv1_out.add_analysis<analysis::Coherence>();
		conv1_out.add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}

// conv2_out.add_postprocessing<process::LateFusion>(experiment.name(), 1, _video_frames, 0);

// experiment.push<process::CompositeChannels2>(experiment.name(), 1, 50);
// experiment.push<process::CompositeChannels>(experiment.name(), 1);
//  experiment.push<process::SimplePreprocessing>(experiment.name(), 1, _draw);
// experiment.push<process::DXDY>(experiment.name(), 0);
// experiment.push<process::OrientationAmplitude>(experiment.name(), 0, 1);
// experiment.push<process::EarlyFusion>(experiment.name(), 0, _video_frames, 1);