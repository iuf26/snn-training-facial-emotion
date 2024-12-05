#include "Experiment.h"
#include "dataset/Video.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecutionNew.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/Acceleration.h"
#include "process/SaveFeatures.h"
#include "process/SimplePreprocessing.h"
#include "process/OrientationAmplitude.h"
#include "process/MaxScaling.h"
#include "process/CompositeChannels.h"
#include "process/OnOffFilter.h"
#include "process/OnOffTempFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/Flatten.h"
#include "process/SeparateSign.h"
#include "tool/AutoFrameNumberSelector.h"
#include "process/SpikingBackgroundSubtraction.h"
#include "process/MotionGrid.h"
#include "process/MotionGridV1.h"
#include "process/MotionGridV2.h"
#include "process/MotionGridV3.h"
#include "process/MotionGridV4.h"
#include "process/ResidualConnection.h"
#include "process/SpikingMotionGrid.h"
#include "process/ReichardtDetector.h"

/**
 *  use this loop to find the ideal t_obj, for (float tobj = 0.10f; tobj <= 1.01f; tobj += 0.05f) float rounded_down = floorf(tobj * 100) / 100;
 */
int main(int argc, char **argv)
{
	// for (int _repeat = 1; _repeat < 4; _repeat++)
	{
		std::string _dataset = "KTH2_Out";

		Experiment<SparseIntermediateExecutionNew> experiment(argc, argv, _dataset, false, false);

		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		size_t _frame_size_width = 80, _frame_size_height = 60;
		// number of sets of frames per video, and how many sets to take from each video.
		size_t _video_frames = 10, _train_sample_per_video = 0, _test_sample_per_video = 0;
		// number of frames to skip, this speeds up the action.
		size_t _frame_gap = 2, _grey = 1, _draw = 0, _th_mv = 0;
		// filter sizes
		size_t filter_size = 5, tmp_filter_size = 1, temp_stride = 1, tmp_pooling_size = 1; // tmp_filter_size == 2 ? 2 : 1;
		size_t filter_number = 6;
		size_t sampling_size = 1000; //(_frame_size_height * _frame_size_width) / (filter_size * filter_size);

		experiment.push<process::MaxScaling>(); // 60, 80, 1
		experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0); //60, 80, 2

		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");

		std::string input_path(input_path_ptr);

		experiment.push<LatencyCoding>(); // 60, 80, 2

		// experiment.push<process::SimplePreprocessing>(experiment.name(), 2, _draw);
		// The location of the dataset Videos, seperated into train and test folders that contain labeled folders of videos.
		experiment.add_train<dataset::Video>(input_path + "/train", _video_frames, _frame_gap, _th_mv, _train_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);
		experiment.add_test<dataset::Video>(input_path + "/test", _video_frames, _frame_gap, _th_mv, _test_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);

		float t_obj = 0.75;
		float t_obj1 = 0.75;
		float t_obj2 = 0.75;

		float th_lr = 1.0f;
		float w_lr = 0.1f;

		// This function takes the following(Layer Name, Kernel width, kernel height, number of kernels, and a flag to draw the weights if 1 or not if 0)
		auto &conv1 = experiment.push<layer::Convolution3D>(7, 7, tmp_filter_size, 8, "", 1, 1, temp_stride);
		conv1.set_name("conv1"); // 56, 76, 8
		conv1.parameter<bool>("draw").set(false);
		conv1.parameter<bool>("save_weights").set(true);
		conv1.parameter<bool>("save_random_start").set(true);
		conv1.parameter<bool>("log_spiking_neuron").set(false);
		conv1.parameter<bool>("inhibition").set(true);
		conv1.parameter<uint32_t>("epoch").set(sampling_size);
		conv1.parameter<float>("annealing").set(0.95f);
		conv1.parameter<float>("min_th").set(1.0f);
		conv1.parameter<float>("t_obj").set(t_obj);
		conv1.parameter<float>("lr_th").set(th_lr);
		conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		// auto &pool1 = experiment.push<layer::Pooling3D>(2, 2, tmp_pooling_size, 2, 2);
		// pool1.set_name("pool1"); // 28, 38, 8

		auto &conv2 = experiment.push<layer::Convolution3D>(5, 5, tmp_filter_size, 9, "", 1, 1, temp_stride);
		conv2.set_name("conv2"); // 24, 34, 32
		conv2.parameter<bool>("draw").set(false);
		conv2.parameter<bool>("save_weights").set(true);
		conv2.parameter<bool>("save_random_start").set(true);
		conv2.parameter<bool>("log_spiking_neuron").set(false);
		conv2.parameter<bool>("inhibition").set(true);
		conv2.parameter<uint32_t>("epoch").set(sampling_size);
		conv2.parameter<float>("annealing").set(0.95f);
		conv2.parameter<float>("min_th").set(1.0f);
		conv2.parameter<float>("t_obj").set(t_obj1);
		conv2.parameter<float>("lr_th").set(th_lr);
		conv2.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv2.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
		conv2.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		// auto &pool2 = experiment.push<layer::Pooling3D>(2, 2, tmp_pooling_size, 2, 2);
		// pool2.set_name("pool2"); // 12, 17, 32

		auto &conv3 = experiment.push<layer::Convolution3D>(3, 3, tmp_filter_size, 10, "", 1, 1, temp_stride);
		conv3.set_name("conv3"); // 8, 13, 250
		conv3.parameter<bool>("draw").set(false);
		conv3.parameter<bool>("save_weights").set(true);
		conv3.parameter<bool>("save_random_start").set(true);
		conv3.parameter<bool>("log_spiking_neuron").set(false);
		conv3.parameter<bool>("inhibition").set(true);
		conv3.parameter<uint32_t>("epoch").set(sampling_size);
		conv3.parameter<float>("annealing").set(0.95f);
		conv3.parameter<float>("min_th").set(1.0f);
		conv3.parameter<float>("t_obj").set(t_obj1);
		conv3.parameter<float>("lr_th").set(th_lr);
		conv3.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv3.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(10.0, 0.1);
		conv3.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

		auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
		//conv1_out.add_postprocessing<process::SaveFeatures>(experiment.name(), conv1.name());
		conv1_out.add_postprocessing<process::SumPooling>(20, 20);
		conv1_out.add_postprocessing<process::FeatureScaling>();
		conv1_out.add_analysis<analysis::Activity>();
		conv1_out.add_analysis<analysis::Coherence>();
		conv1_out.add_analysis<analysis::Svm>();

		auto &conv2_out = experiment.output<TimeObjectiveOutput>(conv2, t_obj1);
		//conv2_out.add_postprocessing<process::SaveFeatures>(experiment.name(), conv2.name());
		conv2_out.add_postprocessing<process::SumPooling>(20, 20);
		// conv2_out.add_postprocessing<process::ResidualConnection>(experiment.name(), "");
		conv2_out.add_postprocessing<process::FeatureScaling>();
		conv2_out.add_analysis<analysis::Activity>();
		conv2_out.add_analysis<analysis::Coherence>();
		conv2_out.add_analysis<analysis::Svm>();

		auto &conv3_out = experiment.output<TimeObjectiveOutput>(conv3, t_obj1);
		conv3_out.add_postprocessing<process::SaveFeatures>(experiment.name(), conv3.name());
		conv3_out.add_postprocessing<process::SumPooling>(20, 20);
		conv3_out.add_postprocessing<process::FeatureScaling>();
		conv3_out.add_analysis<analysis::Activity>();
		conv3_out.add_analysis<analysis::Coherence>();
		conv3_out.add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}

// experiment.push<process::ResizeInput>(experiment.name(), _frame_size_width, _frame_size_height);
// experiment.push<process::ReichardtDetector>(experiment.name());
// experiment.push<process::SimplePreprocessing>(experiment.name(), 0, _draw);
// experiment.push<process::MotionGridV1>(experiment.name(), _draw, 320, 144);
// experiment.push<process::SpikingBackgroundSubtraction>(experiment.name(), 1);
// experiment.push<process::SpikingMotionGrid>(experiment.name());

// experiment.push<process::EarlyFusion>(experiment.name(), _draw, _video_frames);
// experiment_time.push<process::SetTemporalDepth>(experiment_time.name(), 8);
