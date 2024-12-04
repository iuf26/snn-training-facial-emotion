#include "Experiment.h"
#include "dataset/TwoStream.h"
#include "dataset/Video.h"
#include "dataset/Frame.h"
#include "dataset/Image.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/Convolution.h"
#include "layer/Stream.h"
#include "layer/Pooling.h"
#include "Distribution.h"
#include "execution/DenseIntermediateExecution.h"
#include "execution/SparseIntermediateExecutionNew.h"
#include "execution/FusedExecution.h"
#include "analysis/Svm.h"
#include "analysis/Activity.h"
#include "analysis/Coherence.h"
#include "process/Input.h"
#include "process/Scaling.h"
#include "process/Pooling.h"
#include "process/OnOffFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/MaxScaling.h"
#include "process/SetTemporalDepth.h"
#include "process/SimplePreprocessing.h"
#include "process/SeparateSign.h"
#include "process/MotionGrid.h"
#include "process/MotionGridV1.h"
#include "process/MotionGridV2.h"
#include "process/MotionGridV3.h"
#include "process/MotionGridV4.h"
#include "process/MotionGridV5.h"
#include "process/SpikeMotionGrid.h"
#include "process/ResizeInput.h"

/**
 * @brief This function runs Image expirements with a 2D convolutional SNN. This class contains a sequential run of pre-prosessing,
 * neural coding, convolutions, pooling and visualization.
 *
 * @param DefaultOnOffFilter The on-center off-center filter. This specific filter is used with grey-scale information.
 * @param FeatureScaling This normalizes the input pixels into values between zero and one.
 * @param LatencyCoding The type of neural coding chosen for this expirement transforms the normalized pixel values into timestamps.
 * @param w_lr The weights learning rate.
 * @param th_lr The threshould learning rate.
 * @param t_obj The objective time that neuron firing should converge to. This is for threshould adaptation.
 */

int main(int argc, char **argv)
{
	// for (int _repeat = 0; _repeat < 3; _repeat++)
	// {

	// std::string subjects[9] = {"daria", "denis", "eli", "ido", "ira", "lena", "lyova", "moshe", "shahar"}; // for (int _repeat = 0; _repeat < 5; _repeat++){
	// size_t filters[4] = {3, 5, 7, 9};

	// for (std::string subject : subjects)
	for (int _repeat = 0; _repeat < 3; _repeat++)
	{
		std::string _dataset = "MG_V1";

		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		size_t _frame_size_width = 80, _frame_size_height = 60;

		// The name of the experiment is tha name of the dataset, this name is used for the log text file. // flag that permits saving the exp output tensors or not
		Experiment<SparseIntermediateExecutionNew> experiment(argc, argv, _dataset, false, true);

		// number of sets of frames per video.
		size_t _video_frames = 33, _train_sample_per_video = 0, _test_sample_per_video = 0;
		// number of frames to skip, this speeds up the action.
		size_t _th_mv = 0, _frame_gap = 4;
		size_t _grey = 1, _draw = 0;
		// filter sizes
		size_t filter_size = 5, tmp_filter_size = 1, tmp_pooling_size = 1; // tmp_filter_size <= 2 ? 2 : 1;

		// experiment.push<process::ResizeInput>(experiment.name(), _frame_size_width, _frame_size_height);
		//  experiment.push<process::SimplePreprocessing>(experiment.name(), 2, _draw);
		//  experiment.push<process::MotionGrid>(experiment.name(), _draw, 150, 150);
		experiment.push<process::MotionGridV1>(experiment.name(), _draw, 320, 144);
		experiment.push<process::MaxScaling>();
		// experiment.push<process::SetTemporalDepth>(experiment.name(), 5);

		experiment.push<process::DefaultOnOffFilter>(24, 0.5, 5.0);
		// experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);

		size_t sampling_size = 800; //(_frame_size_height * _frame_size_width) / (filter_size * filter_size);

		// sampling_size = sampling_size > 400 ? sampling_size : 400;

		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");

		std::string input_path(input_path_ptr);

		// Temporal coding used, the data value "float" is transformed and reprisented by the spike timestamp.
		experiment.push<LatencyCoding>();
		// "/" + subject
		experiment.add_train<dataset::Video>(input_path + "/train/", _video_frames, _frame_gap, _th_mv, _train_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);
		experiment.add_test<dataset::Video>(input_path + "/test/", _video_frames, _frame_gap, _th_mv, _test_sample_per_video, _grey, experiment.name(), _draw, _frame_size_width, _frame_size_height);

		float t_obj = 0.65;
		float t_obj1 = 0.65;
		float t_obj2 = 0.5;

		float th_lr = 1.0f;
		float w_lr = 0.1f;

		//------------- TIME -------------
		auto &pool1 = experiment.push<layer::Pooling3D>(2, 2, tmp_pooling_size, 2, 2);
		pool1.set_name("pool1");

		auto &conv1_t = experiment.push<layer::Convolution3D>(filter_size, filter_size, tmp_filter_size, 64);
		conv1_t.set_name("conv1_t");
		conv1_t.parameter<bool>("draw").set(false);
		conv1_t.parameter<bool>("save_weights").set(true);
		conv1_t.parameter<bool>("inhibition").set(true);
		conv1_t.parameter<uint32_t>("epoch").set(sampling_size);
		conv1_t.parameter<float>("annealing").set(0.95f);
		conv1_t.parameter<float>("min_th").set(1.0f);
		conv1_t.parameter<float>("t_obj").set(t_obj);
		conv1_t.parameter<float>("lr_th").set(th_lr);
		conv1_t.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1_t.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv1_t.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 1.0);

		auto &conv1_t_out = experiment.output<TimeObjectiveOutput>(conv1_t, t_obj);
		conv1_t_out.add_postprocessing<process::SumPooling>(30, 30);
		conv1_t_out.add_postprocessing<process::FeatureScaling>();
		conv1_t_out.add_analysis<analysis::Activity>();
		conv1_t_out.add_analysis<analysis::Coherence>();
		conv1_t_out.add_analysis<analysis::Svm>();

		experiment.run(10000);
	}
}