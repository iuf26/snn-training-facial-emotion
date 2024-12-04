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
#include "process/SimplePreprocessing.h"
#include "process/OrientationAmplitude.h"
#include "process/SeparateSign.h"
#include "process/MotionGrid.h"
#include "process/SaveFeatures.h"
#include "process/MaxScaling.h"
#include "process/ResizeInput.h"
#include "process/SkeletonDetection.h"

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
	for (int _repeat = 0; _repeat <= 6; _repeat++)
	{
		// 320 × 240
		std::string _dataset = "KTH-TS-F5F15";
		// The name of the experiment_space is tha name of the dataset, this name is used for the log text file. // flag that permits saving the exp output tensors or not.
		Experiment<SparseIntermediateExecutionNew> experiment_space(argc, argv, _dataset, false, true);
		Experiment<SparseIntermediateExecutionNew> experiment_time(argc, argv, _dataset + "_time", false, true);
		Experiment<FusedExecution> experiment_fused(argc, argv, _dataset + "_fused", false, false);

		// The new dimentions of a video frame, set to zero if default dimentions are needed.
		// size_t _frame_size_width = 90, _frame_size_height = 50;
		size_t _frame_size_width = 80, _frame_size_height = 60;
		size_t _video_frames = 10, _train_sample_per_video = 0, _test_sample_per_video = 0, _train_sample_per_video_2 = 0, _test_sample_per_video_2 = 0;
		size_t _temporal_sum_pooling = 2, _sum_pooling = 20; // the dimensions of the output features going into the SVM

		// number of frames to skip, this speeds up the action.
		size_t _th_mv = 0, _frame_gap_train = 3, _frame_gap_test = 3;
		// The fusion method, and if the vieo is in greyscale, and if I want tha dataset drawn or not
		size_t _method = 1, _grey = 1, _draw = 0;
		// filter sizes
		size_t filter_size_L1 = 5, filter_size_L2 = 15, tmp_filter_size = 1, tmp_pooling_size = 1, temp_stride = 1; // the step of the kernel in the temporal dimension
		size_t spatial_stride = 1;
		size_t filter_number = 64;
		size_t sampling_size = 800; //(_frame_size_height * _frame_size_width) / (filter_size * filter_size);

		// skeleton parameters
		float _threshold = 0.08, _point_radius = 2, _edge_width = 2;
		size_t _onFrame = 0, _csv = 0;

		const char *input_path_ptr = std::getenv("INPUT_PATH");

		if (input_path_ptr == nullptr)
			throw std::runtime_error("Require to define INPUT_PATH variable");

		std::string input_path(input_path_ptr);

		std::string HMDBdata_split = "Split2/";
		// The location of the dataset images, seperated into train and test folders that contain labeled folders of images.
		experiment_space.add_train<dataset::Video>(input_path + "/train/", _video_frames, _frame_gap_train, _th_mv, _train_sample_per_video, _grey, experiment_space.name(), _draw, _frame_size_width, _frame_size_height);
		experiment_space.add_test<dataset::Video>(input_path + "/test/", _video_frames, _frame_gap_test, _th_mv, _test_sample_per_video, _grey, experiment_space.name(), _draw, _frame_size_width, _frame_size_height);

		experiment_time.add_train<dataset::Video>(input_path + "/train/", _video_frames, _frame_gap_train, _th_mv, _train_sample_per_video_2, _grey, experiment_time.name(), _draw, _frame_size_width, _frame_size_height);
		experiment_time.add_test<dataset::Video>(input_path + "/test/", _video_frames, _frame_gap_test, _th_mv, _test_sample_per_video_2, _grey, experiment_time.name(), _draw, _frame_size_width, _frame_size_height);

		// experiment_time.push<process::SimplePreprocessing>(experiment_time.name(), 1);
		// experiment_space.push<process::OrientationAmplitude>(experiment_time.name());

		// experiment_time.push<process::ResizeInput>(experiment_time.name(), _frame_size_width, _frame_size_height);
		// experiment_space.push<process::ResizeInput>(experiment_space.name(), _frame_size_width, _frame_size_height);

		// experiment_time.push<process::SkeletonDetection>(experiment_time.name(), _threshold, _point_radius, _edge_width, 0, _onFrame, _csv);

		// experiment_time.push<process::EarlyFusion>(experiment_time.name(), 0, _video_frames, 1);
		// experiment_time.push<process::MotionGrid>(experiment_time.name(), _draw, 320, 144);

		experiment_space.push<process::MaxScaling>();
		experiment_time.push<process::MaxScaling>();
		
		experiment_space.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
		experiment_time.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);

		// Temporal coding used, the data value "float" is transformed and reprisented by the spike timestamp.
		experiment_space.push<LatencyCoding>();
		experiment_time.push<LatencyCoding>();

		float t_obj1 = 0.65;

		float th_lr = 1.0f;
		float w_lr = 0.1f;

		//------------- SPACE -------------

		// auto &pool1_s = experiment_space.push<layer::Pooling3D>(2, 2, 2, 2, 2, 2);
		// pool1_s.set_name("pool1_s");

		auto &conv1_s = experiment_space.push<layer::Convolution3D>(filter_size_L1, filter_size_L1, 1, filter_number);
		conv1_s.set_name("conv1_s");
		conv1_s.parameter<bool>("draw").set(false);
		conv1_s.parameter<bool>("save_weights").set(true);
		conv1_s.parameter<bool>("save_random_start").set(false);
		conv1_s.parameter<bool>("log_spiking_neuron").set(false);
		conv1_s.parameter<bool>("inhibition").set(true);
		conv1_s.parameter<uint32_t>("epoch").set(sampling_size);
		conv1_s.parameter<float>("annealing").set(0.95f);
		conv1_s.parameter<float>("min_th").set(1.0f);
		conv1_s.parameter<float>("t_obj").set(t_obj1);
		conv1_s.parameter<float>("lr_th").set(th_lr);
		conv1_s.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1_s.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv1_s.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 1.0);

		// TimeObjectiveOutput converts timestamps back into float values.
		auto &conv1_s_out = experiment_space.output<TimeObjectiveOutput>(conv1_s, t_obj1);
		conv1_s_out.add_postprocessing<process::SumPooling>(_sum_pooling, _sum_pooling);
		conv1_s_out.add_postprocessing<process::TemporalPooling>(_temporal_sum_pooling);
		conv1_s_out.add_postprocessing<process::FeatureScaling>();
		conv1_s_out.add_analysis<analysis::Activity>();
		conv1_s_out.add_analysis<analysis::Coherence>();
		conv1_s_out.add_analysis<analysis::Svm>();

		experiment_space.run(10000);

		//------------- TIME -------------

		// auto &pool1_t = experiment_time.push<layer::Pooling3D>(2, 2, 2, 2, 2, 2);
		// pool1_t.set_name("pool1_t");

		auto &conv1_t = experiment_time.push<layer::Convolution3D>(filter_size_L2, filter_size_L2, tmp_filter_size, filter_number, "", 1, 1, temp_stride);
		conv1_t.set_name("conv1_t");
		conv1_t.parameter<bool>("draw").set(false);
		conv1_t.parameter<bool>("save_weights").set(true);
		conv1_t.parameter<bool>("save_random_start").set(false);
		conv1_t.parameter<bool>("log_spiking_neuron").set(false);
		conv1_t.parameter<bool>("inhibition").set(true);
		conv1_t.parameter<uint32_t>("epoch").set(sampling_size);
		conv1_t.parameter<float>("annealing").set(0.95f);
		conv1_t.parameter<float>("min_th").set(1.0f);
		conv1_t.parameter<float>("t_obj").set(t_obj1);
		conv1_t.parameter<float>("lr_th").set(th_lr);
		conv1_t.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
		conv1_t.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
		conv1_t.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 1.0);

		auto &conv1_t_out = experiment_time.output<TimeObjectiveOutput>(conv1_t, t_obj1);
		conv1_t_out.add_postprocessing<process::SumPooling>(_sum_pooling, _sum_pooling);
		conv1_t_out.add_postprocessing<process::TemporalPooling>(_temporal_sum_pooling);
		conv1_t_out.add_postprocessing<process::FeatureScaling>();
		conv1_t_out.add_analysis<analysis::Activity>();
		conv1_t_out.add_analysis<analysis::Coherence>();
		conv1_t_out.add_analysis<analysis::Svm>();

		experiment_time.run(10000);

		//------------- FUSION -------------
		// add code to re-fill the saved values from the text files and fuse them here.
		std::string _file_path = std::filesystem::current_path();

		experiment_fused.add_train<dataset::TwoStream>(_file_path + "/ExtractedFeatures/" + experiment_space.name() + "/train/", _method, 0);
		experiment_fused.add_test<dataset::TwoStream>(_file_path + "/ExtractedFeatures/" + experiment_space.name() + "/test/", _method, 0);

		auto &svm = experiment_fused.push<layer::Stream>(1, 1, 1, filter_number);

		// add another SVM step to classify the fused result.
		auto &fused_out = experiment_fused.output<NoOutputConversion>(svm);
		// fused_out.add_postprocessing<process::SaveFeatures>(experiment_fused.name(), "fused_out");
		fused_out.add_postprocessing<process::SumPooling>(_sum_pooling, _sum_pooling);
		fused_out.add_analysis<analysis::Svm>();

		experiment_fused.run(10000);
	}
}