#include "Experiment.h"
#include "dataset/Image.h"
#include "stdp/Multiplicative.h"
#include "stdp/Biological.h"
#include "layer/Convolution3D.h"
#include "layer/Convolution.h"
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
#include "process/OnOffFilter.h"
#include "process/EarlyFusion.h"
#include "process/LateFusion.h"
#include "process/SimplePreprocessing.h"
#include "process/SeparateSign.h"
#include "process/ResizeInput.h"
#include "process/MaxScaling.h"
#include "process/MotionGridV1.h"
#include "process/ImageGrid.h"

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
	std::string _dataset = "Video_Frames";
	// The name of the experiment is tha name of the dataset, this name is used for the log text file.
	Experiment<SparseIntermediateExecutionNew> experiment(argc, argv, _dataset, false, false);

	// The new dimentions of a video frame, set to zero if default dimentions are needed.
	size_t _frame_size_width = 80, _frame_size_height = 60;
	// number of sets of frames per video.
	size_t _frames = 10, _train_sample_per_video = 5, _test_sample_per_video = 0;
	// number of frames to skip, this speeds up the action.
	size_t _frame_gap = 2;
	size_t _grey = 0, _draw = 1;
	// filter sizes
	size_t filter_size = 5, tmp_filter_size = 1, tmp_pooling_size = tmp_filter_size == 2 ? 2 : 1;
	size_t sampling_size = 800; //(_frame_size_height * _frame_size_width) / (filter_size * filter_size);

	experiment.push<process::ImageGrid>(experiment.name(), _draw, 0, 0);
	// experiment.push<process::ResizeInput>(experiment.name(), _frame_size_width, _frame_size_height);

	experiment.push<process::MaxScaling>();
	experiment.push<process::DefaultOnOffFilter>(7, 1.0, 4.0);
	const char *input_path_ptr = std::getenv("INPUT_PATH");

	if (input_path_ptr == nullptr)
		throw std::runtime_error("Require to define INPUT_PATH variable");

	// Get the input path of the dataset.
	std::string input_path(input_path_ptr);

	// Temporal coding used, the data value "float" is transformed and reprisented by the spike timestamp.
	experiment.push<LatencyCoding>();

	// The location of the dataset images, seperated into train and test folders that contain labeled folders of images.
	experiment.add_train<dataset::Image>(input_path + "/train", _frames, _frame_size_width, _frame_size_height, _grey);
	experiment.add_test<dataset::Image>(input_path + "/test", _frames, _frame_size_width, _frame_size_height, _grey);

	float t_obj = 0.75;

	float th_lr = 1.0f;
	float w_lr = 0.1f;

	// First convolution layer, name, width, hight, depth of filter(NUMBER OF FEATURES nf - filter number). stride x and y are fixed to 1 which means there is a big overlapping while extracting features.
	auto &conv1 = experiment.push<layer::Convolution3D>(filter_size, filter_size, tmp_filter_size, 16);
	conv1.set_name("conv1");
	conv1.parameter<bool>("draw").set(false);
	conv1.parameter<bool>("save_weights").set(true);
	conv1.parameter<bool>("inhibition").set(true);
	conv1.parameter<uint32_t>("epoch").set(100);
	conv1.parameter<float>("annealing").set(0.95f);
	conv1.parameter<float>("min_th").set(1.0f);
	conv1.parameter<float>("t_obj").set(t_obj);
	conv1.parameter<float>("lr_th").set(th_lr);
	// conv1.parameter<bool>("wta_infer").set(false);
	conv1.parameter<Tensor<float>>("w").distribution<distribution::Uniform>(0.0, 1.0);
	conv1.parameter<Tensor<float>>("th").distribution<distribution::Gaussian>(8.0, 0.1);
	conv1.parameter<STDP>("stdp").set<stdp::Biological>(w_lr, 0.1f);

#ifdef ENABLE_QT
	conv1.plot_threshold(true);
	conv1.plot_reconstruction(true);
#endif

	auto &conv1_out = experiment.output<TimeObjectiveOutput>(conv1, t_obj);
	conv1_out.add_postprocessing<process::SumPooling>(2, 2);
	conv1_out.add_postprocessing<process::FeatureScaling>();
	conv1_out.add_analysis<analysis::Activity>();
	conv1_out.add_analysis<analysis::Coherence>();
	conv1_out.add_analysis<analysis::Svm>();

	experiment.run(10000);
}