#ifndef _PROCESS_ON_OFF_FILTER_H
#define _PROCESS_ON_OFF_FILTER_H

#include <iostream>
#include "Process.h"
#include "NumpyReader.h"

namespace process
{

	namespace _priv
	{
		class OnOffFilterHelper
		{

		public:
			OnOffFilterHelper() = delete;

			static Tensor<float> generate_filter(size_t filter_size, float center_dev, float surround_dev);
		};
	}

	/**
	 * @brief On-center/off-center filtering in a pre-processing technique that is similar to DoG filtering to detect edges.
	 * In fact, it is to extract the difference in intensity which helps the SNN encode the image as spikes.
	 * This filter puts the on and off cells in 2 different channels.
	 *
	 * @param filter_size size_t - The size of the on-center/off-center filter.
	 * @param center_dev float - The variance of the Gaussian kernels DoG center.
	 * @param surround_dev float - The variance of the Gaussian kernels DoG surround.
	 * @param cutoff float - The minimum pixel value, used to reduce noise.
	 * @param expName float - The name of the experiment.
	 * @param draw size_t - A flag to draw the samples.
	 */
	class DefaultOnOffFilter : public UniquePassProcess
	{

	public:
		DefaultOnOffFilter();
		DefaultOnOffFilter(size_t filter_size, float center_dev = 1, float surround_dev = 4, float cutoff = 0, std::string expName = "", size_t draw = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(const std::string &label, Tensor<float> &in) const;

		size_t _filter_size;
		float _center_dev;
		float _surround_dev;
		float _cutoff;
		size_t _draw;
		std::string _expName;
		std::string _file_path;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		Tensor<float> _filter;
	};

	/**
	 * @brief This is an On-center/off-center filter specific for colored images, it is a pre-processing technique that is similar to DoG filtering to detect edges.
	 * In fact, it is to extract the difference in intensity which helps the SNN encode the image as spikes.
	 * This filter puts the on and off cells in 2 different channels.
	 *
	 * @param filename this takes a folder path that contains numpy arrays. arr_0.npy, arr_1.npy and arr_2.npy. where these arrays are the red, green abd blue values.
	 */
	class CustomRGBOnOffFilter : public UniquePassProcess
	{

	public:
		CustomRGBOnOffFilter();
		CustomRGBOnOffFilter(const std::string &filename);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &in) const;

		Tensor<float> _r;
		Tensor<float> _g;
		Tensor<float> _b;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

	/**
	 * @brief This filter seperates the information into 3 channels, (Black / White) & (Red / Green) & (Yellow / Blue) Similar to the ratina.
	 * On-center/off-center filtering in a pre-processing technique that is similar to DoG filtering to detect edges.
	 * In fact, it is to extract the difference in intensity which helps the SNN encode the image as spikes.
	 * This filter puts the on and off cells in 2 different channels.
	 *
	 * @param filter_size The size of the on-center/off-center filter.
	 * @param center_dev The variance of the Gaussian kernels DoG center.
	 * @param surround_dev The variance of the Gaussian kernels DoG surround.
	 */
	class BiologicalOnOffFilter : public UniquePassProcess
	{

	public:
		BiologicalOnOffFilter();
		BiologicalOnOffFilter(size_t filter_size, float center_dev, float surround_dev);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& in) const;

		size_t _filter_size;
		float _center_dev;
		float _surround_dev;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
		Tensor<float> _filter;
	};

}
#endif
