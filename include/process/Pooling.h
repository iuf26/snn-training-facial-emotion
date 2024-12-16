#ifndef _PROCESS_POOLING_H
#define _PROCESS_POOLING_H

#include "Process.h"
#include "tool/Operations.h"

namespace process
{
	static int _train_sample_count = 0;
	static int _test_sample_count = 0;

	/**
	 * @brief A type of pooling that reduces the size of the input sample by averaging the values of the each set of pixels in the pooling filter. In both the spatial and temporal dimensions.
	 * @param target_width The desired output width after pooling
	 * @param target_height The desired output height after pooling
	 * @param target_conv_depth The desired output depth after pooling
	 */
	class SpatioTemporalSumPooling : public UniquePassProcess
	{

	public:
		SpatioTemporalSumPooling();
		SpatioTemporalSumPooling(size_t target_width, size_t target_height, size_t target_conv_depth = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &sample) const;

		size_t _target_width;
		size_t _target_height;
		size_t _target_conv_depth;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

	/**
	 * @brief A type of pooling that reduces the size of the input sample by averaging the values of the each set of pixels in the pooling filter. In the spatial dimension only.
	 * @param target_width The desired output width after pooling
	 * @param target_height The desired output height after pooling
	 * @param target_conv_depth The desired output depth after pooling
	 */
	class SumPooling : public UniquePassProcess
	{

	public:
		SumPooling();
		SumPooling(size_t target_width, size_t target_height);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& sample) const;

		size_t _target_width;
		size_t _target_height;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

	/**
	 * @brief A type of pooling that reduces the size of the input sample by selecting the maximum values of the each set of pixels in the pooling filter scope.
	 * @param target_width The desired output width after pooling
	 * @param target_height The desired output height after pooling
	 * @param target_depth The desired output depth after pooling
	 */
	class MaxPooling : public UniquePassProcess
	{

	public:
		MaxPooling();
		// MaxPooling(size_t target_width, size_t target_height);
		MaxPooling(size_t target_width, size_t target_height, size_t target_conv_depth = 0);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &sample) const;

		size_t _target_width;
		size_t _target_height;
		size_t _target_conv_depth;

		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

	/**
	 * @brief A type of pooling that only concerns the temporal dimention, the other dimùensions are not pooled.
	 * @param target_conv_depth The desired output depth after pooling
	 */
	class TemporalPooling : public UniquePassProcess
	{

	public:
		TemporalPooling();
		TemporalPooling(size_t target_conv_depth);

		virtual Shape compute_shape(const Shape &shape);
		virtual void process_train(const std::string &label, Tensor<float> &sample);
		virtual void process_test(const std::string &label, Tensor<float> &sample);

	private:
		void _process(Tensor<float> &sample) const;

		size_t _target_conv_depth;
		size_t _width;
		size_t _height;
		size_t _depth;
		size_t _conv_depth;
	};

	class MeanPooling : public UniquePassProcess {

	public:
		MeanPooling();
		MeanPooling(size_t target_width, size_t target_height);

		virtual Shape compute_shape(const Shape& shape);
		virtual void process_train(const std::string& label, Tensor<float>& sample);
		virtual void process_test(const std::string& label, Tensor<float>& sample);

	private:
		void _process(Tensor<float>& sample) const;

		size_t _target_width;
		size_t _target_height;

		size_t _width;
		size_t _height;
		size_t _depth;
	};

}

#endif
