#include "process/Pooling.h"

using namespace process;

static RegisterClassParameter<SpatioTemporalSumPooling, ProcessFactory> _register("SpatioTemporalSumPooling");

SpatioTemporalSumPooling::SpatioTemporalSumPooling() : UniquePassProcess(_register),
													   _target_width(0), _target_height(0), _target_conv_depth(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("width", _target_width);
	add_parameter("height", _target_height);
	add_parameter("conv_depth", _target_conv_depth);
}

SpatioTemporalSumPooling::SpatioTemporalSumPooling(size_t target_width, size_t target_height, size_t target_conv_depth) : SpatioTemporalSumPooling()
{
	parameter<size_t>("width").set(target_width);
	parameter<size_t>("height").set(target_height);
	parameter<size_t>("conv_depth").set(target_conv_depth);
}

Shape SpatioTemporalSumPooling::compute_shape(const Shape &shape)
{
	_train_sample_count = 0;
	_test_sample_count = 0;
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;
	// In case the user didn't want temporal pooling, the _target_conv_depth would be the same as the _conv_depth
	_target_conv_depth = _target_conv_depth == 0 ? _conv_depth : _target_conv_depth;

	return Shape({std::min<size_t>(_target_width, _width),
				  std::min<size_t>(_target_height, _height),
				  _depth, std::min<size_t>(_target_conv_depth, _conv_depth)});
}

void SpatioTemporalSumPooling::process_train(const std::string &, Tensor<float> &sample)
{
	_train_sample_count++;
	_process(sample);
	// draw_progress(_train_sample_count, get_train_count());
}

void SpatioTemporalSumPooling::process_test(const std::string &, Tensor<float> &sample)
{
	_test_sample_count++;
	_process(sample);
	// draw_progress(_test_sample_count, get_test_count());
}

void SpatioTemporalSumPooling::_process(Tensor<float> &in) const
{

	// draw result before pooling
	// Tensor<float>::draw_tensor("/home/melassal/Bureau/test/" + std::to_string(_test_sample_count), in);

	size_t output_width = std::min<size_t>(_target_width, _width);
	size_t output_height = std::min<size_t>(_target_height, _height);
	size_t output_conv_depth = std::min<size_t>(_target_conv_depth, _conv_depth);
	// In the case of joined input samples, make sure we are using the correct _input_conv_depth, not the one before joining.
	size_t _input_conv_depth = in.shape().number() > 3 ? in.shape().dim(3) : 1;

	size_t filter_width = _width / output_width;
	size_t filter_height = _height / output_height;
	size_t filter_conv_depth = _input_conv_depth / output_conv_depth;

	// Tensor<float> out(Shape({output_width, output_height, _depth, output_conv_depth}));
	Tensor<float> out = Tensor<float>(Shape({output_width, output_height, _depth, output_conv_depth}));

	for (size_t x = 0; x < output_width; x++)
		for (size_t y = 0; y < output_height; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < output_conv_depth; k++)
				{
					float v = 0;

					for (size_t fx = 0; fx < filter_width; fx++)
						for (size_t fy = 0; fy < filter_height; fy++)
							for (size_t fk = 0; fk < filter_conv_depth; fk++)
							{
								if (in.shape().number() > 3)
									v += in.at(x * filter_width + fx, y * filter_height + fy, z, k * filter_conv_depth + fk);
								else
									v += in.at(x * filter_width + fx, y * filter_height + fy, z);
							}

					out.at(x, y, z, k) = v;
				}
	in = out;
}

static RegisterClassParameter<SumPooling, ProcessFactory> _registerSum("SumPooling");

SumPooling::SumPooling() : UniquePassProcess(_registerSum),
						   _target_width(0), _target_height(0), _width(0), _height(0), _depth(0)
{
	add_parameter("width", _target_width);
	add_parameter("height", _target_height);
}

SumPooling::SumPooling(size_t target_width, size_t target_height) : SumPooling() {
	parameter<size_t>("width").set(target_width);
	parameter<size_t>("height").set(target_height);
}

Shape SumPooling::compute_shape(const Shape &shape)
{
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;

	return Shape({std::min<size_t>(_target_width, _width),
				  std::min<size_t>(_target_height, _height),
				  _depth, _conv_depth});
}

void SumPooling::process_train(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void SumPooling::process_test(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void SumPooling::_process(Tensor<float> &in) const
{

	size_t output_width = std::min<size_t>(_target_width, _width);
	size_t output_height = std::min<size_t>(_target_height, _height);

	size_t filter_width = _width / output_width;
	size_t filter_height = _height / output_height;

	Tensor<float> out(Shape({output_width, output_height, _depth, _conv_depth}));

	for (size_t x = 0; x < output_width; x++)
	{
		for (size_t y = 0; y < output_height; y++)
		{
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < _conv_depth; k++)
				{
					float v = 0;

					for (size_t fx = 0; fx < filter_width; fx++)
					{
						for (size_t fy = 0; fy < filter_height; fy++)
						{
							v += in.shape().number() > 3 ? in.at(x * filter_width + fx, y * filter_height + fy, z, k) : in.at(x * filter_width + fx, y * filter_height + fy, z);
						}
					}

					out.at(x, y, z, k) = v;
				}
		}
	}

	in = out;
}


static RegisterClassParameter<TemporalPooling, ProcessFactory> _tmp_register("TemporalPooling");

TemporalPooling::TemporalPooling() : UniquePassProcess(_tmp_register),
									 _target_conv_depth(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
}

TemporalPooling::TemporalPooling(size_t target_conv_depth) : TemporalPooling()
{
	_target_conv_depth = target_conv_depth;
}

Shape TemporalPooling::compute_shape(const Shape &shape)
{
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;
	// In case the user didn't want temporal pooling, the _target_conv_depth would be the same as the _conv_depth
	_target_conv_depth = _target_conv_depth == 0 ? _conv_depth : _target_conv_depth;

	return Shape({_width, _height,
				  _depth, std::min<size_t>(_target_conv_depth, _conv_depth)});
}

void TemporalPooling::process_train(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void TemporalPooling::process_test(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void TemporalPooling::_process(Tensor<float> &in) const
{

	size_t output_conv_depth = std::min<size_t>(_target_conv_depth, _conv_depth);
	// In the case of joined input samples, make sure we are using the correct _input_conv_depth, not the one before joining.
	size_t _input_conv_depth = in.shape().number() > 3 ? in.shape().dim(3) : 1;

	size_t filter_conv_depth = _input_conv_depth / output_conv_depth;

	Tensor<float> out(Shape({_width, _height, _depth, output_conv_depth}));

	for (size_t x = 0; x < _width; x++)
		for (size_t y = 0; y < _height; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < output_conv_depth; k++)
				{
					float v = 0;
					for (size_t fk = 0; fk < filter_conv_depth; fk++)
					{
						v += in.at(x, y, z, k * filter_conv_depth + fk);
					}
					out.at(x, y, z, k) = v;
				}
	in = out;
}


static RegisterClassParameter<MeanPooling, ProcessFactory> _register_1("MeanPooling");


MeanPooling::MeanPooling() : UniquePassProcess(_register_1),
	_target_width(0), _target_height(0), _width(0), _height(0), _depth(0) {
	add_parameter("width", _target_width);
	add_parameter("height", _target_height);
}

MeanPooling::MeanPooling(size_t target_width, size_t target_height) : MeanPooling() {
	parameter<size_t>("width").set(target_width);
	parameter<size_t>("height").set(target_height);
}

Shape MeanPooling::compute_shape(const Shape& shape) {
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	return Shape({std::min<size_t>(_target_width, _width),
				  std::min<size_t>(_target_height, _height),
				  _depth});
}

void MeanPooling::process_train(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void MeanPooling::process_test(const std::string&, Tensor<float>& sample) {
	_process(sample);
}

void MeanPooling::_process(Tensor<float>& in) const {

	size_t output_width = std::min<size_t>(_target_width, _width);
	size_t output_height = std::min<size_t>(_target_height, _height);

	size_t filter_width = _width/output_width;
	size_t filter_height = _height/output_height;

	Tensor<float> out(Shape({output_width, output_height, _depth}));

	for(size_t x=0; x<output_width; x++) {
		for(size_t y=0; y<output_height; y++) {
			for(size_t z=0; z<_depth; z++) {
				float v = 0;

				for(size_t fx=0; fx<filter_width; fx++) {
					for(size_t fy=0; fy<filter_height; fy++) {
						v += in.at(x*filter_width+fx, y*filter_height+fy, z);
					}
				}

				out.at(x, y, z) = v / (filter_width*filter_height);
			}
		}
	}

	in = out;
}

static RegisterClassParameter<MaxPooling, ProcessFactory> _registerMax("MaxPooling");

MaxPooling::MaxPooling() : UniquePassProcess(_registerMax),
						   _target_width(0), _target_height(0), _target_conv_depth(0), _width(0), _height(0), _depth(0), _conv_depth(0)
{
	add_parameter("width", _target_width);
	add_parameter("height", _target_height);
	add_parameter("conv_depth", _target_conv_depth);
}

MaxPooling::MaxPooling(size_t target_width, size_t target_height, size_t target_conv_depth) : MaxPooling()
{
	parameter<size_t>("width").set(target_width);
	parameter<size_t>("height").set(target_height);
	parameter<size_t>("conv_depth").set(target_conv_depth);
}

Shape MaxPooling::compute_shape(const Shape &shape)
{
	_width = shape.dim(0);
	_height = shape.dim(1);
	_depth = shape.dim(2);
	_conv_depth = shape.number() > 3 ? shape.dim(3) : 1;
	// In case the user didn't want temporal pooling, the _target_conv_depth would be the same as the _conv_depth
	_target_conv_depth = _target_conv_depth == 0 ? _conv_depth : _target_conv_depth;

	return Shape({std::min<size_t>(_target_width, _width),
				  std::min<size_t>(_target_height, _height),
				  _depth, std::min<size_t>(_target_conv_depth, _conv_depth)});
}

void MaxPooling::process_train(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void MaxPooling::process_test(const std::string &, Tensor<float> &sample)
{
	_process(sample);
}

void MaxPooling::_process(Tensor<float> &in) const
{

	size_t output_width = std::min<size_t>(_target_width, _width);
	size_t output_height = std::min<size_t>(_target_height, _height);
	size_t output_conv_depth = std::min<size_t>(_target_conv_depth, _conv_depth);

	size_t filter_width = _width / output_width;
	size_t filter_height = _height / output_height;
	size_t filter_conv_depth = _conv_depth / output_conv_depth;

	Tensor<float> out(Shape({output_width, output_height, _depth, output_conv_depth}));

	for (size_t x = 0; x < output_width; x++)
		for (size_t y = 0; y < output_height; y++)
			for (size_t z = 0; z < _depth; z++)
				for (size_t k = 0; k < output_conv_depth; k++)
				{
					// the value for pôoling
					float v = 0;

					for (size_t fx = 0; fx < filter_width; fx++)
						for (size_t fy = 0; fy < filter_height; fy++)
							for (size_t fk = 0; fk < filter_conv_depth; fk++)
							{
								v = std::max<float>(v, in.at(x * filter_width + fx, y * filter_height + fy, z, k * filter_conv_depth + fk));
							}

					out.at(x, y, z, k) = v;
				}
	in = out;
}