#ifndef _TENSOR_H
#define _TENSOR_H

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <numeric>
#include "Spike.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/video.hpp>
#include <opencv2/ml.hpp>

#include "dep/ArduinoJson-v6.17.3.h"

#include "Debug.h"

class Shape
{

public:
	Shape() : _dims(), _product()
	{
		_product.push_back(0);
	}

	Shape(const std::vector<size_t> &dims) : _dims(dims), _product()
	{
		for (size_t i = 0; i < _dims.size(); i++)
		{
			_product.push_back(std::accumulate(std::begin(_dims) + i, std::end(_dims), 1, std::multiplies<size_t>()));

		}
		_product.push_back(1);
	}

	Shape(const Shape &that) noexcept : _dims(that._dims), _product(that._product)
	{
	}

	Shape(Shape &&that) noexcept : _dims(std::move(that._dims)), _product(std::move(that._product))
	{
	}

	~Shape()
	{
	}

	Shape& operator=(const Shape& that) noexcept {
		_dims = that._dims;
		_product = that._product;
		return *this;
	}

	/**
	 * @brief return the number of dimentions
	 *
	 * @return size_t
	 */
	size_t number() const
	{
		return _dims.size();
	}

	/**
	 * @brief return the size of the dimention at index i, example of the input is of size 28x28x1, dim(0) returns 28 while dim(2) returns 1.
	 *
	 * @param i index of the dimention
	 * @return size_t
	 */
	size_t dim(size_t i) const
	{
		return _dims.at(i);
	}

	size_t product() const
	{
		return _product.front();
	}

	template <typename... Index>
	size_t to_index(Index &&...index) const
	{
		ASSERT_DEBUG(sizeof...(Index) == _dims.size());
		return _to_index<0, Index...>(std::forward<Index>(index)...);
	}

	bool operator==(const Shape& that) const {
		return _dims == that._dims;
	}

	bool operator!=(const Shape& that) const {
		return _dims != that._dims;
	}

	void print(std::ostream& stream) const {
		stream << "[";

		for(size_t i=0; i<_dims.size(); i++) {
			if(i != 0)
				stream << ", ";
			stream << _dims[i];
		}
		stream << "]";
	}

	std::string to_string() const {
		std::stringstream ss;
		print(ss);
		return ss.str();
	}

private:
	template<size_t Index, typename Head, typename... Tail>
	size_t _to_index(Head head, Tail&&... tail) const {
		ASSERT_DEBUG(static_cast<int64_t>(head) >= 0 && static_cast<int64_t>(head) < static_cast<int64_t>(_dims.at(Index)));
		return _product[Index+1]*head+_to_index<Index+1, Tail...>(std::forward<Tail>(tail)...);
	}

	template<size_t Index>
	size_t _to_index() const {
		return 0;
	}

	std::vector<size_t> _dims;
	std::vector<size_t> _product;

};

template<typename T>
class Tensor {

public:
    typedef T Type;

	Tensor() : _shape(), _data(nullptr) {

	}

	Tensor(const Shape& shape) : _shape(shape), _data(new T[_shape.product()]) {

	}

	Tensor(const Tensor& that) noexcept : _shape(that._shape), _data(new T[_shape.product()]) {
		std::copy(that._data, that._data+_shape.product(), _data);
	}

	Tensor(Tensor&& that) noexcept : _shape(std::move(that._shape)), _data(that._data) {
		that._data = nullptr;
	}

	~Tensor() {
		delete[] _data;
	}

	Tensor& operator=(const Tensor& that) noexcept {
		if(_shape.product() != that.shape().product()) {
			delete[] _data;
			_data = new T[that._shape.product()];
		}
		_shape = that._shape;
		std::copy(that._data, that._data+_shape.product(), _data);
		return *this;
	}

	Tensor& operator=(Tensor&& that) noexcept {
		delete[] _data;
		_shape = std::move(that._shape);
		_data = that._data;
		that._data = nullptr;
		return *this;
	}

	template<typename... Index>
	T& at(Index&&... index) {
		return _data[_shape.to_index(std::forward<Index>(index)...)];
	}

	template<typename... Index>
	T at(Index&&... index) const {
		return _data[_shape.to_index(std::forward<Index>(index)...)];
	}

	template<typename... Index>
	T* ptr(Index&&... index) {
		return _data+_shape.to_index(std::forward<Index>(index)...);
	}

	template<typename... Index>
	const T* ptr(Index&&... index) const {
		return _data+_shape.to_index(std::forward<Index>(index)...);
	}

	T& at_index(size_t index) {
		ASSERT_DEBUG(index < _shape.product());
		return _data[index];
	}

	T at_index(size_t index) const {
		ASSERT_DEBUG(index < _shape.product());
		return _data[index];
	}

	T* ptr_index(size_t index) {
		ASSERT_DEBUG(index < _shape.product());
		return _data+index;
	}

	const T* ptr_index(size_t index) const {
		ASSERT_DEBUG(index < _shape.product());
		return _data+index;
	}

	const Shape& shape() const {
		return _shape;
	}

	void reshape(const Shape& shape) {
		if(shape.product() != _shape.product()) {
			throw std::runtime_error("reshape: Shape must be of same length");
		}
		_shape = shape;
	}

	/**
	 * @brief check if the tensor is empty.
	 *
	 * @return true
	 * @return false
	 */
	bool is_empty()
	{
		return this->_shape.number() == 0;
	}

	T *begin()
	{
		return _data;
	}

	const T *begin() const
	{
		return _data;
	}

	T *end()
	{
		return _data + _shape.product();
	}

	const T *end() const
	{
		return _data + _shape.product();
	}

	void fill(T value)
	{
		std::fill(begin(), end(), value);
	}

	void range_normalize(T min = 0.0, T max = 1.0)
	{
		auto it = std::minmax_element(begin(), end());
		T cmin = *it.first;
		T cmax = *it.second;
		size_t size = _shape.product();

		if(cmin == cmax) {
			for(size_t i=0; i<size; i++) {
				_data[i] = min;
			}
		}
		else {
			for(size_t i=0; i<size; i++) {
				_data[i] = ((_data[i]-cmin)/(cmax-cmin))*(max-min)+min;
			}
		}

	}

	std::pair<T, T> min_max_exclude(T v = std::numeric_limits<T>::max()) const {
		size_t size = _shape.product();
		T min = std::numeric_limits<T>::max();
		T max = std::numeric_limits<T>::lowest();

		for(size_t i=0; i<size; i++) {
			if(_data[i] != v) {
				min = std::min(min, _data[i]);
				max = std::max(max, _data[i]);
			}
		}

		return std::make_pair(min, max);
	}

	void range_normalize_exclude(T v = std::numeric_limits<T>::max(), T min = 0.0, T max = 1.0)
	{
		auto minmax = min_max_exclude(v);
		size_t size = _shape.product();
		for (size_t i = 0; i < size; i++)
		{
			_data[i] = ((_data[i] - minmax.first) / (minmax.second - minmax.first)) * (max - min) + min;
		}
	}

	/**
	 * @brief This function removes negative values from a tensor of floats.
	 *
	 * @param in a tensor of floats.
	 */
	static void absolute_value_tensor(Tensor<float> &in)
	{
		size_t size = in.shape().product(); // a product of all the dimentions.

		for (size_t i = 0; i < size; i++)
		{
			in.at_index(i) = std::abs(in.at_index(i));
		}
	}

	/**
	 * @brief This function takes a tensor and devides all the pixels by the max pixel intensity.
	 *
	 */
	static void normalize_tensor_by_scalar(Tensor<float> &in, size_t _scalar)
	{
		size_t size = in.shape().product(); // a product of all the dimentions.

		// divide by scalar
		for (size_t i = 0; i < size; i++)
		{
			in.at_index(i) = in.at_index(i) / _scalar; // divide by 255
		}
	}

	/**
	 * @brief This function takes a tensor and devides all the pixels by the max pixel intensity.
	 *
	 */
	static void normalize_tensor_by_max(Tensor<float> &in)
	{
		size_t size = in.shape().product(); // a product of all the dimentions.

		float _max = 0;
		for (size_t i = 0; i < size; i++)
			if (in.at_index(i) > _max)
				_max = in.at_index(i);

		// float _min = _max;
		// for (size_t i = 0; i < size; i++)
		// 	if (in.at_index(i) < _min)
		// 		_min = in.at_index(i);

		// divide by max
		for (size_t i = 0; i < size; i++)
		{
			in.at_index(i) = in.at_index(i) / _max; // divide by 255
		}
	}

	/**
	 * @brief This function takes a tensor and devides all the pixels by the max pixel intensity.
	 *
	 */
	static void display_range_tensor(Tensor<float> &in)
	{
		size_t size = in.shape().product(); // a product of all the dimentions.

		float _max = 0;
		for (size_t i = 0; i < size; i++)
			if (in.at_index(i) > _max)
				_max = in.at_index(i);

		float _min = _max;
		for (size_t i = 0; i < size; i++)
			if (in.at_index(i) < _min && _min !=0)
				_min = in.at_index(i);

		std::cout << "The range is: " << std::to_string(_min) << "    -    " << std::to_string(_max) << std::endl;
	}

	/**
	 * @brief This function takes a tensor and devides all the pixels by the max pixel intensity.
	 *
	 */
	static void normalize_tensor_log(Tensor<float> &in)
	{
		size_t size = in.shape().product(); // a product of all the dimentions.

		float _max = 0;
		for (size_t i = 0; i < size; i++)
			if (in.at_index(i) > _max)
				_max = in.at_index(i);

		float c = _max / log(_max + 1);

		// log scale
		for (size_t i = 0; i < size; i++)
		{
			in.at_index(i) = in.at_index(i) == 0 ? 0 : c * log(float(1 + in.at_index(i))); // / log(10)); // divide by 255
																						   // in.at_index(i) = in.at_index(i) == 0? 0 : std::abs(log(in.at_index(i))); // / log(10)); // divide by 255
		}
	}

	/**
	 * @brief This function takes a tensor and devides all the pixels by 255.
	 *
	 */
	static void normalize_tensor(Tensor<float> &in)
	{
		size_t size = in.shape().product(); // a product of all the dimentions.

		for (size_t i = 0; i < size; i++)
		{
			in.at_index(i) = in.at_index(i) / static_cast<float>(std::numeric_limits<uint8_t>::max()); // divide by 255
		}
	}

	/**
	 * @brief This function prints the pixels of a tensor in the terminal, only useful for testing and debugging.
	 *
	 */
	static void print_tensor_pixels(Tensor<float> &in)
	{
		size_t size = in.shape().product(); // a product of all the dimentions
		for (size_t i = 0; i < size; i++)
		{
			std::cout << in.at_index(i) << ", ";
		}
	}

	/**
	 * @brief This function is to test if there is any data in the tensor, only useful for testing and debugging.
	 *
	 */
	static bool tensor_contain_info(Tensor<float> &in)
	{
		size_t size = in.shape().product(); // a product of all the dimentions to get the total size
		for (size_t i = 0; i < size; i++)
		{
			if (in.at_index(i) > 0 && in.at_index(i) < 1)
				return true;
		}
		return false;
	}

	/**
	 * @brief This function returns a tensor at a certain conv depth.
	 *
	 * @param in Tensor with large conv depth.
	 * @param out Tensor with a single conv depth.
	 * @param conv_depth the number at which the depth is.
	 */
	Tensor<float> tensor_at_conv_depth(Tensor<float> &in, size_t conv_depth)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		Tensor<float> out(Shape({_height, _width, _depth}));

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					out.at(i, j, k) = in.at(i, j, k, conv_depth);
				}
		return out;
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrices_to_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames[0].channels();
		size_t _conv_depth = frames.size();

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, _conv_count) = _frame.at<float>(i, j, k);
					}
			_conv_count++;
		}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrices_to_2channel_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = 2;
		size_t _conv_depth = frames.size();

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, _conv_count) = _frame.at<float>(i, j, k);
					}
			_conv_count++;
		}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void multi_matrices_to_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out, size_t depth)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames[0].channels();
		size_t _conv_depth = frames.size() / depth;

		out = Tensor<float>(Shape({_height, _width, depth, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;
		size_t _depth_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, _depth_count, _conv_count) = _frame.at<float>(i, j, k);
					}
			_depth_count++;
			if (_depth_count == depth)
			{
				_depth_count = 0;
				_conv_count++;
			}
		}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrices_to_split_sign_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames[0].channels();
		size_t _conv_depth = frames.size();

		out = Tensor<float>(Shape({_height, _width, _depth * 2, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						if (_frame.at<float>(i, j, k) > 0)
							out.at(i, j, k * 2, _conv_count) = _frame.at<float>(i, j, k);
						else
							out.at(i, j, k * 2 + 1, _conv_count) = std::abs(_frame.at<float>(i, j, k));
					}
			_conv_count++;
		}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrices_to_abs_sign_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames[0].channels();
		size_t _conv_depth = frames.size();

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, _conv_count) = std::abs(_frame.at<float>(i, j, k));
					}
			_conv_count++;
		}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrices_to_colored_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames[0].channels();
		size_t _conv_depth = frames.size();

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, _conv_count) = _frame.at<cv::Vec3b>(i, j)[k];
					}
			_conv_count++;
		}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrices_to_colored_f_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames[0].channels();
		size_t _conv_depth = frames.size();

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, _conv_count) = _frame.at<cv::Vec3f>(i, j)[k];
					}
			_conv_count++;
		}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrices_to_colored_scaled_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames[0].channels();
		size_t _conv_depth = frames.size();

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));
		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, _conv_count) = _frame.at<cv::Vec3b>(i, j)[k] * 100;
					}
			_conv_count++;
		}
	}

	// // imwrite("/home/mireille/Desktop/testframes/frame_difference_" + std::to_string(_conv_count) + ".png", _frame);
	// /**
	//  * @brief This function takes an opencv matrix and returns a tensor.
	//  *
	//  * @param frame The opencv matrix.
	//  */
	// static void matrix_to_colored_tensor(const cv::Mat &frame, Tensor<float> &out)
	// {
	// 	size_t _width = frame.size[0];
	// 	size_t _height = frame.size[1];
	// 	size_t _depth = frame.size.dims();
	// 	size_t _conv_depth = 1;

	// 	out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

	// 	for (size_t _conv = 0; _conv < _conv_depth; _conv++)
	// 		for (size_t i = 0; i < _height; i++)
	// 			for (size_t j = 0; j < _width; j++)
	// 				for (size_t k = 0; k < _depth; k++)
	// 				{
	// 					out.at(i, j, k, _conv) = frame.at<float>(j, i, k);
	// 				}
	// }

	// imwrite("/home/mireille/Desktop/testframes/frame_difference_" + std::to_string(_conv_count) + ".png", _frame);
	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrix3D_to_tensor(const cv::Mat &frame, Tensor<float> &out)
	{
		size_t _height = frame.size[0];
		size_t _width = frame.size[1];
		size_t _depth = frame.channels();
		size_t _conv_depth = 1;

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
					for (size_t m = 0; m < _conv_depth; m++)
					{
						out.at(i, j, k, m) = frame.at<float>(i, j, k);
					}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_3Dmatrices(std::vector<cv::Mat> &frames, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						frame.at<float>(i, j, k) = in.at(i, j, k, conv);
					}
			frames.push_back(frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_3d_matrices(std::vector<cv::Mat> &frames, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		size_t sizes[] = {_height, _width, _depth};
		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame = cv::Mat(_depth, sizes, CV_32FC2, cv::Scalar(0));

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						frame.at<float>(i, j, k) = in.at(i, j, k, conv);
					}
			frames.push_back(frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_matrix(cv::Mat &frame, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						frame.at<float>(i, j) = in.at(i, j, k, conv);
					}
		}
	}

	// /**
	//  * @brief This function takes an opencv matrix and returns a tensor.
	//  *
	//  * @param frame The opencv matrix.
	//  */
	// static void matrix_to_tensor(const cv::Mat &frame, Tensor<float> &out)
	// {
	// 	size_t _height = frame.size[0];
	// 	size_t _width = frame.size[1];
	// 	size_t _depth = frame.channels();
	// 	size_t _conv_depth = 1;

	// 	out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

	// 	for (size_t i = 0; i < _height; i++)
	// 		for (size_t j = 0; j < _width; j++)
	// 		{
	// 			out.at(i, j, 0, 0) = frame.at<float>(i, j);
	// 		}
	// }

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrix_to_tensor(cv::Mat &frame, Tensor<float> &out)
	{
		size_t _height = frame.size[0];
		size_t _width = frame.size[1];
		size_t _depth = frame.channels();

		out = Tensor<float>(Shape({_height, _width, _depth, 1}));

		// add the data inside the tensor.
		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					out.at(i, j, k, 0) = frame.at<float>(i, j, k);
				}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrix_to_scaled_tensor(const cv::Mat &frame, Tensor<float> &out)
	{
		size_t _height = frame.size[0];
		size_t _width = frame.size[1];
		size_t _depth = frame.channels();
		size_t _conv_depth = 1;

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
			{
				out.at(i, j, 0, 0) = frame.at<float>(i, j) * 255;
			}
	}

	/**
	 * @brief This function takes an opencv matrix and returns a tensor.
	 *
	 * @param frame The opencv matrix.
	 */
	static void matrix_to_colored_tensor(const cv::Mat &frame, Tensor<float> &out)
	{
		size_t _height = frame.size[0];
		size_t _width = frame.size[1];
		size_t _depth = frame.channels();
		size_t _conv_depth = 1;

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					out.at(i, j, k, 0) = frame.at<cv::Vec3b>(i, j)[k];
				}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_matrices(std::vector<cv::Mat> &frames, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
				{
					frame.at<float>(i, j) = in.at(i, j, 0, conv);
				}
			frames.push_back(frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices. This function does't dismiss deoth, but transforms depth into even more frames.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_multi_matrices(std::vector<cv::Mat> &frames, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
			for (size_t k = 0; k < _depth; k++)
			{
				cv::Mat frame(_height, _width, CV_32F);
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
						frame.at<float>(i, j) = in.at(i, j, k, conv);

				frames.push_back(frame);
			}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_unscaled_matrices(std::vector<cv::Mat> &frames, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
				{
					frame.at<float>(i, j) = in.at(i, j, 0, conv) * 255;
				}
			frames.push_back(frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_scale_matrices(std::vector<cv::Mat> &frames, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F);
			// cv::Mat frame(_height, _width, CV_32FC2);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
				{
					frame.at<float>(i, j) = in.at(i, j, 0, conv) * 255;
				}
			frames.push_back(frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_colored_matrices(std::vector<cv::Mat> &frames, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32FC3);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (int k = 0; k < _depth; k++)
					{ // Vec3b
						frame.at<cv::Vec3f>(i, j)[k] = in.at(i, j, k, conv);
					}
			frames.push_back(frame);
		}
	}

	/**
	 * @brief This function takes 4 direction opencv matricies and returns a tensor with 4 channels.
	 * So the directions are added as channels.
	 *
	 * @param frame The opencv matrix.
	 */
	static void direction_matrices_to_tensor(const std::vector<cv::Mat> &frames, Tensor<float> &out)
	{
		size_t _height = frames[0].size[0];
		size_t _width = frames[0].size[1];
		size_t _depth = frames.size();
		size_t _conv_depth = out.shape().dim(3);

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		// depth is incremented with every frame.
		size_t _depth_count = 0;

		for (cv::Mat _frame : frames)
		{ // add the data inside the tensor.
			for (size_t conv = 0; conv < _conv_depth; conv++)
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
					{
						out.at(i, j, _depth_count, conv) = _frame.at<float>(i, j);
					}
			_depth_count++;
		}
	}

	/**
	 * @brief This function takes a vector of tensors and returns one single tensor that has these tendors in the temporal depth.
	 *
	 * @param tensors The tensor vector of floats.
	 */
	static void tensors_to_tensor(const std::vector<Tensor<float>> &tensors, Tensor<float> &out)
	{
		size_t _height = tensors[0].shape().dim(0);
		size_t _width = tensors[0].shape().dim(1);
		size_t _depth = tensors[0].shape().dim(2);
		size_t _tmp_depth = tensors[0].shape().dim(3);
		size_t _conv_depth = tensors.size();

		out = Tensor<float>(Shape({_height, _width, _depth, _conv_depth}));

		// CONV_DEPTH by being incremented every frame.
		size_t _conv_count = 0;

		for (Tensor<float> _tensor : tensors)
		{ // add the data inside the tensor.
			for (size_t conv = 0; conv < _tmp_depth; conv++)
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
						for (size_t k = 0; k < _depth; k++)
						{
							out.at(i, j, k, _conv_count) = _tensor.at(i, j, k, conv);
						}

			_conv_count++;
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of tensors.
	 *
	 */
	static void tensor_to_tensors(std::vector<Tensor<float>> &out_tensors, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			Tensor<float> out(Shape({_height, _width, _depth, 1}));

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, 0) = in.at(i, j, k, conv);
					}
			out_tensors.push_back(out);
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void draw_colored_tensor(std::string _path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32FC3);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (int k = 0; k < _depth; k++)
					{
						frame.at<cv::Vec3f>(i, j)[k] = in.at(i, j, k, conv);
					}
			imwrite(_path + std::to_string(conv) + ".png", frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void draw_colored_b_tensor(std::string _path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32FC3);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (int k = 0; k < _depth; k++)
					{
						frame.at<cv::Vec3b>(i, j)[k] = in.at(i, j, k, conv);
					}
			imwrite(_path + std::to_string(conv) + ".png", frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns a vector of opencv matrices.
	 *
	 * @param frame The opencv matrix.
	 */
	static void draw_scaled_colored_tensor(std::string _path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32FC3);

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (int k = 0; k < _depth; k++)
					{
						frame.at<cv::Vec3f>(i, j)[k] = in.at(i, j, k, conv) * 255;
					}
			imwrite(_path + std::to_string(conv) + ".png", frame);
		}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_Mnist_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					frame.at<float>(i, j) = in.at(i, j, k) * 100;
				}

		imwrite(draw_folder_path + ".png", frame);
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_nonscaled_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

			for (size_t k = 0; k < _depth; k++)
			{
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
						frame.at<float>(i, j) = in.at(i, j, k, conv);

				imwrite(draw_folder_path + std::to_string(k) + "_" + std::to_string(conv) + ".png", frame);
			}
		}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_scaled_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

			for (size_t k = 0; k < _depth; k++)
			{
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
						frame.at<float>(i, j) = in.at(i, j, k, conv) * 255;

				imwrite(draw_folder_path + std::to_string(k) + ".png", frame);
			}
		}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

			for (size_t k = 0; k < _depth; k++)
			{
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
						frame.at<float>(i, j) = in.at(i, j, k, conv); // < 1 ? in.at(i, j, k, conv) * 255 : in.at(i, j, k, conv);

				imwrite(draw_folder_path + "_kernel:" + std::to_string(k) + "_tmp:" + std::to_string(conv) + ".png", frame);
			}
		}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_scaled_tensor(std::string draw_folder_path, const Tensor<float> &in, size_t scalar)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

			for (size_t k = 0; k < _depth; k++)
			{
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
						frame.at<float>(i, j) = in.at(i, j, k, conv) * scalar; // < 1 ? in.at(i, j, k, conv) * 255 : in.at(i, j, k, conv);

				imwrite(draw_folder_path + "_kernel:" + std::to_string(k) + "_tmp:" + std::to_string(conv) + ".png", frame);
			}
		}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_one_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);

		cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

		for (size_t k = 0; k < _depth; k++)
		{
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					frame.at<float>(i, j) = in.at(i, j, k, 0); // * 255; // < 1 ? in.at(i, j, k, conv) * 255 : in.at(i, j, k, conv);

			imwrite(draw_folder_path + "_kernel:" + std::to_string(k) + ".png", frame);
		}
	}

	/**
	 * @brief This function takes a tensor and draws only one conv depth and first 5 kernels.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_one_k5_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);

		cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

		for (size_t k = 0; k < 5; k++)
		{
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					frame.at<float>(i, j) = in.at(i, j, k, 0) * 100; // < 1 ? in.at(i, j, k, conv) * 255 : in.at(i, j, k, conv);

			imwrite(draw_folder_path + "_kernel:" + std::to_string(k) + ".png", frame);
		}
	}

	/**
	 * @brief This function takes a tensor and draws only one conv depth and the first kernel.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_one_k1_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);

		cv::Mat frame(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				frame.at<float>(i, j) = in.at(i, j, 0, 0) * 255; // < 1 ? in.at(i, j, k, conv) * 255 : in.at(i, j, k, conv);

		imwrite(draw_folder_path + ".png", frame);
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_resized_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		cv::Size _frame_size(500, 500);
		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			cv::Mat frame(_height, _width, CV_32F);		// When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.
			cv::Mat frame_out(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

			for (size_t k = 0; k < _depth; k++)
			{
				for (size_t i = 0; i < _height; i++)
					for (size_t j = 0; j < _width; j++)
						frame.at<float>(i, j) = in.at(i, j, k, conv) * 100; // * 255; // < 1 ? in.at(i, j, k, conv) * 255 : in.at(i, j, k, conv);

				cv::resize(frame, frame_out, _frame_size);
				imwrite(draw_folder_path + "_" + std::to_string(conv) + ".png", frame_out);
			}
		}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param draw_folder_path The path of the folder to save the frames.
	 * @param frame The opencv matrix.
	 */
	static void draw_split_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		{
			// CONV_DEPTH by being incremented every frame.
			for (size_t conv = 0; conv < _conv_depth; conv++)
			{
				cv::Mat frame_pos(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.
				cv::Mat frame_neg(_height, _width, CV_32F); // When I use CV_32FC3 instead of CV_32F the frame is duplicated and squished.

				for (size_t k = 0; k < _depth; k++)
					for (size_t i = 0; i < _height; i++)
						for (size_t j = 0; j < _width; j++)
						{
							frame_pos.at<float>(i, j) = in.at(i, j, 0, conv); //* 255;
							frame_neg.at<float>(i, j) = in.at(i, j, 1, conv); //* 255;
						}
				imwrite(draw_folder_path + "_pos_" + std::to_string(conv) + ".png", frame_pos);
				imwrite(draw_folder_path + "_neg_" + std::to_string(conv) + ".png", frame_neg);
			}
		}
	}
	/**
	 * @brief This function takes a tensor and draws its representation as an OpenCV matrix image.
	 *
	 * @param in The Tensor of float.
	 */
	static void draw_weight_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _Weights_Number = in.shape().dim(3);
		size_t _conv_depth = in.shape().number() > 4 ? in.shape().dim(4) : 1;

		cv::Size _frame_size(500, 500);
		// CONV_DEPTH by being incremented every frame.
		if (in.shape().number() > 4)
			// looping over the number of filters
			for (size_t _w = 0; _w < _Weights_Number; _w++)
			{
				cv::Mat frame(_height, _width, CV_32FC3);
				// looping over temporal depth -> video frames
				for (size_t conv = 0; conv < _conv_depth; conv++)
					// lopping over the image
					for (size_t i = 0; i < _height; i++)
						for (size_t j = 0; j < _width; j++)
							for (size_t k = 0; k < _depth; k++) // depth is always 2 = OnOffChannels
							{
								frame.at<cv::Vec3f>(i, j)[k] = in.at(i, j, k, _w, conv) * 255;
							}
				if (_depth < 4)
					cv::resize(frame, frame, _frame_size);
				imwrite(draw_folder_path + "_W:" + std::to_string(_w) + ".png", frame);
			}
		else
			// 2D
			for (size_t _w = 0; _w < _Weights_Number; _w++)
			{
				cv::Mat frame(_height, _width, CV_32FC3);
				for (size_t conv = 0; conv < _conv_depth; conv++)
					for (size_t i = 0; i < _height; i++)
						for (size_t j = 0; j < _width; j++)
							for (size_t k = 0; k < _depth; k++)
							{
								frame.at<cv::Vec3f>(i, j)[k] = in.at(i, j, k, _w) * 255;
							}

				cv::resize(frame, frame, _frame_size);
				imwrite(draw_folder_path + "_W:" + std::to_string(_w) + ".png", frame);
			}
	}

	/**
	 * @brief This function takes a tensor and returns one opencv matrix.
	 *
	 * @param frame The opencv matrix.
	 */
	static void draw_feature_tensor(std::string draw_folder_path, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		cv::Size _frame_size(500, 500);
		// CONV_DEPTH by being incremented every frame.
		cv::Mat frame(_height, _width, CV_32F);
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						frame.at<float>(i, j) = in.at(i, j, k, conv) * 100;
					}
		}
		cv::resize(frame, frame, _frame_size);
		imwrite(draw_folder_path + "features.png", frame);
	}

	/**
	 * @brief This function takes a tensor and returns one flattened opencv matrix (needed for SVM).
	 *
	 * @param frame The opencv matrix.
	 */
	static void tensor_to_flat_matrix(cv::Mat &frame, const Tensor<float> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);
		size_t _count = 0;

		// CONV_DEPTH by being incremented every frame.
		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						frame.at<float>(1, _count) = in.at(i, j, k, conv);

						_count++;
					}
		}
	}

	/**
	 * @brief This function saves a tensor into a JSON file, this function only saves the dimentions of the tensor and its values.
	 *
	 * @param fileName The saving directory and name of the json file.
	 */
	void saveJsonString(std::string fileName)
	{
		std::ofstream _jsonTextFile;
		_jsonTextFile.open(fileName, std::ios_base::app);

		std::string JSON_output;
		std::string type_Name;
		DynamicJsonDocument doc(JSON_ARRAY_SIZE(this->shape().product() + this->shape().number()));

		size_t size = this->shape().product();

		uint8_t dim_number = this->shape().number();
		for (size_t i = 0; i < dim_number; i++)
		{
			type_Name = "dim_" + std::to_string(i);
			doc[type_Name] = this->shape().dim(i);
		}

		for (size_t i = 0; i < size; i++)
		{
			doc["data"][i] = this->at_index(i);
		}
		serializeJson(doc, JSON_output);
		_jsonTextFile << JSON_output << ",";
		JSON_output = "";

		_jsonTextFile.close();
	}

	/**
	 * @brief This function saves a tensor into a JSON file, saves the label, dimentions and values of the tensor.
	 *
	 * @param fileName The saving directory and name of the json file.
	 * @param label The tensor label.
	 */
	void saveTensorLabelJsonString(std::string fileName, std::string label)
	{
		std::ofstream _jsonTextFile;
		_jsonTextFile.open(fileName, std::ios_base::app);

		std::string JSON_output;
		std::string type_Name;
		DynamicJsonDocument doc(JSON_ARRAY_SIZE(this->shape().product() + this->shape().number()));

		doc["label"] = label;

		size_t size = this->shape().product();

		uint8_t dim_number = this->shape().number();
		for (size_t i = 0; i < dim_number; i++)
		{
			type_Name = "dim_" + std::to_string(i);
			doc[type_Name] = this->shape().dim(i);
		}

		for (size_t i = 0; i < size; i++)
		{
			doc["data"][i] = this->at_index(i);
		}
		serializeJson(doc, JSON_output);
		_jsonTextFile << JSON_output << ",";
		JSON_output = "";

		_jsonTextFile.close();
	}

	/**
	 * @brief This function takes a tensor of time which has a temporal depth of more than one and returns multiple tensors of time.
	 */
	static void time_tensor_to_tensors(Tensor<Time> &in, std::vector<Tensor<Time>> &out)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		for (size_t conv = 0; conv < _conv_depth; conv++)
		{
			Tensor<Time> t(Shape({_height, _width, _depth, 1}));

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						t.at(i, j, k, 0) = in.at(i, j, k, conv);
					}
			out.push_back(t);
		}
	}

	/**
	 * @brief This function takes multiple tensors of time and returns a tensor of time.
	 */
	static void tensors_to_time_tensor(std::vector<Tensor<Time>> &in, Tensor<Time> &out)
	{
		size_t _height = in[0].shape().dim(0);
		size_t _width = in[0].shape().dim(1);
		size_t _depth = in[0].shape().dim(2);
		size_t _conv_depth = in.size();

		for (size_t conv = 0; conv < _conv_depth; conv++)
			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, conv) = in[conv].at(i, j, k, 0);
					}
	}

	/**
	 * @brief This function takes a multiple tensors of time and returns tensor of time.
	 */
	static void time_tensors_to_tensor(std::vector<Tensor<Time>> &in, Tensor<Time> &out)
	{
		size_t _height = in[0].shape().dim(0);
		size_t _width = in[0].shape().dim(1);
		size_t _depth = in[0].shape().dim(2);

		for (size_t conv = 0; conv < in.size(); conv++)
		{
			Tensor<Time> t = in[conv];

			for (size_t i = 0; i < _height; i++)
				for (size_t j = 0; j < _width; j++)
					for (size_t k = 0; k < _depth; k++)
					{
						out.at(i, j, k, conv) = t.at(i, j, k, 0);
					}
		}
	}

	/**
	 * @brief This function takes a tensor of time which has a temporal depth of 2 returns two tensors of time.
	 */
	static void split_time_tensor(Tensor<Time> &t1, Tensor<Time> &t2, Tensor<Time> &in)
	{
		size_t _height = in.shape().dim(0);
		size_t _width = in.shape().dim(1);
		size_t _depth = in.shape().dim(2);
		size_t _conv_depth = in.shape().dim(3);

		ASSERT_DEBUG(_conv_depth == 2);

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					t1.at(i, j, k, 0) = in.at(i, j, k, 0);
					t2.at(i, j, k, 0) = in.at(i, j, k, 1);
				}
	}

	/**
	 * @brief This function takes two tensors of time and returns a tensor of time which has a temporal depth of 2.
	 */
	static void join_time_tensor(Tensor<Time> &t1, Tensor<Time> &t2, Tensor<Time> &out)
	{
		size_t _height = out.shape().dim(0);
		size_t _width = out.shape().dim(1);
		size_t _depth = out.shape().dim(2);
		size_t _conv_depth = out.shape().dim(3);

		ASSERT_DEBUG(_conv_depth == 2);

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					out.at(i, j, k, 0) = t1.at(i, j, k, 0);
					out.at(i, j, k, 1) = t2.at(i, j, k, 0);
				}
	}

	/**
	 * @brief This function takes two tensors of time and returns a tensor of time which has a temporal depth of 1.
	 */
	static void fuse_time_tensor(Tensor<Time> &t1, Tensor<Time> &t2, Tensor<Time> &out)
	{
		size_t _height = out.shape().dim(0);
		size_t _width = out.shape().dim(1);
		size_t _depth = out.shape().dim(2);
		size_t _conv_depth = out.shape().dim(3);

		ASSERT_DEBUG(_conv_depth == 1);

		for (size_t i = 0; i < _height; i++)
			for (size_t j = 0; j < _width; j++)
				for (size_t k = 0; k < _depth; k++)
				{
					out.at(i, j, k, 0) = t1.at(i, j, k, 0) + t2.at(i, j, k, 0);
				}
	}

	void save(std::ostream &stream) const
	{
		uint8_t dim_number = _shape.number();
		stream.write(reinterpret_cast<const char *>(&dim_number), sizeof(uint8_t));
		for (size_t i = 0; i < dim_number; i++)
		{
			uint16_t dim = _shape.dim(i);
			stream.write(reinterpret_cast<const char *>(&dim), sizeof(uint16_t));
		}
		size_t size = _shape.product();
		size_t null_value = 0;
		for (uint32_t i = 0; i < size; i++)
		{
			if (at_index(i) == 0.0)
				null_value++;
		}

		if (null_value > size / 2 - 1)
		{ // sparse
			uint8_t flag = 0;
			stream.write(reinterpret_cast<const char *>(&flag), sizeof(uint8_t));
			for (uint32_t i = 0; i < size; i++)
			{
				if (at_index(i) != 0.0)
				{
					stream.write(reinterpret_cast<const char *>(&i), sizeof(uint32_t));
					float f = at_index(i);
					stream.write(reinterpret_cast<const char *>(&f), sizeof(float));
				}
			}
			uint32_t eol = 0xFFFFFFFF;
			stream.write(reinterpret_cast<const char *>(&eol), sizeof(uint32_t));
		}
		else
		{ // dense
			uint8_t flag = 1;
			stream.write(reinterpret_cast<const char *>(&flag), sizeof(uint8_t));
			stream.write(reinterpret_cast<const char *>(begin()), sizeof(float) * size);
		}
	}

	void load(std::istream &stream)
	{
		uint8_t dim_number;
		stream.read(reinterpret_cast<char *>(&dim_number), sizeof(uint8_t));
		std::vector<size_t> dims;
		dims.reserve(dim_number);
		for (size_t i = 0; i < dim_number; i++)
		{
			uint16_t dim;
			stream.read(reinterpret_cast<char *>(&dim), sizeof(uint16_t));
			dims.push_back(dim);
		}

		Shape new_shape(dims);

		if(new_shape.product() != _shape.product()) {
			delete[] _data;
			_data = new T[new_shape.product()];
		}

		_shape = std::move(new_shape);

		size_t size = _shape.product();

		uint8_t flag = 0;
		stream.read(reinterpret_cast<char*>(&flag), sizeof(uint8_t));

		if(flag == 0) { //sparse
			uint32_t index;
			stream.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
			float value;
			while(index != 0xFFFFFFFF) {
				stream.read(reinterpret_cast<char*>(&value), sizeof(float));
				at_index(index) = value;
				stream.read(reinterpret_cast<char*>(&index), sizeof(uint32_t));
			}
		}
		else if(flag == 1){
			stream.read(reinterpret_cast<char*>(begin()), sizeof(float)*size);
		}
		else {
			throw std::runtime_error("Tensor load: unkown flag");
		}
	}

private:
	Shape _shape;
	T* _data;

};

#endif
