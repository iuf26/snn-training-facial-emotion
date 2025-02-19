#include "dataset/Image.h"
#include <iostream>
#include <algorithm> 

using namespace dataset;

Image::Image(const std::string &images_folder_path, 
             const size_t &frame_size_width, 
             const size_t &frame_size_height, 
             const size_t &grey, 
             size_t max_read) 
    : _images_folder_path(images_folder_path), 
      _frame_size_width(frame_size_width), 
      _frame_size_height(frame_size_height), 
      _grey(grey), 
      _size(0), 
      _frame_cursor(0), 
      _max_read(max_read)
{

	std::vector<cv::String> fnSize;
	std::cout << "Searching for images in folder: " << _images_folder_path << std::endl;

	for (const auto &file : std::filesystem::directory_iterator(_images_folder_path))
	{
		std::string folder_path = file.path();
		if (!std::filesystem::is_directory(folder_path)) {
			continue; 
		}
		std::string label = folder_path.substr(folder_path.find_last_of("/") + 1);
		std::vector<cv::String> image_files;
		cv::glob(folder_path + "/*.png", image_files, false);

		if (image_files.empty()) {
			std::cerr << "Warning: No images found in folder: " << folder_path << std::endl;
		}else{
			_label_list.push_back(label);
		}
		for (const auto &image_path : image_files) {
			_images_path_list.push_back(image_path);
			// std::cout << "Found image: " << image_path << std::endl;
		}
	}
	std::cout << "Total number of classes (labels): " << _label_list.size() << std::endl;
	if (_images_path_list.empty()) {
		throw std::runtime_error("No images found in dataset path: " + _images_folder_path);
	}
	cv::Mat _size_frame = cv::imread(_images_path_list[0]);
	if (_size_frame.empty()) {
		throw std::runtime_error("Error loading first image: " + _images_path_list[0]);
	}

	if (_grey == 1)
		cv::cvtColor(_size_frame, _size_frame, cv::COLOR_BGR2GRAY);

	size_t _width = (_frame_size_width == 0) ? _size_frame.cols : _frame_size_width;
	size_t _height = (_frame_size_height == 0) ? _size_frame.rows : _frame_size_height;
	size_t _depth = _size_frame.channels();

	_shape = Shape(std::vector<size_t>({_height, _width, _depth}));
	std::cout << "Exited from image";
    
}

bool Image::has_next() const
{
    return (_frame_cursor < _images_path_list.size());
}

std::pair<std::string, Tensor<InputType>> Image::next()
{
    if (!has_next()) {
        throw std::runtime_error("No more images to process.");
    }
	std::cout << "frame cursor" << _frame_cursor;
    std::string image_path = _images_path_list[_frame_cursor];
    cv::Mat frame = cv::imread(image_path);

    if (frame.empty()) {
        throw std::runtime_error("Error loading image: " + image_path);
    }

    if (_frame_size_height != 0 || _frame_size_width != 0)
        cv::resize(frame, frame, cv::Size(_frame_size_width, _frame_size_height));

    if (_grey == 1)
        cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);

    // Assign a label
    size_t _label = assign_label_to_sample(image_path);
    std::pair<std::string, Tensor<InputType>> out(std::to_string(_label), _shape);

    // Convert image to tensor
    for (int i = 0; i < frame.rows; i++)
        for (int j = 0; j < frame.cols; j++)
            for (int k = 0; k < frame.channels(); k++)
            {
                if (frame.channels() > 1)
                    out.second.at(i, j, k) = (frame.at<cv::Vec3b>(i, j)[k]);
                else
                    out.second.at(i, j, k) = (frame.at<unsigned char>(i, j));
            }

    _frame_cursor++;  // Move to the next image

    return out;
}

void Image::reset()
{
    _frame_cursor = 0;
}

uint32_t Image::assign_label_to_sample(std::string image_path)
{
    std::string _label = image_path.substr(0, image_path.find_last_of("/"));
    _label = _label.substr(_label.find_last_of("/") + 1);

    size_t _label_index = std::distance(_label_list.begin(), std::find(_label_list.begin(), _label_list.end(), _label));
    return _label_index;
}

void Image::close()
{
    // No specific close operations needed
}

size_t Image::size() const
{
    return std::min(static_cast<size_t>(_images_path_list.size()), static_cast<size_t>(_max_read));
}

std::string Image::to_string() const
{
    return "Image Dataset at (" + _images_folder_path + ")";
}

const Shape &Image::shape() const
{
    return _shape;
}
