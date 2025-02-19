#ifndef _DATASET_IMAGE_H
#define _DATASET_IMAGE_H

#include <filesystem>
#include <string>
#include <cassert>
#include <fstream>
#include <limits>
#include <tuple>

#include "Tensor.h"
#include "Input.h"

/**
 * @brief An image sample in IMAGE is 28x28 and with a depth of 1 correspoiding to Greyscale.
 *
 * @param IMAGE_WIDTH is the width of the frame.
 * @param IMAGE_HEIGHT is the height of the frame.
 * @param CONV_DEPTH is the depth of the convolution in case of 3D convolution.
 */
#define IMAGE_WIDTH 1
#define IMAGE_HEIGHT 1
#define CONV_DEPTH 1
#define JOIN_FRAMES 1 // 1 //manipulates the number of fused samples at the end (also used in case of temporal pooling)

namespace dataset
{

	/**
	 * @brief This class fills images into tensors of floats.
	 * @param images_folder_path string - The path to the folder that contains the training and testing images.
	 * @param frame_size_width frame width size that is set to zero takes the default size.
	 * @param frame_size_height frame height size that is set to zero takes the default size.
	 * @param grey a flag that turns a frame into greyscale.
	 */
	class Image : public Input
	{

	public:
		Image(const std::string &images_folder_path, const size_t &frame_size_width = 0, const size_t &frame_size_height = 0, const size_t &grey = 0, size_t max_read = std::numeric_limits<size_t>::max());
		/**
		 * @brief A function that is called to fetch the next sample if the size is not reached. check the load function of the OptimizedLayerByLayer class.
		 */
		virtual bool has_next() const;

		/**
		 * @brief loads a image sample using openCV.
		 *
		 * @return std::pair<std::string, Tensor<InputType>>
		 */
		virtual std::pair<std::string, Tensor<InputType>> next();

		virtual uint32_t assign_label_to_sample(std::string _current_images_path);

		virtual void reset();
		virtual void close();

		size_t size() const;
		virtual std::string to_string() const;

		virtual const Shape &shape() const;

	private:
		uint32_t swap(uint32_t v);
		// The path of the folder that contains the images.
		std::string _images_folder_path;
		uint32_t _frame_size_width;
		uint32_t _frame_size_height;
		// Number of frames to be fused into one sample.
		size_t _grey;
		// A list that contains all the video names.
		std::vector<std::string> _images_path_list;
		// A list to store folder names which are  the label names.
		std::vector<std::string> _label_list;
		std::string _current_images_path;
		uint32_t _folder_cursor;
		uint32_t _frame_cursor;
		uint32_t _size;
		// counter for the number of frames that form a video sample
		uint32_t _frame_number;
		Shape _shape;
		// The threshould that checks if the image is empty. (checks if the sum of pîxel intensity is less than this threshould.)
		int _threshould;
		uint32_t _max_read;
	};

}

#endif
