#ifndef _ANALYSIS_SAVE_OUTPUT_NUMPY_H
#define _ANALYSIS_SAVE_OUTPUT_NUMPY_H

#include<iostream>
#include "Analysis.h"
#include"include/npy.hpp"

namespace analysis {

	class SaveOutputNumpy : public UniquePassAnalysis {

	public:
		SaveOutputNumpy();
		SaveOutputNumpy(const std::string& train_filename, const std::string& test_filename);
		SaveOutputNumpy(const SaveOutputNumpy& that) = delete;
		SaveOutputNumpy& operator=(const SaveOutputNumpy& that) = delete;

		void resize(const Shape&);

		void before_train();
		void process_train(const std::string& label, const Tensor<float>& sample);
		void after_train();

		void before_test();
		void process_test(const std::string& label, const Tensor<float>& sample);
		void after_test();

	private:

		std::string _train_filename;
		std::string _test_filename;

		std::vector<float> _data_train;
		std::vector<float> _data_test;
	};

}

#endif
