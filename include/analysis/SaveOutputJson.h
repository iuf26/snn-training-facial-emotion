#ifndef _ANALYSIS_SAVE_OUTPUT_JSON_H
#define _ANALYSIS_SAVE_OUTPUT_JSON_H

#include <fstream>
#include "Analysis.h"
#include "dep/ArduinoJson-v6.17.3.h"

namespace analysis {

	class SaveOutputJson : public UniquePassAnalysis {

	public:
		SaveOutputJson();
		SaveOutputJson(const std::string& train_filename, const std::string& test_filename);
		SaveOutputJson(const SaveOutputJson& that) = delete;
		SaveOutputJson& operator=(const SaveOutputJson& that) = delete;

		void resize(const Shape&);

		void before_train();
		void process_train(const std::string& label, const Tensor<float>& sample);
		void after_train();

		void before_test();
		void process_test(const std::string& label, const Tensor<float>& sample);
		void after_test();

		std::string _to_json_string(const std::string& label, const Tensor<float>& sample);

	private:
		std::string _train_filename;
		std::string _test_filename;
		std::ofstream _json_train_file;
		std::ofstream _json_test_file;
		bool _first_train_sample; // used to write comma between samples
		bool _first_test_sample; // used to write comma between samples
	};

}

#endif
