#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>
#include <algorithm>
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
#include "process/MaxScaling.h"
#include "process/SimplePreprocessing.h"
#include "process/SeparateSign.h"

using json = nlohmann::json;

// === SVM_Predictor class (inline) ===
class SVM_Predictor {
private:
    svm_model* model;

public:
    SVM_Predictor(const std::string& model_path) {
        model = svm_load_model(model_path.c_str());
        if (!model) {
            throw std::runtime_error("Failed to load SVM model from: " + model_path);
        }
    }

    ~SVM_Predictor() {
        if (model) {
            svm_free_and_destroy_model(&model);
        }
    }

    bool isLoaded() {
        return model != nullptr;
    }

    double predict(const std::vector<float>& features) {
        if (!model) {
            throw std::runtime_error("SVM model not loaded.");
        }

        svm_node* nodes = new svm_node[features.size() + 1];
        for (size_t i = 0; i < features.size(); ++i) {
            nodes[i].index = static_cast<int>(i + 1);
            nodes[i].value = features[i];
        }
        nodes[features.size()].index = -1;

        double label = svm_predict(model, nodes);
        delete[] nodes;
        return label;
    }
};

// === Load convolution weights from JSON ===
void load_convolution_weights_from_json(layer::Convolution& conv_layer, const std::string& json_path) {
    std::ifstream file(json_path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open JSON file at: " + json_path);
    }

    json j;
    file >> j;

    int dim_0 = j["dim_0"];
    int dim_1 = j["dim_1"];
    int dim_2 = j["dim_2"];
    int dim_3 = j["dim_3"];
    const std::vector<float>& data = j["data"];

    Tensor<float> weights(Shape({(size_t)dim_0, (size_t)dim_1, (size_t)dim_2, (size_t)dim_3}));
    std::copy(data.begin(), data.end(), weights.begin());

    // conv_layer.parameter<Tensor<float>>("w").set(weights);
    conv_layer.set_weights(weights);
    std::cout << "[INFO] Loaded convolution weights from " << json_path << std::endl;

}

void save_tensor_as_image(const Tensor<float>& tensor, const std::string& path) {
      const Shape& shape = tensor.shape();
      std::cout << "[DEBUG] Saving tensor with shape: ";
  
    if (const_cast<Tensor<float>&>(tensor).shape().number() < 2) {
        std::cerr << "[WARN] Tensor has less than 2 dimensions. Skipping save to: " << path << std::endl;
        return;
    }

    int height = static_cast<int>(tensor.shape().dim(0));
    int width = static_cast<int>(tensor.shape().dim(1));
    cv::Mat out_img(height, width, CV_32F);

    for (int r = 0; r < height; ++r) {
        for (int c = 0; c < width; ++c) {
            out_img.at<float>(r, c) = tensor.at(r, c);
        }
    }

    cv::Mat normalized_img;
    cv::normalize(out_img, normalized_img, 0, 255, cv::NORM_MINMAX);
    normalized_img.convertTo(normalized_img, CV_8U);
    cv::imwrite(path, normalized_img);
}

int main(int argc, char** argv) {
    const char* input_path_ptr = std::getenv("INPUT_PATH");
    if (!input_path_ptr)
        throw std::runtime_error("INPUT_PATH not set");
    std::string input_path(input_path_ptr);
    std::string model_dir = input_path + "/model/layer.Convolution.fc1";

    // === Setup processing pipeline ===
    process::DefaultOnOffFilter onoff(7, 1.0, 4.0);
    process::MaxScaling scale;
    LatencyCoding code;
    layer::Convolution conv(5, 5, 16);
    conv.set_name("fc1");
    conv.parameter<bool>("inhibition").set(true);
    conv.parameter<bool>("wta_infer").set(false);

    process::SumPooling sum_pool(20, 20);
    process::FeatureScaling scaling;

    SVM_Predictor svm(model_dir + "/model.svm");
    if (!svm.isLoaded())
        throw std::runtime_error("Failed to load SVM model");

    std::string infer_dir = input_path + "/data/infer";
    std::vector<std::string> image_paths;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(infer_dir)) {
        if (entry.path().extension() == ".png") {
            image_paths.push_back(entry.path().string());
        }
    }
    if (image_paths.empty())
        throw std::runtime_error("No .png images found in: " + infer_dir);

    for (const auto& image_path : image_paths) {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (img.empty()) {
            std::cerr << "Failed to load image: " << image_path << std::endl;
            continue;
        }

        Tensor<float> tensor(Shape({static_cast<size_t>(img.rows), static_cast<size_t>(img.cols)}));
        for (int r = 0; r < img.rows; ++r)
            for (int c = 0; c < img.cols; ++c)
                tensor.at(r, c) = static_cast<float>(img.at<uchar>(r, c)) / 255.0f;
        // Tensor<float> tensor(Shape({img.rows, img.cols, 3}));
        // for (int r = 0; r < img.rows; ++r) {
        //     for (int c = 0; c < img.cols; ++c) {
        //         float val = static_cast<float>(img.at<uchar>(r, c)) / 255.0f;
        //         for (int ch = 0; ch < 3; ++ch) {
        //             tensor.at(r, c, ch) = val;
        //         }
        //     }
        // }


        std::string label = std::filesystem::path(image_path).parent_path().filename().string();

        onoff.compute_shape(tensor.shape());
        onoff.process_test(label, tensor);
        // std::cout << "onoff.process_test" <<  tensor.shape().to_string() << std::endl;
        scale.process_test(label, tensor);
        // std::cout << "scale.process_test" <<  tensor.shape().to_string() << std::endl;
        Tensor<Time> latency_tensor(tensor.shape());
        code.process(tensor, latency_tensor);
        Tensor<float> reconverted_tensor(latency_tensor.shape());
        for (size_t i = 0; i < latency_tensor.shape().product(); ++i) {
            float val = static_cast<float>(latency_tensor.at_index(i));
            reconverted_tensor.at_index(i) = std::clamp(val, 0.0f, 1.0f);
        }
        // >>> COMPUTE SHAPE BEFORE LOADING WEIGHTS <<<
        // std::cout << "reconverted tensor shape: " <<  reconverted_tensor.shape().to_string() << std::endl;
        conv.compute_shape(reconverted_tensor.shape());
        // std::cout << "[INFO CONV]Conv output shape: " << conv.shape().to_string() << std::endl;

        load_convolution_weights_from_json(conv, model_dir + "/weights.json");


        Tensor<float> thresholds(Shape({conv.depth()}));
        std::fill(thresholds.begin(), thresholds.end(), 10.0f);
        conv.set_thresholds(thresholds);
        std::cout << "thresholds set shape: " << reconverted_tensor.shape().to_string() << std::endl;
        conv.process_test_sample_inference(label, reconverted_tensor, 0, image_paths.size());
        std::cout << "After conv.process_test_sample " <<  reconverted_tensor.shape().to_string() << std::endl;
        // Continue with feature processing
        sum_pool.process_test(label, reconverted_tensor);
        std::cout << "After sum_pool.process_test" <<  reconverted_tensor.shape().to_string() << std::endl;

        scaling.process_test(label, reconverted_tensor);
        std::cout << "after scaling.process_test " <<  reconverted_tensor.shape().to_string() << std::endl;


        std::vector<float> flat_feature;
        flat_feature.reserve(reconverted_tensor.shape().product());
        for (auto it = reconverted_tensor.begin(); it != reconverted_tensor.end(); ++it)
            flat_feature.push_back(*it);

        std::cout << "reconverted tensor shape: sum, scaling AFTER " <<  reconverted_tensor.shape().to_string() << std::endl;
        std::cout << "flat_feature size: " << flat_feature.size() << std::endl;


        double prediction = svm.predict(flat_feature);
        std::cout << "Predicted: " << prediction << " for image: " << image_path << std::endl;
    }
    return 0;
}



// int main(int argc, char** argv) {
//     const char* input_path_ptr = std::getenv("INPUT_PATH");
//     if (!input_path_ptr)
//         throw std::runtime_error("INPUT_PATH not set");
//     std::string input_path(input_path_ptr);

//     std::string model_dir = input_path + "/model/layer.Convolution.fc1";

//     // === SNN processing pipeline ===
//     process::DefaultOnOffFilter onoff(7, 1.0, 4.0);
//     process::MaxScaling scale;
//     LatencyCoding code;
//     layer::Convolution conv(5, 5, 16);
//     conv.set_name("fc1");
//     conv.parameter<bool>("inhibition").set(true);
//     conv.parameter<bool>("wta_infer").set(false);

//     load_convolution_weights_from_json(conv, model_dir + "/weights.json");

//     Tensor<float> thresholds(Shape({16}));
//     std::fill(thresholds.begin(), thresholds.end(), 10.0f);
//     conv.parameter<Tensor<float>>("th").set(thresholds);
//     process::SumPooling sum_pool(20, 20);
//     process::FeatureScaling scaling;

//     SVM_Predictor svm(model_dir + "/model.svm");
//     if (!svm.isLoaded()) {
//         throw std::runtime_error("Failed to load SVM model using SVM_Predictor");
//     }

//     std::string infer_dir = input_path + "/data/infer";
//     std::vector<std::string> image_paths;

//     for (const auto& entry : std::filesystem::recursive_directory_iterator(infer_dir)) {
//         if (entry.path().extension() == ".png") {
//             image_paths.push_back(entry.path().string());
//         }
//     }

//     if (image_paths.empty()) {
//         throw std::runtime_error("No .png images found in: " + infer_dir);
//     }

//     for (const auto& image_path : image_paths) {
//         cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
//         if (img.empty()) {
//             std::cerr << "Failed to load image: " << image_path << std::endl;
//             continue;
//         }

//         Tensor<float> tensor(Shape({static_cast<size_t>(img.rows), static_cast<size_t>(img.cols)}));
//         for (int r = 0; r < img.rows; ++r) {
//             for (int c = 0; c < img.cols; ++c) {
//                 tensor.at(r, c) = static_cast<float>(img.at<uchar>(r, c)) / 255.0f;
//             }
//         }
//         std::cout << "Tensor shape: " << tensor.shape().to_string() << std::endl;


//         std::string label = std::filesystem::path(image_path).parent_path().filename().string();
//         std::cout << "Label: " << label << std::endl;
//         onoff.compute_shape(tensor.shape());
//         onoff.process_test(label, tensor);
//         std::cout << "Tensor shape: " << tensor.shape().to_string() << std::endl;

//         scale.process_test(label, tensor);
//         // save_tensor_as_image(tensor, "debug_outputs/scaled.png");
//         std::cout << "Tensor shape: " << tensor.shape().to_string() << std::endl;
//         std::cout << "Tensor shape: code" << std::endl;
//         Tensor<Time> output_tensor(tensor.shape());
//         code.process(tensor, output_tensor);
//         Tensor<float> reconverted_tensor(output_tensor.shape());
//         for (size_t i = 0; i < output_tensor.shape().product(); ++i) {
//             reconverted_tensor.at_index(i) = static_cast<float>(output_tensor.at_index(i));
//         }
//         for (size_t i = 0; i < output_tensor.shape().product(); ++i) {
//             float v = static_cast<float>(output_tensor.at_index(i));
//             v = std::clamp(v, 0.0f, 1.0f); // clamp to avoid weird values
//             reconverted_tensor.at_index(i) = v;
//         }

//         conv.compute_shape(reconverted_tensor.shape());
//         conv.process_test_sample(label, reconverted_tensor, 0, image_paths.size());
//         std::cout << "Tensor shape: conv" << std::endl;
//         sum_pool.process_test(label, tensor);
//         std::cout << "Tensor shape:sum" << std::endl;

//         scaling.process_test(label, tensor);
//         std::cout << "Tensor shape: scaling" << std::endl;
//         std::vector<float> flat_feature;
//         flat_feature.reserve(tensor.shape().product());
//         for (auto it = tensor.begin(); it != tensor.end(); ++it) {
//             flat_feature.push_back(*it);
//         }
//         std::cout << "Tensor shape: end" << std::endl;

//         double prediction = svm.predict(flat_feature);
//         std::cout << "Predicted: " << prediction << " for image: " << image_path << std::endl;
//     }

//     return 0;
// }
