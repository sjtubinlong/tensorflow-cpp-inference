
#include <fstream>
#include <utility>
#include <vector>
#include <ctime>
#include <chrono>
#include <set>

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/image_ops.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/stringprintf.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/init_main.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/util/command_line_flags.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv/cv.hpp>

#include <dirent.h>
#include <sys/types.h>

using tensorflow::Flag;
using tensorflow::int32;
using tensorflow::Status;
using tensorflow::string;
using tensorflow::Tensor;
using tensorflow::tstring;


const int g_channels = 3;
const float g_mean = 0.5;
const float g_std = 1.0;
const int g_height = 1025;
const int g_width = 2049;
const int g_num_class = 19; 
const int g_batch = 2;

Status loadGraph(const string &graph_file_name,
                 std::unique_ptr<tensorflow::Session> *session) {
    tensorflow::GraphDef graph_def;
    Status load_graph_status =
            ReadBinaryProto(tensorflow::Env::Default(), graph_file_name, &graph_def);
    if (!load_graph_status.ok()) {
        return tensorflow::errors::NotFound("Failed to load compute graph at '",
                                            graph_file_name, "'");
    }
    session->reset(tensorflow::NewSession(tensorflow::SessionOptions()));
    Status session_create_status = (*session)->Create(graph_def);
    if (!session_create_status.ok()) {
        return session_create_status;
    }
    return Status::OK();
}

Status single_preprocess(std::string& img, std::vector<Tensor>& out_tensors)
{
    auto root = tensorflow::Scope::NewRootScope();
    using namespace ::tensorflow::ops;

    string input_name = "file_reader";
    string output_name = "normalized";

    auto file_reader = tensorflow::ops::ReadFile(root.WithOpName(input_name), img);
    auto image_reader = DecodePng(root.WithOpName("png_reader"), file_reader, DecodePng::Channels(g_channels));
    auto float_caster = Cast(root.WithOpName("float_caster"), image_reader, tensorflow::DT_FLOAT);
    auto dims_expander = ExpandDims(root, float_caster, 0);
    auto resized = ResizeBilinear(
            root, dims_expander,
            Const(root.WithOpName("size"), {g_height, g_width}));
    auto div_out = Div(root.WithOpName(output_name), Sub(root, resized, {g_mean}), {g_std});
    Cast(root.WithOpName("uint8_caster"), div_out, tensorflow::DT_UINT8);

    tensorflow::GraphDef graph;
    TF_RETURN_IF_ERROR(root.ToGraphDef(&graph));

    std::unique_ptr<tensorflow::Session> session(
      tensorflow::NewSession(tensorflow::SessionOptions()));
    TF_RETURN_IF_ERROR(session->Create(graph));
    TF_RETURN_IF_ERROR(session->Run({}, {"uint8_caster"}, {}, &out_tensors));
    return Status::OK();
}

Status batch_preprocess(const std::vector<std::string>& imgs, std::vector<Tensor>& out_tensors)
{
    if (imgs.size() <= 0) {
        return Status::OK();   
    }
    auto real_batch = std::min(g_batch, (int)imgs.size());
    tensorflow::Tensor input_tensor(
            tensorflow::DT_UINT8,
            tensorflow::TensorShape({real_batch, g_height, g_width, g_channels}));
    auto input_data = input_tensor.flat<tensorflow::uint8>().data();
	for (int i = 0; i < real_batch; ++i)
	{
        auto& item = imgs[i];
        cv::Mat im = cv::imread(item, -1);
        cv::resize(im, im, cv::Size(g_width, g_height), 0, 0, cv::INTER_LINEAR);
        cv::cvtColor(im, im, CV_BGR2RGB);
        im = (im - g_mean) / g_std;

        auto ptr = input_data + i * g_height * g_width * g_channels;
        cv::Mat fake_im(g_height, g_width, CV_8UC3, ptr);
        im.convertTo(fake_im, CV_8UC3);
	}
    out_tensors.push_back(input_tensor);
    return Status::OK();
}

std::vector<std::string> get_directory_images(const std::string& path, const std::string& exts)
{
    std::vector<std::string> imgs;
    struct dirent *entry;
    DIR *dir = opendir(path.c_str());
    if (dir == NULL) {
        closedir(dir);
        return imgs;
    }
    while ((entry = readdir(dir)) != NULL) {
        std::string item = entry->d_name;
        auto ext = strrchr(entry->d_name, '.');
        if (!ext || std::string(ext) == "." || std::string(ext) == "..") {
            continue;
        }
        if (exts.find(ext) != std::string::npos) {
            imgs.push_back(tensorflow::io::JoinPath(path, entry->d_name));
            std::cout << imgs.back() << std::endl;
        }
    }
    return imgs;
}
 
int main(int argc, char* argv[]) {
    std::string MODEL_ROOT_DIR = "/root/projects/tfmodels/";
    std::string MODEL_GRAPH_PATH = "deeplabv3_cityscapes_train/frozen_inference_graph.pb";
    std::string DATA_INPUT_DIR = "/root/projects/tfdataset/cityscape";

    std::string input_layer = "ImageTensor:0";
    std::string output_layer = "SemanticPredictions:0";

    std::unique_ptr<tensorflow::Session> session;
    string graph_path = tensorflow::io::JoinPath(MODEL_ROOT_DIR, MODEL_GRAPH_PATH);

    LOG(INFO) << "MODEL_GRAPH_PATH:[" << graph_path << "]";
    Status loadGraphStatus = loadGraph(graph_path, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return -1;
    } else {
        LOG(INFO) << "loadGraph(): frozen graph loaded";
    }

    std::vector<Tensor> inputs;
    auto imgs = get_directory_images(DATA_INPUT_DIR, ".png");

    //Status input_status = single_preprocess(imgs.back(), inputs);
    Status input_status = batch_preprocess(imgs, inputs);
    if (!input_status.ok()) {
        LOG(ERROR) << "Single Preprocess failed: " << input_status;
        return -1;
    }
    LOG(ERROR) << "inputs.size=" << inputs.size() << inputs[0].DebugString();

    std::vector<Tensor> outputs;
    auto t1 = std::chrono::high_resolution_clock::now();
    Status run_status = session->Run({{input_layer, inputs[0]}}, {output_layer}, {}, &outputs);
    if (!run_status.ok()) {
        LOG(ERROR) << "Running model failed: " << run_status;
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();
    std::cout << "runtime = " << duration << std::endl;

    LOG(INFO) << outputs[0].DebugString();
    auto out_raw = outputs[0].flat<tensorflow::int64>();
    auto shape = outputs[0].shape().dim_sizes();
    
    int out_len = g_height * g_width;
    std::vector<uchar> mask(out_len, 0);
    printf("img=%s\n", imgs.back().c_str());
    for (int i = 0; i < shape[1]; ++i) {
        for (int j = 0; j < shape[2]; ++j) {
            mask[i * shape[2] + j] = uchar(out_raw(i * shape[2] + j));
        }
    }

    cv::Mat mask_png = cv::Mat(g_height, g_width, CV_8UC1);
    mask_png.data = mask.data();
    std::string mask_save_name = "mask.png";
    cv::imwrite(mask_save_name, mask_png);

    return 0;
}
