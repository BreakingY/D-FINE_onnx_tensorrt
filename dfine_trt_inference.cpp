#if 0
// old API
// test version TensorRT-8.5.1.7
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <cstring>
#include <tuple>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include "NvInfer.h"
#define PROC_GPU
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif 
struct Detection {
    cv::Rect2f box;   // x, y, w, h  (左上角坐标 + 宽高)
    float score;      // 置信度
    int class_id;     // 类别索引
};
const char *class_names[2] = {"dog", "person"};
class Logger: public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if(severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

char* ReadFromPath(std::string eng_path,int &model_size){
    std::ifstream file(eng_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << eng_path << " error!" << std::endl;
        return nullptr;
    }
    char *trt_model_stream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    if(!trt_model_stream){
        return nullptr;
    }
    file.read(trt_model_stream, size);
    file.close();
    model_size = size;
    return trt_model_stream;
}
// 返回值: tuple<cv::Mat, float, float, float> -> (resized_img, scale_ratio, dw, dh)
std::tuple<cv::Mat, float, float, float> Letterbox_resize(const cv::Mat& img,int new_h, int new_w,const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    int orig_h = img.rows;
    int orig_w = img.cols;

    float r = std::min(static_cast<float>(new_h) / orig_h, static_cast<float>(new_w) / orig_w);

    int new_unpad_w = static_cast<int>(std::round(orig_w * r));
    int new_unpad_h = static_cast<int>(std::round(orig_h * r));

    float dw = new_w - new_unpad_w;
    float dh = new_h - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return std::move(std::make_tuple(out, r, dw, dh));
}
std::tuple<float, float, float>  PreprocessImage(std::string path, void *buffer, int channel, int input_h, int input_w, cudaStream_t stream){
    cv::Mat img = cv::imread(path);
    std::tuple<cv::Mat, float, float, float> res = Letterbox_resize(img, input_h, input_w);
    cv::Mat &res_mat = std::get<0>(res);
    float &r = std::get<1>(res);
    float &dw = std::get<2>(res);
    float &dh = std::get<3>(res);
    // cv::imwrite("output.jpg", res_mat);
    cv::Mat img_float;
    cv::cvtColor(res_mat, res_mat, cv::COLOR_BGR2RGB); // 如果颜色通道顺序不对，模型检测精度会下降很多
    res_mat.convertTo(img_float, CV_32FC3, 1.f / 255.0);

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(channel);
    cv::split(img_float, input_channels); // cv::split 把多通道图像拆分成单通道图像。RRRR GGGG BBBB

    std::vector<float> result(input_h * input_w * channel);
    auto data = result.data();
    int channel_length = input_h * input_w;
    for (int i = 0; i < channel; ++i) {
        memcpy(data, input_channels[i].data, channel_length * sizeof(float));
        data += channel_length;
    }

    CUDA_CHECK(cudaMemcpyAsync(buffer, (void *)result.data(), input_h * input_w * channel * sizeof(float), cudaMemcpyHostToDevice, stream));
    return std::move(std::make_tuple(r, dw, dh));
}
// 返回值: tuple<cv::Mat, float, float, float> -> (resized_img, scale_ratio, dw, dh)
std::tuple<float, float, float> Letterbox_resize_GPU(int orig_h, int orig_w, void *img_buffer, void *out_buffer,int new_h, int new_w, const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    float r = std::min(static_cast<float>(new_h) / orig_h, static_cast<float>(new_w) / orig_w);
    int new_unpad_w = static_cast<int>(std::round(orig_w * r));
    int new_unpad_h = static_cast<int>(std::round(orig_h * r));
    float dw = new_w - new_unpad_w;
    float dh = new_h - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    Npp8u *pu8_src = static_cast<Npp8u*>(img_buffer);
    Npp8u *pu8_dst = static_cast<Npp8u*>(out_buffer);

    Npp8u color_array[3] = {(Npp8u)color[0], (Npp8u)color[1], (Npp8u)color[2]};
    NppiSize dst_size{new_w, new_h};
    NppStatus ret = nppiSet_8u_C3R(color_array, pu8_dst, new_w * 3, dst_size);
    if(ret != 0){
        std::cerr << "nppiSet_8u_C3R error: " << ret << std::endl;
        return std::make_tuple(r, dw, dh);
    }
    Npp8u *pu8_resized = nullptr;
    CUDA_CHECK(cudaMalloc(&pu8_resized, new_unpad_h * new_unpad_w * 3));

    NppiSize src_size{orig_w, orig_h};
    NppiRect src_roi{0,0,orig_w,orig_h};
    NppiSize resize_size{new_unpad_w, new_unpad_h};
    NppiRect dst_roi{0,0,new_unpad_w,new_unpad_h};

    ret = nppiResize_8u_C3R(pu8_src, orig_w * 3, src_size, src_roi, pu8_resized, new_unpad_w * 3, resize_size, dst_roi, NPPI_INTER_LINEAR);
    if(ret != 0){
        std::cerr << "nppiResize_8u_C3R error: " << ret << std::endl;
        CUDA_CHECK(cudaFree(pu8_resized));
        return std::make_tuple(r, dw, dh);
    }
    NppiSize copy_size{new_unpad_w, new_unpad_h};
    ret = nppiCopy_8u_C3R(pu8_resized, new_unpad_w * 3, pu8_dst + top * new_w * 3 + left * 3, new_w * 3, copy_size);
    if(ret != 0){
        std::cerr << "nppiCopy_8u_C3R error: " << ret << std::endl;
    }

    CUDA_CHECK(cudaFree(pu8_resized));
#if 0
    cv::Mat img_cpu(new_h, new_w, CV_8UC3);
    
    size_t bytes = new_w * new_h * 3 * sizeof(Npp8u);
    CUDA_CHECK(cudaMemcpy(img_cpu.data, out_buffer, bytes, cudaMemcpyDeviceToHost));
    if(!cv::imwrite("output.jpg", img_cpu)){
        std::cerr << "Failed to save image"  << std::endl;
    } 
#endif
    return std::make_tuple(r, dw, dh);
}
std::tuple<float, float, float>  PreprocessImage_GPU(std::string path, void *buffer, int channel, int input_h, int input_w, cudaStream_t stream){
    cv::Mat img = cv::imread(path);
    void *img_buffer = nullptr;
    int orig_h = img.rows;
    int orig_w = img.cols;
    CUDA_CHECK(cudaMalloc(&img_buffer, orig_h * orig_w * 3));
    void *img_ptr = img.data;
    CUDA_CHECK(cudaMemcpyAsync(img_buffer, img_ptr, orig_h * orig_w * 3, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::tuple<float, float, float> res = Letterbox_resize_GPU(orig_h, orig_w, img_buffer, buffer, input_h, input_w);

    float &r = std::get<0>(res);
    float &dw = std::get<1>(res);
    float &dh = std::get<2>(res);
    
    Npp8u *pu8_rgb = nullptr;
    CUDA_CHECK(cudaMalloc(&pu8_rgb, input_h * input_w * 3));
    // BGR-->RGB
    int aOrder[3] = {2, 1, 0};
    NppiSize size = {input_w, input_h};
    NppStatus ret = nppiSwapChannels_8u_C3R((Npp8u*)buffer, input_w * 3, pu8_rgb, input_w * 3, size, aOrder);
    if(ret != 0){
        std::cerr << "nppiSwapChannels_8u_C3R error: " << ret << std::endl;
    }

    // 转 float 并归一化
    NppiSize fsize = {input_w, input_h};
    ret = nppiConvert_8u32f_C3R(pu8_rgb, input_w * 3, (Npp32f*)buffer, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        std::cerr << "nppiConvert_8u32f_C3R error: " << ret << std::endl;
    }
    Npp32f aConstants[3] = {1.f / 255.f, 1.f / 255.f,1.f / 255.f};
    ret = nppiMulC_32f_C3IR(aConstants, (Npp32f*)buffer, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        std::cerr << "nppiMulC_32f_C3IR error: " << ret << std::endl;
    }

    // HWC TO CHW
    NppiSize chw_size = {input_w, input_h};
    float* buffer_chw = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer_chw, input_h * input_w * 3 * sizeof(float)));
    Npp32f* dst_planes[3];
    dst_planes[0] = (Npp32f*)buffer_chw;                           // R
    dst_planes[1] = (Npp32f*)buffer_chw + input_h * input_w;       // G
    dst_planes[2] = (Npp32f*)buffer_chw + input_h * input_w * 2;   // B
    ret = nppiCopy_32f_C3P3R((Npp32f*)buffer, input_w * 3 * sizeof(float), dst_planes, input_w * sizeof(float), chw_size);
    if (ret != 0) {
        std::cerr << "nppiCopy_32f_C3P3R error: " << ret << std::endl;
    }
    CUDA_CHECK(cudaMemcpy(buffer, buffer_chw, input_h * input_w * 3 * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(buffer_chw));
    CUDA_CHECK(cudaFree(img_buffer));
    CUDA_CHECK(cudaFree(pu8_rgb));
    return std::move(std::make_tuple(r, dw, dh));
}
int Inference(nvinfer1::IExecutionContext* context, void**buffers, void* output_labels, void* output_boxes, void* output_scores, int label_output_len, int boxes_output_len, int scores_output_len,
              const int batch_size, int channel, int input_h, int input_w, int input_index_images, int input_index_size, int output_index_score, int output_index_label, int output_index_boxes, cudaStream_t stream){
    context->setBindingDimensions(0, nvinfer1::Dims4(batch_size, channel, input_h, input_w));
    context->setBindingDimensions(1, nvinfer1::Dims2(1, 2));
    if(!context->enqueueV2(buffers, stream, nullptr)) {
        std::cerr << "enqueueV2 failed!" << std::endl;
        return -2;
    }
    CUDA_CHECK(cudaMemcpyAsync(output_scores, buffers[output_index_score], batch_size * scores_output_len * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output_labels, buffers[output_index_label], batch_size * label_output_len * sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(output_boxes, buffers[output_index_boxes], batch_size * boxes_output_len * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
static float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}

static std::vector<int> NMS(const std::vector<Detection>& dets, float iou_thres) {
    std::vector<int> order(dets.size());
    for (size_t idx = 0; idx < order.size(); ++idx) {
        order[idx] = static_cast<int>(idx);
    }
    std::sort(order.begin(), order.end(), [&](int i, int j){
        return dets[i].score > dets[j].score;
    });

    std::vector<int> keep;
    std::vector<char> removed(dets.size(), 0);
    for (size_t _i = 0; _i < order.size(); ++_i) {
        int i = order[_i];
        if (removed[i]) continue;
        keep.push_back(i);
        for (size_t _j = _i + 1; _j < order.size(); ++_j) {
            int j = order[_j];
            if (removed[j]) continue;
            if (dets[i].class_id != dets[j].class_id) continue; // 不同类不互相抑制（常见策略）
            if (IoU(dets[i].box, dets[j].box) > iou_thres) {
                removed[j] = 1;
            }
        }
    }
    return keep;
}
static std::vector<Detection> PostprocessDetections(
    const int32_t * feat_labels, const float* feat_boxes, const float* feat_scores,
    int output_num,
    float r, float dw, float dh,    // 反 letterbox 参数
    int orig_w, int orig_h,         // 原图大小
    float conf_thres = 0.5f,
    float iou_thres  = 0.5f)
{
    int num_classes = (int)(sizeof(class_names)/sizeof(class_names[0]));
    std::vector<Detection> dets;
    dets.reserve(512);
    int walk = 4;
    // for(int i = 0; i < 300*4;i++){
    //     std::cout << feat_boxes[i];
    // }
    // printf("\n");
    // for(int i = 0; i < 300;i++){
    //     std::cout << feat_labels[i];
    // }
    // printf("\n");
    // for(int i = 0; i < 300;i++){
    //     std::cout << feat_scores[i];
    // }
    // printf("\n");
    for(int i = 0; i < output_num; i++){
        int label = static_cast<int>(feat_labels[i]);
        float scores = feat_scores[i];
        float x1 = feat_boxes[i * walk];
        float y1 = feat_boxes[i * walk + 1];
        float x2 = feat_boxes[i * walk + 2];
        float y2 = feat_boxes[i * walk + 3];
        // std::cout << "label:" << label << " scores:" << scores << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << std::endl;
        if(scores < conf_thres){
            continue;
        }
        float w  = x2 - x1;
        float h  = y2 - y1;
        float cx = x1 + w / 2;
        float cy = y1 + h / 2;

        float x = (cx - w * 0.5f - dw) / r;
        float y = (cy - h * 0.5f - dh) / r;
        float ww = w / r;
        float hh = h / r;

        x  = std::max(0.f, std::min(x,  (float)orig_w  - 1.f));
        y  = std::max(0.f, std::min(y,  (float)orig_h - 1.f));
        ww = std::max(0.f, std::min(ww, (float)orig_w  - x));
        hh = std::max(0.f, std::min(hh, (float)orig_h - y));

        if (ww <= 0.f || hh <= 0.f) continue;

        Detection d;
        d.box = cv::Rect2f(x, y, ww, hh);
        d.score = scores;
        d.class_id = label;
        dets.push_back(d);
    }
    // NMS
    // std::vector<int> keep = NMS(dets, iou_thres);
    // std::vector<Detection> out;
    // out.reserve(keep.size());
    // for (int idx : keep) out.push_back(dets[idx]);
    // return out;
    return dets;
}
int main(int argc, char **argv){
    if(argc < 3){
        std::cerr << "./bin eng_path test.jpg" << std::endl;
        return 0;
    }
    const char *eng_path = argv[1];
    const char *img_path = argv[2];
    int device_id = 0;
    cudaStream_t stream;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream));
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	assert(runtime != nullptr);
    int model_size = 0;
    char *trt_model_stream = ReadFromPath(eng_path,model_size);
    assert(trt_model_stream != nullptr);

	nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(trt_model_stream, model_size);
	assert(engine != nullptr);

    nvinfer1::IExecutionContext* context = engine->createExecutionContext();
	assert(context != nullptr);
    delete []trt_model_stream;

    int num_bindings = engine->getNbBindings();
	std::cout << "input/output : " << num_bindings << std::endl;
	std::vector<const char*> input_names;
	std::vector<const char*> output_names;
	for (int i = 0; i < num_bindings; ++i) {
		const char* binding_name = engine->getBindingName(i);
		if (engine->bindingIsInput(i)) {
			input_names.push_back(binding_name);
		}
		else {
			output_names.push_back(binding_name);
		}
	}
    for(int i = 0; i < input_names.size(); i++){
        std::cout << "input " << i << ":" << input_names[i] << std::endl;
    }
    for(int i = 0; i < output_names.size(); i++){
        std::cout << "output " << i << ":" << output_names[i] << std::endl;
    }
    
    int input_index_images = engine->getBindingIndex("images");
    int input_index_size = engine->getBindingIndex("orig_target_sizes");
    
	int output_index_score = engine->getBindingIndex("scores"); // n*300
    int output_index_label = engine->getBindingIndex("labels"); // n*300
    int output_index_boxes = engine->getBindingIndex("boxes"); // n*300*4
    std::cout << "input_index_images:" << input_index_images << " input_index_size:" << input_index_size << " output_index_score:" << output_index_score << 
                " output_index_label:" << output_index_label << "output_index_boxes:" << output_index_boxes << std::endl;
    // int batch_size = engine->getBindingDimensions(input_index).d[0]; // 动态维度返回-1
    int batch_size = 4; // trtexex转模型设置的最大batch
    int channel = engine->getBindingDimensions(input_index_images).d[1];
    assert(channel == 3);
    int input_h = engine->getBindingDimensions(input_index_images).d[2];
	int input_w = engine->getBindingDimensions(input_index_images).d[3];
    std::cout << "batch_size:" << batch_size << " channel:" << channel << " input_h:" << input_h << " input_w:" << input_w << std::endl;

    // int batch_size = engine->getBindingDimensions(output_index_0).d[0];
    int score_num = engine->getBindingDimensions(output_index_score).d[1];
    int label_num = engine->getBindingDimensions(output_index_label).d[1];
    int boxes_num = engine->getBindingDimensions(output_index_boxes).d[1];
    int boxes_loc_num = engine->getBindingDimensions(output_index_boxes).d[2];
    std::cout << "label_num:" << label_num << std::endl;
    std::cout << "boxes_num:" << boxes_num << " boxes_loc_num:" << boxes_loc_num << std::endl;
    std::cout << "score_num:" << score_num << std::endl;

    nvinfer1::DataType images_type = engine->getBindingDataType(input_index_images);
    if (images_type == nvinfer1::DataType::kINT32) {
        std::cout << "images 类型为 int32" << std::endl;
    } 
    // 8.5.1.7 have not kINT64
    // else if (images_type == nvinfer1::DataType::kINT64) {
    //     std::cout << "images 类型为 int64" << std::endl;
    // } 
    else if (images_type == nvinfer1::DataType::kFLOAT) {
        std::cout << "images 类型为 float" << std::endl;
    }

    nvinfer1::DataType size_type = engine->getBindingDataType(input_index_size);
    if (size_type == nvinfer1::DataType::kINT32) {
        std::cout << "orig_target_sizes 类型为 int32" << std::endl;
    } 
    // 8.5.1.7 have not kINT64
    // else if (size_type == nvinfer1::DataType::kINT64) {
    //     std::cout << "orig_target_sizes 类型为 int64" << std::endl;
    // } 
    else if (size_type == nvinfer1::DataType::kFLOAT) {
        std::cout << "orig_target_sizes 类型为 float" << std::endl;
    }

    nvinfer1::DataType label_type = engine->getBindingDataType(output_index_label);
    if (label_type == nvinfer1::DataType::kINT32) {
        std::cout << "labels 类型为 int32" << std::endl;
    } 
    // 8.5.1.7 have not kINT64
    // else if (label_type == nvinfer1::DataType::kINT64) {
    //     std::cout << "labels 类型为 int64" << std::endl;
    // } 
    else if (label_type == nvinfer1::DataType::kFLOAT) {
        std::cout << "labels 类型为 float" << std::endl;
    }

    nvinfer1::DataType box_type = engine->getBindingDataType(output_index_boxes);
    if (box_type == nvinfer1::DataType::kINT32) {
        std::cout << "boxes 类型为 int32" << std::endl;
    } 
    // 8.5.1.7 have not kINT64
    // else if (box_type == nvinfer1::DataType::kINT64) {
    //     std::cout << "boxes 类型为 int64" << std::endl;
    // } 
    else if (box_type == nvinfer1::DataType::kFLOAT) {
        std::cout << "boxes 类型为 float" << std::endl;
    }

    nvinfer1::DataType score_type = engine->getBindingDataType(output_index_score);
    if (score_type == nvinfer1::DataType::kINT32) {
        std::cout << "scores 类型为 int32" << std::endl;
    } 
    // 8.5.1.7 have not kINT64
    // else if (score_type == nvinfer1::DataType::kINT64) {
    //     std::cout << "scores 类型为 int64" << std::endl;
    // } 
    else if (score_type == nvinfer1::DataType::kFLOAT) {
        std::cout << "scores 类型为 float" << std::endl;
    }

    void* buffers[5] = {NULL, NULL, NULL, NULL, NULL};
    // images
    CUDA_CHECK(cudaMalloc(&buffers[input_index_images], batch_size * input_h * input_w * 3 * sizeof(float)));
    // orig_target_sizes
    CUDA_CHECK(cudaMalloc(&buffers[input_index_size], batch_size * 2 * sizeof(int32_t)));
    // scores n*300
    int scores_output_len = score_num;
	CUDA_CHECK(cudaMalloc(&buffers[output_index_score], batch_size * scores_output_len * sizeof(float)));
    // labels n*300
    int label_output_len = label_num;
	CUDA_CHECK(cudaMalloc(&buffers[output_index_label], batch_size * label_output_len * sizeof(int32_t)));
    // boxes n*300*4
    int boxes_output_len = boxes_num * boxes_loc_num;
	CUDA_CHECK(cudaMalloc(&buffers[output_index_boxes], batch_size * boxes_output_len * sizeof(float)));

    int32_t* output_labels = new int32_t[batch_size * label_output_len];
    float* output_boxes = new float[batch_size * boxes_output_len];
    float* output_scores = new float[batch_size * scores_output_len];
    int test_batch = 2;
    std::vector<std::tuple<float, float, float>> res_pre;
    int buffer_idx = 0;
    char* input_ptr_images = static_cast<char*>(buffers[input_index_images]);
    char* input_ptr_orig_target_sizes = static_cast<char*>(buffers[input_index_size]);
    for(int i = 0; i < test_batch; i++){
#ifdef PROC_GPU
        std::tuple<float, float, float> res = PreprocessImage_GPU(img_path, input_ptr_images + buffer_idx, channel, input_h, input_w, stream);
#else
        std::tuple<float, float, float> res = PreprocessImage(img_path, input_ptr_images + buffer_idx, channel, input_h, input_w, stream);
#endif
        buffer_idx += input_h * input_w * 3 * sizeof(float);
        res_pre.push_back(res);
    }   
    int32_t img_size[2] = {(int32_t)input_h, (int32_t)input_w};
    CUDA_CHECK(cudaMemcpyAsync(input_ptr_orig_target_sizes, img_size, sizeof(int32_t) * 2, cudaMemcpyHostToDevice, stream)); 
    Inference(context, buffers, (void*)output_labels, (void*)output_boxes, (void*)output_scores, label_output_len, boxes_output_len, scores_output_len,
              res_pre.size(), channel, input_h, input_w, input_index_images, input_index_size, output_index_score, output_index_label, output_index_boxes, stream);
    cv::Mat original = cv::imread(img_path);
    int orig_h = original.rows, orig_w = original.cols;

    for (int b = 0; b < test_batch; ++b) {
        auto [r, dw, dh] = res_pre[b];
        int32_t* feat_labels = output_labels + b * label_output_len;
        float* feat_boxes = output_boxes + b * boxes_output_len;
        float* feat_scores = output_scores + b * scores_output_len;

        std::vector<Detection> dets = PostprocessDetections(
            feat_labels, feat_boxes, feat_scores, label_num, r, dw, dh, orig_w, orig_h, /*conf*/0.5f, /*iou*/0.5f);

        cv::Mat vis = original.clone();
        for (const auto& d : dets) {
            cv::rectangle(vis, d.box, cv::Scalar(0, 255, 0), 2);
            char text[128];
            const char* cname = (d.class_id >= 0 && d.class_id < (int)(sizeof(class_names)/sizeof(class_names[0])))
                                    ? class_names[d.class_id] : "cls";
            snprintf(text, sizeof(text), "%s: %.2f", cname, d.score);
            int baseline = 0;
            cv::Size tsize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(vis, cv::Rect(cv::Point((int)d.box.x, (int)d.box.y - tsize.height - 4),
                                        cv::Size(tsize.width + 4, tsize.height + 4)),
                        cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(vis, text, cv::Point((int)d.box.x + 2, (int)d.box.y - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        std::string save_name = std::string("result_") + std::to_string(b) + ".jpg";
        cv::imwrite(save_name, vis);
        std::cout << "Saved: " << save_name << "  dets=" << dets.size() << std::endl;
    }
    for(int i = 0; i < 5; i++){
        CUDA_CHECK(cudaFree(buffers[i]));
    }
    delete []output_labels;
    delete []output_boxes;
    delete []output_scores;
    CUDA_CHECK(cudaStreamDestroy(stream));
    context->destroy();
    engine->destroy();
    return 0;
}
#else
// new API
// test version TensorRT-10.4.0.26
#include <fstream>
#include <iostream>
#include <sstream>
#include <cassert>
#include <string>
#include <vector>
#include <cstring>
#include <tuple>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <npp.h>
#include "NvInfer.h"
#define PROC_GPU
#ifndef CUDA_CHECK
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
#endif 
struct Detection {
    cv::Rect2f box;   // x, y, w, h  (左上角坐标 + 宽高)
    float score;      // 置信度
    int class_id;     // 类别索引
};
const char *class_names[2] = {"dog", "person"};
class Logger: public nvinfer1::ILogger{
    void log(Severity severity, const char* msg) noexcept override{
        if(severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
}logger;

char* ReadFromPath(std::string eng_path,int &model_size){
    std::ifstream file(eng_path, std::ios::binary);
    if (!file.good()) {
        std::cerr << "read " << eng_path << " error!" << std::endl;
        return nullptr;
    }
    char *trt_model_stream = nullptr;
    size_t size = 0;
    file.seekg(0, file.end);
    size = file.tellg();
    file.seekg(0, file.beg);
    trt_model_stream = new char[size];
    if(!trt_model_stream){
        return nullptr;
    }
    file.read(trt_model_stream, size);
    file.close();
    model_size = size;
    return trt_model_stream;
}
// 返回值: tuple<cv::Mat, float, float, float> -> (resized_img, scale_ratio, dw, dh)
std::tuple<cv::Mat, float, float, float> Letterbox_resize(const cv::Mat& img,int new_h, int new_w,const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    int orig_h = img.rows;
    int orig_w = img.cols;

    float r = std::min(static_cast<float>(new_h) / orig_h, static_cast<float>(new_w) / orig_w);

    int new_unpad_w = static_cast<int>(std::round(orig_w * r));
    int new_unpad_h = static_cast<int>(std::round(orig_h * r));

    float dw = new_w - new_unpad_w;
    float dh = new_h - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;

    cv::Mat resized;
    cv::resize(img, resized, cv::Size(new_unpad_w, new_unpad_h), 0, 0, cv::INTER_LINEAR);

    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    cv::Mat out;
    cv::copyMakeBorder(resized, out, top, bottom, left, right, cv::BORDER_CONSTANT, color);

    return std::move(std::make_tuple(out, r, dw, dh));
}
std::tuple<float, float, float>  PreprocessImage(std::string path, void *buffer, int channel, int input_h, int input_w, cudaStream_t stream){
    cv::Mat img = cv::imread(path);
    std::tuple<cv::Mat, float, float, float> res = Letterbox_resize(img, input_h, input_w);
    cv::Mat &res_mat = std::get<0>(res);
    float &r = std::get<1>(res);
    float &dw = std::get<2>(res);
    float &dh = std::get<3>(res);
    // cv::imwrite("output.jpg", res_mat);
    cv::Mat img_float;
    cv::cvtColor(res_mat, res_mat, cv::COLOR_BGR2RGB); // 如果颜色通道顺序不对，模型检测精度会下降很多
    res_mat.convertTo(img_float, CV_32FC3, 1.f / 255.0);

    // HWC TO CHW
    std::vector<cv::Mat> input_channels(channel);
    cv::split(img_float, input_channels); // cv::split 把多通道图像拆分成单通道图像。RRRR GGGG BBBB

    std::vector<float> result(input_h * input_w * channel);
    auto data = result.data();
    int channel_length = input_h * input_w;
    for (int i = 0; i < channel; ++i) {
        memcpy(data, input_channels[i].data, channel_length * sizeof(float));
        data += channel_length;
    }

    CUDA_CHECK(cudaMemcpyAsync(buffer, (void *)result.data(), input_h * input_w * channel * sizeof(float), cudaMemcpyHostToDevice, stream));
    return std::move(std::make_tuple(r, dw, dh));
}
// 返回值: tuple<cv::Mat, float, float, float> -> (resized_img, scale_ratio, dw, dh)
std::tuple<float, float, float> Letterbox_resize_GPU(int orig_h, int orig_w, void *img_buffer, void *out_buffer,int new_h, int new_w, const cv::Scalar& color = cv::Scalar(114, 114, 114))
{
    float r = std::min(static_cast<float>(new_h) / orig_h, static_cast<float>(new_w) / orig_w);
    int new_unpad_w = static_cast<int>(std::round(orig_w * r));
    int new_unpad_h = static_cast<int>(std::round(orig_h * r));
    float dw = new_w - new_unpad_w;
    float dh = new_h - new_unpad_h;
    dw /= 2.0f;
    dh /= 2.0f;
    int top = static_cast<int>(std::round(dh - 0.1f));
    int bottom = static_cast<int>(std::round(dh + 0.1f));
    int left = static_cast<int>(std::round(dw - 0.1f));
    int right = static_cast<int>(std::round(dw + 0.1f));

    Npp8u *pu8_src = static_cast<Npp8u*>(img_buffer);
    Npp8u *pu8_dst = static_cast<Npp8u*>(out_buffer);

    Npp8u color_array[3] = {(Npp8u)color[0], (Npp8u)color[1], (Npp8u)color[2]};
    NppiSize dst_size{new_w, new_h};
    NppStatus ret = nppiSet_8u_C3R(color_array, pu8_dst, new_w * 3, dst_size);
    if(ret != 0){
        std::cerr << "nppiSet_8u_C3R error: " << ret << std::endl;
        return std::make_tuple(r, dw, dh);
    }
    Npp8u *pu8_resized = nullptr;
    CUDA_CHECK(cudaMalloc(&pu8_resized, new_unpad_h * new_unpad_w * 3));

    NppiSize src_size{orig_w, orig_h};
    NppiRect src_roi{0,0,orig_w,orig_h};
    NppiSize resize_size{new_unpad_w, new_unpad_h};
    NppiRect dst_roi{0,0,new_unpad_w,new_unpad_h};

    ret = nppiResize_8u_C3R(pu8_src, orig_w * 3, src_size, src_roi, pu8_resized, new_unpad_w * 3, resize_size, dst_roi, NPPI_INTER_LINEAR);
    if(ret != 0){
        std::cerr << "nppiResize_8u_C3R error: " << ret << std::endl;
        CUDA_CHECK(cudaFree(pu8_resized));
        return std::make_tuple(r, dw, dh);
    }
    NppiSize copy_size{new_unpad_w, new_unpad_h};
    ret = nppiCopy_8u_C3R(pu8_resized, new_unpad_w * 3, pu8_dst + top * new_w * 3 + left * 3, new_w * 3, copy_size);
    if(ret != 0){
        std::cerr << "nppiCopy_8u_C3R error: " << ret << std::endl;
    }

    CUDA_CHECK(cudaFree(pu8_resized));
#if 0
    cv::Mat img_cpu(new_h, new_w, CV_8UC3);
    
    size_t bytes = new_w * new_h * 3 * sizeof(Npp8u);
    CUDA_CHECK(cudaMemcpy(img_cpu.data, out_buffer, bytes, cudaMemcpyDeviceToHost));
    if(!cv::imwrite("output.jpg", img_cpu)){
        std::cerr << "Failed to save image"  << std::endl;
    } 
#endif
    return std::make_tuple(r, dw, dh);
}
std::tuple<float, float, float>  PreprocessImage_GPU(std::string path, void *buffer, int channel, int input_h, int input_w, cudaStream_t stream){
    cv::Mat img = cv::imread(path);
    void *img_buffer = nullptr;
    int orig_h = img.rows;
    int orig_w = img.cols;
    CUDA_CHECK(cudaMalloc(&img_buffer, orig_h * orig_w * 3));
    void *img_ptr = img.data;
    CUDA_CHECK(cudaMemcpyAsync(img_buffer, img_ptr, orig_h * orig_w * 3, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    std::tuple<float, float, float> res = Letterbox_resize_GPU(orig_h, orig_w, img_buffer, buffer, input_h, input_w);

    float &r = std::get<0>(res);
    float &dw = std::get<1>(res);
    float &dh = std::get<2>(res);
    
    Npp8u *pu8_rgb = nullptr;
    CUDA_CHECK(cudaMalloc(&pu8_rgb, input_h * input_w * 3));
    // BGR-->RGB
    int aOrder[3] = {2, 1, 0};
    NppiSize size = {input_w, input_h};
    NppStatus ret = nppiSwapChannels_8u_C3R((Npp8u*)buffer, input_w * 3, pu8_rgb, input_w * 3, size, aOrder);
    if(ret != 0){
        std::cerr << "nppiSwapChannels_8u_C3R error: " << ret << std::endl;
    }

    // 转 float 并归一化
    NppiSize fsize = {input_w, input_h};
    ret = nppiConvert_8u32f_C3R(pu8_rgb, input_w * 3, (Npp32f*)buffer, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        std::cerr << "nppiConvert_8u32f_C3R error: " << ret << std::endl;
    }
    Npp32f aConstants[3] = {1.f / 255.f, 1.f / 255.f,1.f / 255.f};
    ret = nppiMulC_32f_C3IR(aConstants, (Npp32f*)buffer, input_w * 3 * sizeof(float), fsize);
    if(ret != 0){
        std::cerr << "nppiMulC_32f_C3IR error: " << ret << std::endl;
    }

    // HWC TO CHW
    NppiSize chw_size = {input_w, input_h};
    float* buffer_chw = nullptr;
    CUDA_CHECK(cudaMalloc(&buffer_chw, input_h * input_w * 3 * sizeof(float)));
    Npp32f* dst_planes[3];
    dst_planes[0] = (Npp32f*)buffer_chw;                           // R
    dst_planes[1] = (Npp32f*)buffer_chw + input_h * input_w;       // G
    dst_planes[2] = (Npp32f*)buffer_chw + input_h * input_w * 2;   // B
    ret = nppiCopy_32f_C3P3R((Npp32f*)buffer, input_w * 3 * sizeof(float), dst_planes, input_w * sizeof(float), chw_size);
    if (ret != 0) {
        std::cerr << "nppiCopy_32f_C3P3R error: " << ret << std::endl;
    }
    CUDA_CHECK(cudaMemcpy(buffer, buffer_chw, input_h * input_w * 3 * sizeof(float), cudaMemcpyDeviceToDevice));

    CUDA_CHECK(cudaFree(buffer_chw));
    CUDA_CHECK(cudaFree(img_buffer));
    CUDA_CHECK(cudaFree(pu8_rgb));
    return std::move(std::make_tuple(r, dw, dh));
}
int Inference(nvinfer1::IExecutionContext* context, nvinfer1::ICudaEngine* engine, std::vector<std::pair<int, std::string>> in_tensor_info, std::vector<std::pair<int, std::string>> out_tensor_info,
              void**buffers, void** host_outs, const int batch_size, int channel, int input_h, int input_w, 
              int max_out0_size_byte, int max_out1_size_byte, int max_out2_size_byte, cudaStream_t stream){
    nvinfer1::Dims trt_in0_dims{}, trt_in1_dims{};
    trt_in0_dims.nbDims = 4;
    trt_in0_dims.d[0] = batch_size;
    trt_in0_dims.d[1] = 3;
    trt_in0_dims.d[2] = input_h;
    trt_in0_dims.d[3] = input_w;
    context->setInputShape(in_tensor_info[0].second.c_str(), trt_in0_dims);
    trt_in1_dims.nbDims = 2;
    trt_in1_dims.d[0] = 1;
    trt_in1_dims.d[1] = 2;
    context->setInputShape(in_tensor_info[1].second.c_str(), trt_in1_dims);
    if(!context->enqueueV3(stream)) {
        std::cerr << "enqueueV3 failed!" << std::endl;
        return -2;
    }
    CUDA_CHECK(cudaMemcpyAsync(host_outs[0], buffers[2], max_out0_size_byte, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_outs[1], buffers[3], max_out1_size_byte, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(host_outs[2], buffers[4], max_out2_size_byte, cudaMemcpyDeviceToHost, stream));

    CUDA_CHECK(cudaStreamSynchronize(stream));
    return 0;
}
static float IoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float inter = (a & b).area();
    float uni = a.area() + b.area() - inter;
    return uni <= 0.f ? 0.f : inter / uni;
}

static std::vector<int> NMS(const std::vector<Detection>& dets, float iou_thres) {
    std::vector<int> order(dets.size());
    for (size_t idx = 0; idx < order.size(); ++idx) {
        order[idx] = static_cast<int>(idx);
    }
    std::sort(order.begin(), order.end(), [&](int i, int j){
        return dets[i].score > dets[j].score;
    });

    std::vector<int> keep;
    std::vector<char> removed(dets.size(), 0);
    for (size_t _i = 0; _i < order.size(); ++_i) {
        int i = order[_i];
        if (removed[i]) continue;
        keep.push_back(i);
        for (size_t _j = _i + 1; _j < order.size(); ++_j) {
            int j = order[_j];
            if (removed[j]) continue;
            if (dets[i].class_id != dets[j].class_id) continue; // 不同类不互相抑制（常见策略）
            if (IoU(dets[i].box, dets[j].box) > iou_thres) {
                removed[j] = 1;
            }
        }
    }
    return keep;
}
static std::vector<Detection> PostprocessDetections(
    const int64_t * feat_labels, const float* feat_boxes, const float* feat_scores,
    int output_num,
    float r, float dw, float dh,    // 反 letterbox 参数
    int orig_w, int orig_h,         // 原图大小
    float conf_thres = 0.5f,
    float iou_thres  = 0.5f)
{
    int num_classes = (int)(sizeof(class_names)/sizeof(class_names[0]));
    std::vector<Detection> dets;
    dets.reserve(512);
    int walk = 4;
    // for(int i = 0; i < 300*4;i++){
    //     std::cout << feat_boxes[i];
    // }
    // printf("\n");
    // for(int i = 0; i < 300;i++){
    //     std::cout << feat_labels[i];
    // }
    // printf("\n");
    // for(int i = 0; i < 300;i++){
    //     std::cout << feat_scores[i];
    // }
    // printf("\n");
    for(int i = 0; i < output_num; i++){
        int label = static_cast<int>(feat_labels[i]);
        float scores = feat_scores[i];
        float x1 = feat_boxes[i * walk];
        float y1 = feat_boxes[i * walk + 1];
        float x2 = feat_boxes[i * walk + 2];
        float y2 = feat_boxes[i * walk + 3];
        // std::cout << "label:" << label << " scores:" << scores << " x1:" << x1 << " y1:" << y1 << " x2:" << x2 << " y2:" << y2 << std::endl;
        if(scores < conf_thres){
            continue;
        }
        float w  = x2 - x1;
        float h  = y2 - y1;
        float cx = x1 + w / 2;
        float cy = y1 + h / 2;

        float x = (cx - w * 0.5f - dw) / r;
        float y = (cy - h * 0.5f - dh) / r;
        float ww = w / r;
        float hh = h / r;

        x  = std::max(0.f, std::min(x,  (float)orig_w  - 1.f));
        y  = std::max(0.f, std::min(y,  (float)orig_h - 1.f));
        ww = std::max(0.f, std::min(ww, (float)orig_w  - x));
        hh = std::max(0.f, std::min(hh, (float)orig_h - y));

        if (ww <= 0.f || hh <= 0.f) continue;

        Detection d;
        d.box = cv::Rect2f(x, y, ww, hh);
        d.score = scores;
        d.class_id = label;
        dets.push_back(d);
    }
    // NMS
    // std::vector<int> keep = NMS(dets, iou_thres);
    // std::vector<Detection> out;
    // out.reserve(keep.size());
    // for (int idx : keep) out.push_back(dets[idx]);
    // return out;
    return dets;
}
size_t CountElement(const nvinfer1::Dims &dims, int batch_zise)
{
    int64_t total = batch_zise;
    for (int32_t i = 1; i < dims.nbDims; ++i){
        total *= dims.d[i];
    }
    return static_cast<size_t>(total);
}
int main(int argc, char **argv){
    if(argc < 3){
        std::cerr << "./bin eng_path test.jpg" << std::endl;
        return 0;
    }
    const char *eng_path = argv[1];
    const char *img_path = argv[2];
    int device_id = 0;
    cudaStream_t stream;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaStreamCreate(&stream));
    nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(logger);
	assert(runtime != nullptr);
    int model_size = 0;
    char *trt_model_stream = ReadFromPath(eng_path,model_size);
    assert(trt_model_stream != nullptr);
    auto engine{runtime->deserializeCudaEngine(trt_model_stream, model_size)};
	assert(engine != nullptr);

    auto context{engine->createExecutionContext()};
	assert(context != nullptr);
    delete []trt_model_stream;

    int num_bindings = engine->getNbIOTensors();
	std::cout << "input/output : " << num_bindings << std::endl;
	std::vector<std::pair<int, std::string>> in_tensor_info;
	std::vector<std::pair<int, std::string>> out_tensor_info;
    for (int i = 0; i < num_bindings; ++i)
    {
        const char *tensor_name = engine->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine->getTensorIOMode(tensor_name);
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            in_tensor_info.push_back({i, std::string(tensor_name)});
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            out_tensor_info.push_back({i, std::string(tensor_name)});
    }
    for(int idx = 0; idx < in_tensor_info.size(); idx++){
        nvinfer1::Dims in_dims=context->getTensorShape(in_tensor_info[idx].second.c_str());
        std::cout << "input: " << in_tensor_info[idx].second.c_str() << std::endl;
        for(int i = 0; i < in_dims.nbDims; i++){
            std::cout << "dims [" << i << "]: " << in_dims.d[i] << std::endl;
        }
        nvinfer1::DataType size_type = engine->getTensorDataType(in_tensor_info[idx].second.c_str());
        if (size_type == nvinfer1::DataType::kINT32) {
            std::cout << "类型为 int32" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kINT64) {
            std::cout << "类型为 int64" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kFLOAT) {
            std::cout << "类型为 float" << std::endl;
        }
        std::cout << std::endl;
    }
    for(int idx = 0; idx < out_tensor_info.size(); idx++){
        nvinfer1::Dims out_dims=context->getTensorShape(out_tensor_info[idx].second.c_str());
        std::cout << "output: " << out_tensor_info[idx].second.c_str() << std::endl;
        for(int i = 0; i < out_dims.nbDims; i++){
            std::cout << "dims [" << i << "]: " << out_dims.d[i] << std::endl;
        }
        nvinfer1::DataType size_type = engine->getTensorDataType(out_tensor_info[idx].second.c_str());
        if (size_type == nvinfer1::DataType::kINT32) {
            std::cout << "类型为 int32" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kINT64) {
            std::cout << "类型为 int64" << std::endl;
        } 
        else if (size_type == nvinfer1::DataType::kFLOAT) {
            std::cout << "类型为 float" << std::endl;
        }
        std::cout << std::endl;
    }
    int batch_size = 4;
    size_t max_in0_size_byte = CountElement(context->getTensorShape(in_tensor_info[0].second.c_str()), batch_size) * sizeof(float);
    size_t max_in1_size_byte = CountElement(context->getTensorShape(in_tensor_info[1].second.c_str()), batch_size) * sizeof(int64_t); // 注意这里是int64_t-orig_target_sizes
    size_t max_out0_size_byte = CountElement(context->getTensorShape(out_tensor_info[0].second.c_str()), batch_size) * sizeof(int64_t); // 注意这里是int64_t-labels
    size_t max_out1_size_byte = CountElement(context->getTensorShape(out_tensor_info[1].second.c_str()), batch_size) * sizeof(float);
    size_t max_out2_size_byte = CountElement(context->getTensorShape(out_tensor_info[2].second.c_str()), batch_size) * sizeof(float);
    char *host_outs[3] = {NULL, NULL, NULL};
    host_outs[0] = new char[max_out0_size_byte];
    host_outs[1] = new char[max_out1_size_byte];
    host_outs[2] = new char[max_out2_size_byte];

    void* buffers[5] = {NULL, NULL, NULL, NULL, NULL};
    CUDA_CHECK(cudaMalloc(&buffers[in_tensor_info[0].first], max_in0_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[in_tensor_info[1].first], max_in1_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[out_tensor_info[0].first], max_out0_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[out_tensor_info[1].first], max_out1_size_byte));
    CUDA_CHECK(cudaMalloc(&buffers[out_tensor_info[2].first], max_out2_size_byte));
    // set in/out tensor address
    context->setInputTensorAddress(in_tensor_info[0].second.c_str(), buffers[in_tensor_info[0].first]);
    context->setInputTensorAddress(in_tensor_info[1].second.c_str(), buffers[in_tensor_info[1].first]);
    context->setOutputTensorAddress(out_tensor_info[0].second.c_str(), buffers[out_tensor_info[0].first]);
    context->setOutputTensorAddress(out_tensor_info[1].second.c_str(), buffers[out_tensor_info[1].first]);
    context->setOutputTensorAddress(out_tensor_info[2].second.c_str(), buffers[out_tensor_info[2].first]);

    int test_batch = 2;
    std::vector<std::tuple<float, float, float>> res_pre;
    int buffer_idx = 0;
    char* input_ptr_images = static_cast<char*>(buffers[0]);
    char* input_ptr_orig_target_sizes = static_cast<char*>(buffers[1]);
    int channel = 3;
    int input_h = 640;
    int input_w = 640;
    for(int i = 0; i < test_batch; i++){
#ifdef PROC_GPU
        std::tuple<float, float, float> res = PreprocessImage_GPU(img_path, input_ptr_images + buffer_idx, channel, input_h, input_w, stream);
#else
        std::tuple<float, float, float> res = PreprocessImage(img_path, input_ptr_images + buffer_idx, channel, input_h, input_w, stream);
#endif
        buffer_idx += input_h * input_w * 3 * sizeof(float);
        res_pre.push_back(res);
        
    }
    // must be int64_t
    int64_t img_size[2] = {(int64_t)input_h, (int64_t)input_w};
    CUDA_CHECK(cudaMemcpyAsync(input_ptr_orig_target_sizes, img_size, sizeof(int64_t) * 2, cudaMemcpyHostToDevice, stream));
    Inference(context, engine, in_tensor_info, out_tensor_info, buffers, (void**)host_outs,
              res_pre.size(), channel, input_h, input_w, max_out0_size_byte, max_out1_size_byte, max_out2_size_byte, stream);
    cv::Mat original = cv::imread(img_path);
    int orig_h = original.rows, orig_w = original.cols;
    int64_t  *output_labels = reinterpret_cast<int64_t  *>(host_outs[0]);
    float *output_boxes = reinterpret_cast<float *>(host_outs[1]);
    float *output_scores = reinterpret_cast<float *>(host_outs[2]);
    for (int b = 0; b < test_batch; ++b) {
        auto [r, dw, dh] = res_pre[b];
        int64_t * feat_labels = output_labels + b * 300;
        float* feat_boxes = output_boxes + b * 300 * 4;
        float* feat_scores = output_scores + b * 300;

        std::vector<Detection> dets = PostprocessDetections(
            feat_labels, feat_boxes, feat_scores, 300, r, dw, dh, orig_w, orig_h, /*conf*/0.5f, /*iou*/0.5f);

        cv::Mat vis = original.clone();
        for (const auto& d : dets) {
            cv::rectangle(vis, d.box, cv::Scalar(0, 255, 0), 2);
            char text[128];
            const char* cname = (d.class_id >= 0 && d.class_id < (int)(sizeof(class_names)/sizeof(class_names[0])))
                                    ? class_names[d.class_id] : "cls";
            snprintf(text, sizeof(text), "%s: %.2f", cname, d.score);
            int baseline = 0;
            cv::Size tsize = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::rectangle(vis, cv::Rect(cv::Point((int)d.box.x, (int)d.box.y - tsize.height - 4),
                                        cv::Size(tsize.width + 4, tsize.height + 4)),
                        cv::Scalar(0, 255, 0), cv::FILLED);
            cv::putText(vis, text, cv::Point((int)d.box.x + 2, (int)d.box.y - 2),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        std::string save_name = std::string("result_") + std::to_string(b) + ".jpg";
        cv::imwrite(save_name, vis);
        std::cout << "Saved: " << save_name << "  dets=" << dets.size() << std::endl;
    }
    for(int i = 0; i < 5; i++){
        CUDA_CHECK(cudaFree(buffers[i]));
    }
    CUDA_CHECK(cudaStreamDestroy(stream));
    delete []host_outs[0];
    delete []host_outs[1];
    delete []host_outs[2];
    return 0;
}
#endif