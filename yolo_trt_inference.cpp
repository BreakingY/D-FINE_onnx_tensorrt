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
int Inference(nvinfer1::IExecutionContext* context, void** buffers, void* output, int one_output_len, const int batch_size, int channel, int input_h, int input_w, int input_index, int output_index, cudaStream_t stream){
    context->setBindingDimensions(0, nvinfer1::Dims4(batch_size, channel, input_h, input_w));
    if(!context->enqueueV2(buffers, stream, nullptr)) {
        std::cerr << "enqueueV2 failed!" << std::endl;
        return -2;
    }
    CUDA_CHECK(cudaMemcpyAsync(output, buffers[output_index], batch_size * one_output_len * sizeof(float), cudaMemcpyDeviceToHost, stream));
    cudaStreamSynchronize(stream);
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
// 解析形状: [output_pred, anchors], 每一列表示一个目标的所有信息 4 + len(class_names)
static std::vector<Detection> PostprocessDetections(
    const float* feat,              // 指向单张图的输出首地址
    int output_pred,                // 4 + num_classes
    int anchors,                    // 锚点总数
    float r, float dw, float dh,    // 反 letterbox 参数
    int orig_w, int orig_h,         // 原图大小
    float conf_thres = 0.5f,
    float iou_thres  = 0.5f)
{
    int num_classes = (int)(sizeof(class_names)/sizeof(class_names[0]));
    std::vector<Detection> dets;
    dets.reserve(512);

    // feat 的内存布局：维度 [output_pred, anchors]
    // 访问方式：feat[i * anchors + j]  (i: 0..output_pred-1, j: 0..anchors-1)
    const float* cx_ptr = feat + 0 * anchors;
    const float* cy_ptr = feat + 1 * anchors;
    const float* w_ptr  = feat + 2 * anchors;
    const float* h_ptr  = feat + 3 * anchors;
    const float* cls_ptr= feat + 4 * anchors;  // 后面紧跟 num_classes * anchors

    for (int j = 0; j < anchors; ++j) {
        // 取类别最大值与 id
        int best_c = -1;
        float best_s = -1.f;
        for (int c = 0; c < num_classes; ++c) {
            float s = cls_ptr[c * anchors + j];
            if (s > best_s) { best_s = s; best_c = c; }
        }
        if (best_s < conf_thres) continue;

        float cx = cx_ptr[j];
        float cy = cy_ptr[j];
        float w  = w_ptr[j];
        float h  = h_ptr[j];

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
        d.score = best_s;
        d.class_id = best_c;
        dets.push_back(d);
    }

    // NMS
    std::vector<int> keep = NMS(dets, iou_thres);
    std::vector<Detection> out;
    out.reserve(keep.size());
    for (int idx : keep) out.push_back(dets[idx]);
    return out;
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
    assert(input_names.size() == 1);
    assert(output_names.size() == 1);
    
    int input_index = engine->getBindingIndex(input_names[0]);
	int output_index = engine->getBindingIndex(output_names[0]);

    // int batch_size = engine->getBindingDimensions(input_index).d[0]; // 动态维度返回-1
    int batch_size = 4; // trtexex转模型设置的最大batch
    int channel = engine->getBindingDimensions(input_index).d[1];
    assert(channel == 3);
    int input_h = engine->getBindingDimensions(input_index).d[2];
	int input_w = engine->getBindingDimensions(input_index).d[3];
    std::cout << "batch_size:" << batch_size << " channel:" << channel << " input_h:" << input_h << " input_w:" << input_w << std::endl;

    // int batch_size = engine->getBindingDimensions(input_index).d[0];
    int output_pred = engine->getBindingDimensions(output_index).d[1]; // 4(c_x, c_y, w, h) + len(class_names)
	int anchors = engine->getBindingDimensions(output_index).d[2]; // 3 个特征图融合后的总 anchor 数量
	// 计算 YOLO 模型输出的预测框数量
    // input_h, input_w: 模型输入的高度和宽度
    // YOLO 会在 3 个尺度进行预测：
    //   1. 下采样 8 倍 (P3 层)，特征图尺寸为 (input_h / 8) × (input_w / 8)
    //   2. 下采样 16 倍 (P4 层)，特征图尺寸为 (input_h / 16) × (input_w / 16)
    //   3. 下采样 32 倍 (P5 层)，特征图尺寸为 (input_h / 32) × (input_w / 32)
    // 每个特征图位置会预测 3 个 anchor 框
    int anchors_model = (
        (input_h / 8)  * (input_w / 8)  +  // 尺度1 (stride=8)
        (input_h / 16) * (input_w / 16) +  // 尺度2 (stride=16)
        (input_h / 32) * (input_w / 32)    // 尺度3 (stride=32)
    ) * 3; // 每个位置 3 个 anchor
    if(anchors != anchors_model){
        if(anchors_model != anchors * 3)
            std::cout << "anchors " << anchors << "anchors_model" << anchors_model << std::endl;
    }
    std::cout << "output_pred:" << output_pred << " anchors:" << anchors << std::endl;

    void* buffers[2] = {NULL, NULL};
    CUDA_CHECK(cudaMalloc(&buffers[input_index], batch_size * input_h * input_w * 3 * sizeof(float)));
    int one_output_len = output_pred * anchors;
	CUDA_CHECK(cudaMalloc(&buffers[output_index], batch_size * one_output_len * sizeof(float)));
    float* output = new float[batch_size * one_output_len];
    int test_batch = 2;
    std::vector<std::tuple<float, float, float>> res_pre;
    int buffer_idx = 0;
    char* input_ptr = static_cast<char*>(buffers[input_index]);
    for(int i = 0; i < test_batch; i++){
#ifdef PROC_GPU
        std::tuple<float, float, float> res = PreprocessImage_GPU(img_path, input_ptr + buffer_idx, channel, input_h, input_w, stream);
#else
        std::tuple<float, float, float> res = PreprocessImage(img_path, input_ptr + buffer_idx, channel, input_h, input_w, stream);
#endif
        buffer_idx += input_h * input_w * 3 * sizeof(float);
        res_pre.push_back(res);
    }
    Inference(context, buffers, (void*)output, one_output_len, res_pre.size(), channel, input_h, input_w, input_index, output_index, stream);
    cv::Mat original = cv::imread(img_path);
    int orig_h = original.rows, orig_w = original.cols;

    for (int b = 0; b < test_batch; ++b) {
        auto [r, dw, dh] = res_pre[b];
        float* feat_b = output + b * one_output_len;

        std::vector<Detection> dets = PostprocessDetections(
            feat_b, output_pred, anchors, r, dw, dh, orig_w, orig_h, /*conf*/0.5f, /*iou*/0.5f);

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
    CUDA_CHECK(cudaFree(buffers[0]));
    CUDA_CHECK(cudaFree(buffers[1]));
    delete []output;
    CUDA_CHECK(cudaStreamDestroy(stream));
    context->destroy();
    engine->destroy();
    return 0;
}