#pragma once
#include <iostream>
#include <deque>
#include <vector>
#include <ctime>
#include <chrono>
#include <Eigen/Eigen>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/opencv.hpp>

struct roiInfo{
    cv::RotatedRect srcRect;
    cv::Mat roiImg;
    int id;
};

struct windmill{
    cv::Point2f center;
    int r;
};

void initSvmKernal(std::string xmlPath, int win_size = 64, int block_size = 16, int block_stride = 8, int cell_size = 4, int nbins = 9);

void preProcess(cv::Mat &img, cv::Mat &dst, int threshold,int color);

int getClass(cv::Mat &input);

void doDetect(cv::Mat &img, cv::Mat &dst, bool isDebug, std::vector<roiInfo> &windmillBatch);

void processWindmill(std::vector<roiInfo> &windmillBatch, int windmill_l, int windmill_r);

