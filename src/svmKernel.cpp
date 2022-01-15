#include <libSVM.h>

cv::HOGDescriptor *svmHog;
cv::Ptr<cv::ml::SVM> svmDetector;

void initSvmKernal(std::string xmlPath, int win_size, int block_size, int block_stride, int cell_size, int nbins)
{
    svmHog = new cv::HOGDescriptor(cv::Size(win_size, win_size), cv::Size(block_size, block_size), cv::Size(block_stride, block_stride), cv::Size(cell_size, cell_size), nbins);
    svmDetector = cv::ml::SVM::load(xmlPath);
    std::cout << "Initlize successfully." << std::endl;
}

cv::Mat getHOG(cv::Mat &sample)
{
    std::vector<float> SVM_vector;
    std::vector<float> descriptors;
    cv::resize(sample, sample, cv::Size(64, 64));
    svmHog->compute(sample, descriptors, cv::Size(1, 1), cv::Size(0, 0));
    cv::Mat SVM_input = cv::Mat(1, descriptors.size(), CV_32FC1);
    for (int i = 0; i < descriptors.size(); i++)
    {
        SVM_input.at<float>(0, i) = descriptors[i];
    }
    SVM_input.convertTo(SVM_input, CV_32FC1);
    return SVM_input;
}

int getClass(cv::Mat &input)
{
    cv::Mat SVM_input = getHOG(input);
    int classid = -1;
    float a = svmDetector->getGamma();
    std::cout << a;
    classid = svmDetector->predict(SVM_input);
    return classid;
}
