#include <libSVM.h>
int threshold = 100;
int color = 0;

int main(int argc, char **argv)
{
    
    std::string path = std::string(argv[1]);
    std::string image = std::string(argv[2]);
    std::string type = std::string(argv[3]);

    std::cout << path << " " << image << std::endl;
    
    initSvmKernal(path, 64, 16, 8, 4, 9);
    if (type == "m")
    {
        auto start = std::chrono::system_clock::now();
        std::vector<roiInfo> batch;
        cv::Mat testImg, proc, dst;
        testImg = cv::imread(image);
        preProcess(testImg, proc, threshold, color); // dst, dstImg are processed images.
        cv::imwrite("process.png", proc);
        doDetect(testImg, proc, true, batch);
        cv::imwrite("result.png", testImg);
        auto end = std::chrono::system_clock::now();
        std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;
    }
    else if (type == "v")
    {
        
        cv::VideoCapture cap;
        cap.open(image);
        if (!cap.isOpened())
            return 1;

        cv::Mat frame;
        while (1)
        {
            std::vector<roiInfo> batch;
            cap >> frame;
            if (frame.empty())
                break;
            cv::Mat dst;
            cv::Mat dstImg;
            preProcess(frame, dst, threshold, color);
            cv::imshow("process", dst);
            doDetect(frame, dst, true,batch);
            cv::imshow("result", frame);
            cv::waitKey(1);
        }
        cap.release();
    }
}