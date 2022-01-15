#include <libSVM.h>

int count_flag=0;

std::deque<cv::Point2f> pointBuf;

std::vector<roiInfo> getROI(cv::Mat &src, cv::Mat &binary);

void CircleInfo2(std::vector<cv::Point2f>& pts, cv::Point2f& center, float& radius);

cv::Mat PerspectiveTransform(cv::Mat &binary, cv::RotatedRect &rect);

float getDistance(cv::Point2f pointA, cv::Point2f pointB);

float getDistance(cv::Point2f pointA, cv::Point2f pointB)
{
    float distance;
    distance = powf((pointA.x - pointB.x), 2) + powf((pointA.y - pointB.y), 2);
    distance = sqrtf(distance);
    return distance;
}

void drawRect(cv::Mat &img, cv::RotatedRect rect)
{
    cv::Point2f *vertices = new cv::Point2f[4];
    rect.points(vertices);
    for (size_t i = 0; i < 4; i++)
    {
        cv::line(img, vertices[i], vertices[(i + 1) % 4], cv::Scalar(255, 0, 255), 4, 8, 0);
    }
}

cv::Mat PerspectiveTransform(cv::Mat &binary, cv::RotatedRect &rect)
{
    cv::Point2f tempPoints[4];
    rect.points(tempPoints);
    cv::Point2f srcRect[4];
    cv::Point2f dstRect[4];

    float width = getDistance(tempPoints[0], tempPoints[1]);
    float height = getDistance(tempPoints[1], tempPoints[2]);

    if (width > height)
    {
        srcRect[0] = tempPoints[0];
        srcRect[1] = tempPoints[1];
        srcRect[2] = tempPoints[2];
        srcRect[3] = tempPoints[3];
    }
    else
    {
        cv::swap(width, height);
        srcRect[0] = tempPoints[1];
        srcRect[1] = tempPoints[2];
        srcRect[2] = tempPoints[3];
        srcRect[3] = tempPoints[0];
    }

    dstRect[0] = cv::Point2f(0, 0);
    dstRect[1] = cv::Point2f(width, 0);
    dstRect[3] = cv::Point2f(0, height);
    dstRect[2] = cv::Point2f(width, height);

    cv::Mat transform = getPerspectiveTransform(srcRect, dstRect);
    Eigen::Matrix<double, 3, 3> R_matrix;
    cv::cv2eigen(transform, R_matrix);
    R_matrix(2, 0) = 0;
    R_matrix(2, 1) = 0;
    Eigen::Matrix3d recMat = R_matrix.inverse();

    cv::Mat ROI = cv::Mat(height, width, CV_8UC1, cv::Scalar(0,0,0));
    for (int y = 3; y < height-3; y++)
    {
        for (int x = 3; x < width-3; x++)
        {
            Eigen::Matrix<double, 3, 1> srcPoint;
            srcPoint(0, 0) = x;
            srcPoint(1, 0) = y;
            srcPoint(2, 0) = 1;
            Eigen::Matrix<double, 3, 1> dstPoint;
            dstPoint = recMat * srcPoint;
            if (dstPoint(1, 0) < binary.rows && dstPoint(0, 0) < binary.cols)
            {
                ROI.at<uchar>(y, x) = binary.at<uchar>((int)dstPoint(1, 0), (int)dstPoint(0, 0));
            }
        }
    }
    return ROI;
}

void doDetect(cv::Mat &img, cv::Mat &dst, bool isDebug, std::vector<roiInfo> &windmillBatch)
{

    std::vector<roiInfo> dstrInfo = getROI(img, dst);

    if (isDebug)
    {
        std::cout << "ROI Rect Num: " << dstrInfo.size() << std::endl;
    }
    for (int i = 0; i < dstrInfo.size(); i++)
    {
        int result = getClass(dstrInfo[i].roiImg);
        dstrInfo[i].id = result;
        if(dstrInfo.size()>30){
            pointBuf.pop_front();
            pointBuf.push_back(dstrInfo[i].srcRect.center);
        }
        else{
            pointBuf.push_back(dstrInfo[i].srcRect.center);
        }
        if (isDebug)
        {
            std::cout << "Result ID: " << result << std::endl;
            drawRect(img, dstrInfo[i].srcRect);
        }
    }
    if (isDebug)
    {
        std::cout << "ROI area: " << dstrInfo.size() << std::endl;
    }
    windmillBatch = dstrInfo;
}

std::vector<roiInfo> getROI(cv::Mat &src, cv::Mat &binary)
{
    std::vector<roiInfo> dstrInfo;
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    findContours(binary, contours, hierarchy, CV_RETR_TREE, cv::CHAIN_APPROX_NONE);
    std::vector<cv::RotatedRect> rectRes;

    std::vector<cv::Mat> ROI;

    for (int i = 0; i < contours.size(); i++)
    {
        if (contours[i].size() > 100 && hierarchy[i][2] != -1)
        {
            cv::RotatedRect temp = minAreaRect(contours[i]);
            rectRes.push_back(temp);
            drawRect(binary, temp);
        }
    }

    for (int i = 0; i < rectRes.size(); i++)
    {
        roiInfo tmpInfo;
        tmpInfo.srcRect = rectRes[i];
        cv::Mat rrectROI = PerspectiveTransform(binary, rectRes[i]);
        tmpInfo.roiImg = rrectROI;
        dstrInfo.push_back(tmpInfo);
    }
    return dstrInfo;
}

void preProcess(cv::Mat &img, cv::Mat &dst, int threshold, int color)
{
    std::vector<cv::Mat> channels;
    cv::Mat gray, threImg;

    cv::split(img, channels);
    if (color == 1)
    { // RED
        cv::subtract(channels[2], channels[0], gray);
    }
    else if (color == 0)
    { // BLUE
        cv::subtract(channels[0], channels[2], gray);
    }
    cv::threshold(gray, threImg, threshold, 255, cv::THRESH_BINARY);

    cv::Mat dilate_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7), cv::Point(-1, -1));
    cv::Mat close_kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
    cv::dilate(threImg, dst, dilate_kernel);
    cv::erode(dst, dst, close_kernel);
    count_flag++;
}

