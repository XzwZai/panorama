#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
class Camera
{
public:
	double focal = 0;
	double ppx = 0, ppy = 0;
	Mat K;
	Mat R;
	Camera()
	{
		K = Mat::zeros(Size(3, 3), CV_64FC1);
		R = Mat::eye(Size(3, 3), CV_64FC1);
	}

	void setK()
	{
		K.at<double>(0, 0) = focal; K.at<double>(1, 1) = focal; K.at<double>(2, 2) = 1;
		K.at<double>(0, 2) = ppx; K.at<double>(1, 2) = ppy;
	}
};