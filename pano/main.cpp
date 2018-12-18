#include <iostream>

#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "cylinder.h"

using namespace cv;
using namespace std;
const float GOOD_MATCH_PERCENT = 0.15f;

Size imgSize = Size(640, 960);
int f = 660;
int maxNum(int a,int b)
{
	return a > b ? a : b;
}

int minNum(int a, int b)
{
	return a < b ? a : b;
}

Mat normalizeMat(Mat& m)
{
	Mat t;
	m.copyTo(t);
	t.at<double>(0, 0) /= t.at<double>(2, 0);
	t.at<double>(1, 0) /= t.at<double>(2, 0);
	return t;
}

Mat cylinder(Mat imgIn, int f = 330) {
	int colNum, rowNum; colNum = 2 * f*atan(0.5*imgIn.cols / f);
	rowNum = 0.5*imgIn.rows*f / sqrt(pow(f, 2)) + 0.5*imgIn.rows;
	Mat imgOut = Mat::zeros(rowNum, colNum, CV_8UC3); Mat_<uchar> im1(imgIn);
	Mat_<uchar> im2(imgOut); 
	int x1(0), y1(0);
	for (int i = 0; i < imgIn.rows; i++)
		for (int j = 0; j < imgIn.cols; j++) {
			x1 = f * atan((j - 0.5*imgIn.cols) / f) + f * atan(0.5*imgIn.cols / f);
			y1 = f * (i - 0.5*imgIn.rows) / sqrt(pow(j - 0.5*imgIn.cols, 2) + pow(f, 2)) + 0.5*imgIn.rows;
			if (x1 >= 0 && x1 < colNum&&y1 >= 0 && y1 < rowNum) {
				//im2(y1, x1) = im1(i, j); 
				imgOut.at<Vec3b>(y1, x1) = imgIn.at<Vec3b>(i, j);
			}
		}
	return imgOut;
}

void stitch1(vector<Mat> imgs)
{
	vector<Point2i> corners;
	corners.push_back(Point2i(0, 0));
	corners.push_back(Point2i(1, 0));
	corners.push_back(Point2i(0, 1));
	corners.push_back(Point2i(1, 1));
	Mat temp = imgs[0];
	Rect maskRect(0, 0, temp.cols, temp.rows);
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(1600);
	BFMatcher matcher;
	//Rect maskRect(imgs[0].cols, imgs[0].rows, imgs[0].cols, imgs[0].rows);
	for (int i = 1; i < imgs.size(); i++)
	{
		Mat mask = Mat::zeros(temp.size(), CV_8UC1);
		mask(maskRect).setTo(255);
		Mat imgRight = imgs[i];
		vector<KeyPoint> kps1, kps2;
		Mat descriptors1, descriptors2;
		vector<DMatch> matches;
		sift->detectAndCompute(temp, mask, kps1, descriptors1);
		sift->detectAndCompute(imgRight, Mat(), kps2, descriptors2);

		matcher.match(descriptors1, descriptors2, matches, Mat());
		sort(matches.begin(), matches.end());


		const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
		matches.erase(matches.begin() + numGoodMatches, matches.end());
		Mat imMatches;
		Mat h;
		std::vector<Point2f> points1, points2;
		for (size_t i = 0; i < matches.size(); i++)
		{
			points1.push_back(kps1[matches[i].queryIdx].pt);
			points2.push_back(kps2[matches[i].trainIdx].pt);
		}
		drawMatches(temp, kps1, imgRight, kps2, matches, imMatches);
		imshow("4", imMatches);
		h = findHomography(points1, points2, RANSAC);
		cout << "h:" << h << endl;
		vector<Point2i> corners_loc;
		for (int i = 0; i < 4; i++)
		{
			Mat m = h * (Mat_<double>(3, 1) << temp.cols * corners[i].x, temp.rows * corners[i].y, 1);
			m = normalizeMat(m);
			cout << m << endl;
			//cout << m.at<double>(0, 0) << m.at<double>(1, 0) << endl;
			corners_loc.push_back(Point2i(m.at<double>(0, 0), m.at<double>(1, 0)));
		}
		cout << endl;
		Rect bounding = boundingRect(corners_loc);
		Mat imgaligned;
		Mat trans = Mat::eye(3, 3, CV_64F);
		trans.at<double>(0, 2) = -bounding.x;
		trans.at<double>(1, 2) = -bounding.y;
		h = trans * h;
		Size tempSize = Size(imgRight.cols - bounding.x, maxNum(imgRight.rows, bounding.y + bounding.height - 1) - minNum(0, bounding.y));
		warpPerspective(temp, imgaligned, h, tempSize);
		imshow("2", imgaligned);
		Rect dtsRect = Rect(-bounding.x, maxNum(0, -bounding.y), imgRight.cols, imgRight.rows);
		imgRight.copyTo(imgaligned(dtsRect));
		maskRect = dtsRect;
		imgaligned.copyTo(temp);
		rectangle(imgaligned, maskRect, Scalar(255, 0, 0));
		imshow("1", imgRight);
		imshow("3", temp);
		waitKey(0);
	}
}

void stitch2(vector<Mat> imgs)
{
	vector<Point2i> corners;
	corners.push_back(Point2i(0, 0));
	corners.push_back(Point2i(1, 0));
	corners.push_back(Point2i(0, 1));
	corners.push_back(Point2i(1, 1));
	Mat resultImg = imgs[0];
	Rect maskRect(0, 0, resultImg.cols, resultImg.rows);
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(1600);
	BFMatcher matcher;
	//Rect maskRect(imgs[0].cols, imgs[0].rows, imgs[0].cols, imgs[0].rows);
	for (int i = 1; i < imgs.size(); i++)
	{
		Mat mask = Mat::zeros(resultImg.size(), CV_8UC1);
		mask(maskRect).setTo(255);
		Mat imgRight = imgs[i];
		vector<KeyPoint> kps1, kps2;
		Mat descriptors1, descriptors2;
		vector<DMatch> matches;
		sift->detectAndCompute(resultImg, mask, kps1, descriptors1);
		sift->detectAndCompute(imgRight, Mat(), kps2, descriptors2);

		matcher.match(descriptors1, descriptors2, matches, Mat());
		sort(matches.begin(), matches.end());
		const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
		matches.erase(matches.begin() + numGoodMatches, matches.end());
		Mat imMatches;
		Mat h;
		std::vector<Point2f> points1, points2;
		for (size_t i = 0; i < matches.size(); i++)
		{
			points1.push_back(kps1[matches[i].queryIdx].pt);
			points2.push_back(kps2[matches[i].trainIdx].pt);
		}
		drawMatches(resultImg, kps1, imgRight, kps2, matches, imMatches);
		imshow("4", imMatches);
		h = findHomography(points2, points1, RANSAC);
		cout << "h:" << h << endl;
		vector<Point2i> corners_loc;
		vector<int> xs;
		for (int i = 0; i < 4; i++)
		{
			Mat m = h * (Mat_<double>(3, 1) << imgRight.cols * corners[i].x, imgRight.rows * corners[i].y, 1);
			m = normalizeMat(m);
			cout << m << endl;
			//cout << m.at<double>(0, 0) << m.at<double>(1, 0) << endl;
			corners_loc.push_back(Point2i(m.at<double>(0, 0), m.at<double>(1, 0)));
			xs.push_back(m.at<double>(0, 0));
		}
		sort(xs.begin(), xs.end());
		cout << endl;
		Rect bounding = boundingRect(corners_loc);
		Mat imgaligned;
		Mat trans = Mat::eye(3, 3, CV_64F);
		trans.at<double>(0, 2) = -bounding.x;
		trans.at<double>(1, 2) = -bounding.y;
		h = trans * h;
		warpPerspective(imgRight, imgaligned, h, bounding.size());
		bounding.x = xs[1] + 1;
		bounding.width = xs[2] - xs[1];
		Mat img;
		imgaligned.copyTo(img);
		imgaligned = imgaligned(Rect(xs[1] - xs[0] + 1, 0, bounding.width, imgaligned.rows));
		imshow("b", img);
		imshow("2", imgaligned);
		//waitKey(0);
		Size tempSize = Size(maxNum(resultImg.cols,bounding.x + bounding.width), maxNum(resultImg.rows, bounding.y + bounding.height - 1) - minNum(0, bounding.y)+1);
		Mat temp = Mat::zeros(tempSize, CV_8UC3);
		Rect dstRect = Rect(bounding.x, maxNum(0, bounding.y), imgaligned.cols, imgaligned.rows);
		resultImg.copyTo(temp(Rect(0, maxNum(0, -bounding.y),resultImg.cols,resultImg.rows)));
		imgaligned.copyTo(temp(dstRect));
		//imgRight.copyTo(imgaligned(dstRect));
		maskRect = dstRect;
		/*maskRect.y += 30;
		maskRect.height -= 60;*/
		temp.copyTo(resultImg);
		//rectangle(resultImg, maskRect, Scalar(255, 0, 0));
		imshow("1", imgRight);
		imshow("3", resultImg);
		waitKey(0);
	}
}

int main(int argc, char** argv)
{
	
	ifstream files("D:/WorkSpace/data/pano/Synthetic/imgs.txt");
	//ifstream files("C:/Users/China/Downloads/example-data/example-data/zijing/imgs.txt");
	string imgpath;
	vector<Mat> imgs;
	
	/*while (files >> imgpath)
	{
		Mat img = imread(imgpath);
		resize(img, img, imgSize);		
		imgs.push_back(img);
	}
	cout << "read imgs" << endl;*/
	CylinderStitcher cs;
	//vector<double> all_focus;
	//double f0, f1;
	//bool f0_ok, f1_ok;
	//Mat t0 = Mat::eye(Size(3, 3), CV_64FC1);
	//Mat t1 = Mat::eye(Size(3, 3), CV_64FC1);
	//t0.at<double>(0, 2) = -imgs[0].cols / 2;
	//t0.at<double>(1, 2) = -imgs[0].rows / 2;
	//t1.at<double>(0, 2) = imgs[0].cols / 2;
	//t1.at<double>(1, 2) = imgs[0].rows / 2;
	//for (int i = 1; i < imgs.size(); i++) {
	//	Mat h = cs.getHomography(imgs[i-1], imgs[i]);
	//	//cout << h << endl;
	//	//h = t0 * h * t1;
	//	//cout << h << endl;
	//	cs.focalsFromHomography(h, f0, f1, f0_ok, f1_ok);
	//	cout << f0 << " " << f1 << " " << f0_ok << " " << f1_ok << " " << sqrt(f0*f1) << endl;
	//	all_focus.push_back(sqrt(f0*f1));
	//}
	//sort(all_focus.begin(), all_focus.end());
	//double focus;
	//if (all_focus.size() % 2 == 0) {
	//	focus = all_focus[all_focus.size() / 2];
	//}
	//else {
	//	focus = (all_focus[all_focus.size() / 2 + 1] + all_focus[all_focus.size() / 2]) * 0.5;
	//}
	//cout << focus << endl;	
	//Mat stitched = cs.stitch(imgs, f);
	//imshow("stitched", stitched);
	//imwrite("2.jpg", stitched);
	//cs.calFocus(imgs[0], imgs[1]);
	
	/*Mat img1 = imread("d://1.jpg");
	Mat img2 = imread("d://3.jpg");*/
	vector<Point> ps1, ps2;
	ps1.push_back(Point(600,800));
	ps1.push_back(Point(800, 600));
	ps1.push_back(Point(800, 0));
	ps1.push_back(Point(600, 0));
	ps2.push_back(Point(266.7, 777.1));
	ps2.push_back(Point(400, 682.8));
	ps2.push_back(Point(400, 117.1));
	ps2.push_back(Point(266.7,22.9));
	Mat h;
	double f0, f1;
	bool f0_ok, f1_ok;
	h = findHomography(ps1, ps2, RANSAC);	
	
	cs.focalsFromHomography(h, f0, f1, f0_ok, f1_ok);
	cout << f0 << " " << f1 << " " << f0_ok << " " << f1_ok << " " << sqrt(f0*f1) << endl;
	cout << h << endl;
	//cs.calFocus(img1, img2);
	waitKey(0);
}
