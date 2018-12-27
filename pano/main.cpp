#include <iostream>

#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "cylinder.h"
#include "wraper.h"
#include "kdtree.h"
#include "matcher.h"
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
vector<Point2f> ps1, ps2;
void on_mouse(int EVENT, int x, int y, int flags, void* userdata) { 
	Mat hh; hh = *(Mat*)userdata; Point2f p(x, y); 
	switch (EVENT) { 
		case EVENT_LBUTTONDOWN: { 
			ps1.push_back(p);
			cout << "e" << endl;
			/*printf("b=%d\t", hh.at<Vec3b>(p)[0]); 
			printf("g=%d\t", hh.at<Vec3b>(p)[1]); 
			printf("r=%d\n", hh.at<Vec3b>(p)[2]); */
			circle(hh, p, 2, Scalar(255), 3); } 
		break; 
	} 
}
void on_mouse2(int EVENT, int x, int y, int flags, void* userdata) {
	Mat hh; hh = *(Mat*)userdata; Point p(x, y);
	switch (EVENT) {
	case EVENT_LBUTTONDOWN: {
		ps2.push_back(p);
		cout << "e" << endl;
		/*printf("b=%d\t", hh.at<Vec3b>(p)[0]);
		printf("g=%d\t", hh.at<Vec3b>(p)[1]);
		printf("r=%d\n", hh.at<Vec3b>(p)[2]); */
		circle(hh, p, 2, Scalar(255), 3); }
							break;
	}
}

void testMyMatcher()
{
	Mat img = 
}

int main(int argc, char** argv)
{
	
	ifstream files("D:/WorkSpace/data/pano/Synthetic/imgs.txt");
	//ifstream files("C:/Users/China/Downloads/example-data/example-data/zijing/imgs.txt");
	string imgpath;
	vector<Mat> imgs;
	CylinderStitcher cs;
	IdealCylinderStitcher ics;
	//while (files >> imgpath)
	//{
	//	Mat img = imread(imgpath);
	//	resize(img, img, imgSize);		
	//	imgs.push_back(img);
	//}
	//cout << "read imgs" << endl;
	////cs.stitch(imgs);
	//ics.stitch(imgs, f);
	/*Mat mat = (Mat_<double>(6, 2) << 2, 3, 4, 7, 5, 4, 9, 6, 8, 1, 7, 2);
	Kdtree kdtree(mat);
	feature* f = (feature*)calloc(1,sizeof(feature));
	f->descr = (double*)calloc(2, sizeof(double));
	f->descr[0] = 5.5; f->descr[1] = 5;
	vector<bfeature> result = kdtree.kdtree_bbf_knn(f,6);
	for (int i = 0; i < result.size(); i++)
	{
		cout << result[i].d << " ";
		for (int j = 0; j < 2; j++)
		{
			cout << result[i].feat->descr[j] << " ";
		}
		cout << endl;
	}*/
	
}
