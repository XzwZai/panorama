#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "estimator.h"
#include "wraper.h"
using namespace cv;
using namespace std;
class CylinderStitcher
{
public:	
	
	CylinderStitcher()
	{
		sift = xfeatures2d::SIFT::create(1600);				
	}

	bool inImg(Point p, int width, int height)
	{
		return p.x < width && p.x >= 0 && p.y < height && p.y >= 0;
	}	
	
	Mat cylindrical(Mat& imgIn, int f,int mode) {
		if (mode == 1)
		{
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
		else if (mode == 2)
		{
			int colNum; colNum = 2 * f*atan(0.5*imgIn.cols / f);
			Mat imgOut = Mat::zeros(imgIn.size(), imgIn.type());
			int width(imgIn.cols), height(imgIn.rows);
			for (int i = 0; i < height; i++)
			{
				for (int j = 0; j < width; j++)
				{
					float x = f * tan((float)(j - width / 2) / f) + width / 2;
					float alpha = atan((x - width / 2) / f);
					float y = (i - height / 2) / cos(alpha) + height / 2;
					vector<Point> nears;
					nears.push_back(Point(int(x), int(y)));
					nears.push_back(Point(int(x) + 1, int(y)));
					nears.push_back(Point(int(x), int(y) + 1));
					nears.push_back(Point(int(x) + 1, int(y) + 1));
					float totalDis(0);
					//cout << x << " " << y << endl;
					if (inImg(Point(x, y), width, height))
					{
						for (int k = 0; k < nears.size(); k++)
						{
							if (inImg(nears[k], width, height))
							{
								totalDis += norm(nears[k] - Point(x, y));
							}
						}
						Vec3b color(0, 0, 0);
						for (int k = 0; k < nears.size(); k++)
						{
							if (inImg(nears[k], width, height))
							{
								color += imgIn.at<Vec3b>(nears[k]) * norm(nears[k] - Point(x, y)) / totalDis;
							}
						}
						imgOut.at<Vec3b>(i, j) = color;
					}
					/*if (i == height / 2 && j == width / 2)
					{
						cout << x << " " << y << endl;
					}*/

				}
			}
			imgOut = imgOut(Rect((imgIn.cols - colNum) / 2, 0, colNum, imgIn.rows));
			return imgOut;
		}		
	}

	Mat getHomography(vector<Point> ps1, vector<Point> ps2)
	{

	}	

	void stitch(vector<Mat> &imgs)
	{
		double focal;
		MyEstimator estimator;
		vector<Camera> cameras;
		estimator.estimate(imgs, cameras);
		/*for (int i = 0; i < cameras.size(); i++) {
			cout << cameras[i].K << endl;
			cout << cameras[i].R << endl;
		}*/
		vector<Point> cornors(imgs.size());
		vector<Mat> images_warped(imgs.size());
		MyWarper warper(660);
		for (int i = 0; i < imgs.size(); i++)
		{
			Mat_<float> K,R;
			cameras[i].K.convertTo(K, CV_32F);
			cameras[i].R.convertTo(R, CV_32F);
			cout << K << endl;
			cout << R << endl;
			cornors[i] = warper.warp(imgs[i], K, R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
			cout << cornors[i] << endl;
			imshow("1", images_warped[i]);
			waitKey(0);
		}
	}

	


protected:
	Ptr<Feature2D> sift;
	BFMatcher matcher;
};

class IdealCylinderStitcher : public CylinderStitcher
{
public:
	float err;
	IdealCylinderStitcher()
		: CylinderStitcher() 
	{
		err = 1.0f / 600;
	}

	struct IMG
	{
		Mat img;
		Mat homo;
		IMG(Mat _img, Mat _homo) { img = _img; homo = _homo; }
	};

	Mat stitch(vector<Mat> imgs,int f)
	{		
		for (int i = 0; i < imgs.size(); i++)
		{
			imgs[i] = cylindrical(imgs[i], f, 1);
		}
		cout << "cylindrical" << endl;
		vector<IMG> IMGs;
		Mat homo = Mat::eye(Size(3, 3), CV_64FC1);	
		Mat h;
		homo.copyTo(h);
		IMG img(imgs[0], h);
		IMGs.push_back(img);
		for (int i = 1; i < imgs.size(); i++)
		{
			Mat h = homo * getHomography(imgs[i - 1], imgs[i]);
			
			IMGs.push_back(IMG(imgs[i], h));
			h.copyTo(homo);
		}
		cout << "homo computed" << endl;
		Size finalSize = Size(IMGs[0].img.cols + homo.at<double>(0, 2), IMGs[0].img.rows);
		Mat stitch = Mat::zeros(finalSize, CV_8UC3);
		for (int i = 0; i < IMGs.size(); i++)
		{
			Rect r = Rect(IMGs[i].homo.at<double>(0, 2), 0, IMGs[i].img.cols, IMGs[i].img.rows);
			IMGs[i].img.copyTo(stitch(r));
			/*imshow("1", stitch);
			waitKey(0);*/
		}		
		//imshow("stitch", stitch);
		return stitch;
	}

	Mat getHomography(Mat& img1, Mat& img2)
	{
		vector<KeyPoint> kps1, kps2;
		Mat descriptors1, descriptors2;
		sift->detectAndCompute(img1, Mat(), kps1, descriptors1);
		sift->detectAndCompute(img2, Mat(), kps2, descriptors2);
		vector<DMatch> matches, matches1;
		matcher.match(descriptors1, descriptors2, matches, Mat());
		float horizonTrans, totalTrans(0);
		for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++)
		{
			if (abs(kps1[(*it).queryIdx].pt.y - kps2[(*it).trainIdx].pt.y) < err * img1.rows)
			{
				//cout << abs(kps1[(*it).queryIdx].pt.y - kps2[(*it).trainIdx].pt.y) << " " << kps1[(*it).queryIdx].pt.x - kps2[(*it).trainIdx].pt.x << endl;
				totalTrans += kps1[(*it).queryIdx].pt.x - kps2[(*it).trainIdx].pt.x;
				matches1.push_back(*it);
			}
		}
		horizonTrans = totalTrans / matches1.size();
		Mat h = Mat::eye(Size(3, 3), CV_64FC1);
		h.at<double>(0, 2) = horizonTrans;		
		return h;
	}

private:

};