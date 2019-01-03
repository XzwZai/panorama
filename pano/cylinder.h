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
#include "matcher.h"

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
			Mat imgOut = Mat::zeros(rowNum, colNum, imgIn.type()); 
			//imgOut.setTo(255);
			int x1(0), y1(0);
			for (int i = 0; i < imgIn.rows; i++)
				for (int j = 0; j < imgIn.cols; j++) {
					x1 = f * atan((j - 0.5*imgIn.cols) / f) + f * atan(0.5*imgIn.cols / f);
					y1 = f * (i - 0.5*imgIn.rows) / sqrt(pow(j - 0.5*imgIn.cols, 2) + pow(f, 2)) + 0.5*imgIn.rows;
					if (x1 >= 0 && x1 < colNum&&y1 >= 0 && y1 < rowNum) {
						//im2(y1, x1) = im1(i, j); 
						if (imgIn.type() == CV_8UC1)
						{
							imgOut.at<uchar>(y1, x1) = imgIn.at<uchar>(i, j);
						}
						else if (imgIn.type() == CV_8UC3)
						{
							imgOut.at<Vec3b>(y1, x1) = imgIn.at<Vec3b>(i, j);
						}
					}
				}
			/*Mat imgOut;
			imgOut.create(imgIn.size(), CV_8UC3);
			imgOut.setTo(255);
			int x, y;
			for (int i = 0; i < imgIn.rows; i++) 
			{
				for (int j = 0; j < imgIn.cols; j++)
				{
					
				}
			}*/
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
				}
			}
			imgOut = imgOut(Rect((imgIn.cols - colNum) / 2, 0, colNum, imgIn.rows));
			return imgOut;
		}		
	}	

	void stitch(vector<Mat> &imgs)
	{
		double focal;
		MyEstimator estimator;
		vector<Camera> cameras;
		estimator.estimate(imgs, cameras);
		cout << "focal : " << cameras[0].focal << endl;
		/*for (int i = 0; i < cameras.size(); i++) {
			cout << cameras[i].K << endl;
			cout << cameras[i].R << endl;
		}*/
		vector<Point> cornors(imgs.size());
		vector<Mat> images_warped(imgs.size());
		vector<Mat> masks_warped(imgs.size());
		MyWarper warper(cameras[0].focal);
		for (int i = 0; i < imgs.size(); i++)
		{
			Mat_<float> K,R;
			cameras[i].K.convertTo(K, CV_32F);
			cameras[i].R.convertTo(R, CV_32F);
			//cout << K << endl;
			printf("img%d :\n", i);
			cout << R << endl;
			
			cornors[i] = warper.warp(imgs[i], K, R, INTER_LINEAR, BORDER_REFLECT, images_warped[i]);
			Mat mask, mask_warped;
			mask.create(imgs[i].size(), CV_8U);
			mask.setTo(Scalar::all(255));
			warper.warp(mask, K, R, INTER_LINEAR, BORDER_CONSTANT, masks_warped[i]);
			cout << cornors[i] << endl;
			//imshow("1", images_warped[i]);
			Mat img_mask;
			images_warped[i].copyTo(img_mask, masks_warped[i]);
			//imshow("2", img_mask);
			//waitKey(0);
		}
		vector<Point> ps;
		for (int i = 0; i < cornors.size(); i++) {
			ps.push_back(cornors[i]);
			Point br;
			br.x = cornors[i].x + images_warped[i].size().width;
			br.y = cornors[i].y + images_warped[i].size().height;
			ps.push_back(br);
		}
		Rect bound = boundingRect(ps);
		Mat result = Mat::zeros(bound.size(), CV_8UC3);
		for (int i = 0; i < imgs.size(); i++)
		{
			Point cornor = cornors[i];
			cornor.x -= bound.tl().x;
			cornor.y -= bound.tl().y;
			Rect r(cornor.x, cornor.y, images_warped[i].size().width, images_warped[i].size().height);
			images_warped[i].copyTo(result(r), masks_warped[i]);
		}
		imshow("result", result);
		imwrite("result.jpg", result);
		waitKey(0);
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

	Mat stitch(vector<Mat> imgs, float f = 800)
	{		
		MyEstimator estimator;				
		f = estimator.getFocal(imgs);
		cout << "focal: " << f << endl;
		vector<Mat> masks_warped;
		for (int i = 0; i < imgs.size(); i++)
		{
			Mat mask;
			mask.create(imgs[i].size(), CV_8UC1);
			mask.setTo(Scalar::all(255));
			mask = cylindrical(mask, f, 1);
			masks_warped.push_back(mask);
			imgs[i] = cylindrical(imgs[i], f, 1);
		}
		cout << "cylindrical" << endl;
		vector<IMG> IMGs;
		vector<Point> cornors;
		Mat homo = Mat::eye(Size(3, 3), CV_64FC1);	
		Mat h;
		homo.copyTo(h);
		IMG img(imgs[0], h);
		IMGs.push_back(img);
		Point cuCornor(0,0);
		cornors.push_back(Point(0, 0));
		for (int i = 1; i < imgs.size(); i++)
		{
			Point cornor = cuCornor + getTrans(imgs[i - 1], imgs[i]);	
			cornors.push_back(cornor);
			//IMGs.push_back(IMG(imgs[i], h));
			cuCornor.x = cornor.x;
			cuCornor.y = cornor.y;
		}
		cout << "homo computed" << endl;
		vector<Point> ps;
		for (int i = 0; i < cornors.size(); i++) {
			ps.push_back(cornors[i]);
			Point br;
			br.x = cornors[i].x + imgs[i].size().width;
			br.y = cornors[i].y + imgs[i].size().height;
			ps.push_back(br);
		}
		Rect bound = boundingRect(ps);
		Mat result = Mat::zeros(bound.size(), CV_8UC3);
		int right;

		for (int i = 0; i < imgs.size(); i++)
		{
			Point cornor = cornors[i];
			cornor.x -= bound.tl().x;
			cornor.y -= bound.tl().y;
			Rect rect(cornor.x, cornor.y, imgs[i].size().width, imgs[i].size().height);
			if (i == 0)
			{
				imgs[i].copyTo(result(rect), masks_warped[i]);
				right = rect.x + rect.width;
			}
			else {
				for (int r = 0; r < masks_warped[i].rows; r++)
				{
					for (int c = 0; c < masks_warped[i].cols; c++) 
					{
						if (masks_warped[i].at<uchar>(r, c) != 0) {
							if (c + rect.x < right) {
								float d = c * 1.0f / (right - rect.x);
								Vec3b color = (1 - d)*result.at<Vec3b>(rect.y + r, rect.x + c) + d * imgs[i].at<Vec3b>(r, c);
								result.at<Vec3b>(rect.y + r, rect.x + c) = color;
							}
							else {
								Vec3b color = imgs[i].at<Vec3b>(r, c);
								result.at<Vec3b>(rect.y + r, rect.x + c) = color;
							}
						}
					}
				}
				right = rect.x + rect.width;
			}
		}
		imshow("stitch", result);
		waitKey(0);
		imwrite("idealcs.jpg", result);
		return result;
		
	}

	Mat getHoriHomography(Mat& img1, Mat& img2)
	{
		vector<KeyPoint> kps1, kps2;
		Mat descriptors1, descriptors2;
		sift->detectAndCompute(img1, Mat(), kps1, descriptors1);
		sift->detectAndCompute(img2, Mat(), kps2, descriptors2);
		vector<DMatch> matches, matches1;
		MyMatcher mymatcher;
		mymatcher.KDmatch(descriptors1, descriptors2, matches);
		Mat img;
		drawMatches(img1, kps1, img2, kps2, matches, img);
		imshow("1", img);
		waitKey(0);
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

	Point getTrans(Mat& img1, Mat& img2)
	{
		vector<KeyPoint> kps1, kps2;
		Mat descriptors1, descriptors2;
		sift->detectAndCompute(img1, Mat(), kps1, descriptors1);
		sift->detectAndCompute(img2, Mat(), kps2, descriptors2);
		vector<DMatch> matches, matches1;
		MyMatcher mymatcher;
		mymatcher.KDmatch(descriptors1, descriptors2, matches);
		//cout << "match" << endl;
		/*Mat img;
		drawMatches(img1, kps1, img2, kps2, matches, img);
		imshow("1", img);
		waitKey(0);*/
		int maxT = 20;
		float err = 0.1;
		int maxinlinenum = 0;
		vector<int> maxinlineindexs;
		for (int t = 0; t < maxT; t++)
		{
			int choose = rand() % matches.size();
			int xoff = kps1[matches[choose].queryIdx].pt.x - kps2[matches[choose].trainIdx].pt.x;
			int yoff = kps1[matches[choose].queryIdx].pt.y - kps2[matches[choose].trainIdx].pt.y;
			vector<int> indexs;
			int inlinenum = 0;
			int index = 0;
			for (vector<DMatch>::iterator it = matches.begin(); it != matches.end(); it++)
			{
				int txoff = kps1[(*it).queryIdx].pt.x - kps2[(*it).trainIdx].pt.x;
				int tyoff = kps1[(*it).queryIdx].pt.y - kps2[(*it).trainIdx].pt.y;
				int xerr = txoff - xoff;
				int yerr = tyoff - yoff;
				if (sqrt(xerr*xerr + yerr * yerr) < err * sqrt(xoff*xoff + yoff * yoff))
				{
					inlinenum++;
					indexs.push_back(index);
				}
				index++;
			}
			if (inlinenum > maxinlinenum)
			{
				maxinlinenum = inlinenum;
				maxinlineindexs = indexs;
			}
			//cout << t << " ";
		}
		int totalxoff = 0,totalyoff = 0;
		for (int i = 0; i < maxinlinenum; i++)
		{
			int index = maxinlineindexs[i];
			totalxoff += kps1[matches[index].queryIdx].pt.x - kps2[matches[index].trainIdx].pt.x;
			totalyoff += kps1[matches[index].queryIdx].pt.y - kps2[matches[index].trainIdx].pt.y;
		}
		cout << totalxoff / maxinlinenum << ", " << totalyoff / maxinlinenum <<endl;
		Point p(totalxoff / maxinlinenum, totalyoff / maxinlinenum);
		return p;
	}

private:

};