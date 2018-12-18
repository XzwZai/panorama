#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>

using namespace cv;
using namespace std;
class CylinderStitcher
{
public:	
	float err;
	CylinderStitcher()
	{
		sift = xfeatures2d::SIFT::create(1600);		
		err = 1.0 / 600;
	}

	bool inImg(Point p, int width, int height)
	{
		return p.x < width && p.x >= 0 && p.y < height && p.y >= 0;
	}

	Mat getHomography(Mat &img1,Mat& img2)
	{
		vector<KeyPoint> kps1, kps2;
		Mat descriptors1, descriptors2;
		sift->detectAndCompute(img1, Mat(), kps1, descriptors1);
		sift->detectAndCompute(img2, Mat(), kps2, descriptors2);
		vector<DMatch> matches, matches1;
		matcher.match(descriptors1, descriptors2, matches, Mat());
		sort(matches.begin(), matches.end());
		const int numGoodMatches = matches.size() * 0.15;
		matches.erase(matches.begin() + numGoodMatches, matches.end());
		Mat h;
		std::vector<Point2f> points1, points2;
		for (size_t i = 0; i < matches.size(); i++)
		{
			points1.push_back(kps1[matches[i].queryIdx].pt);
			points2.push_back(kps2[matches[i].trainIdx].pt);
		}		
		h = findHomography(points2, points1, RANSAC);
		Mat img;
		drawMatches(img1, kps1, img2, kps2, matches, img);
		//imshow("1", img);
		//cout << h << endl;
		return h;
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

	/*Mat getHomography(Mat& img1,Mat& img2)
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
				cout << abs(kps1[(*it).queryIdx].pt.y - kps2[(*it).trainIdx].pt.y) << " " << kps1[(*it).queryIdx].pt.x - kps2[(*it).trainIdx].pt.x << endl;
				totalTrans += kps1[(*it).queryIdx].pt.x - kps2[(*it).trainIdx].pt.x;
				matches1.push_back(*it);
			}
		}
		horizonTrans = totalTrans / matches1.size();
		Mat h = Mat::eye(Size(3, 3), CV_64FC1);
		h.at<double>(0, 2) = horizonTrans;
		Mat imgAligned;		
		warpPerspective(img2, imgAligned, h, Size(img2.size().width + horizonTrans,img2.size().height));
		imshow("aligned", imgAligned);
		Mat imgStitched;
		imgAligned.copyTo(imgStitched);
		img1.copyTo(imgStitched(Rect(0, 0, img1.cols, img1.rows)));
		imshow("stitched", imgStitched);
		Mat imgMatch;
		drawMatches(img1, kps1, img2, kps2, matches1, imgMatch);
		imshow("match", imgMatch);
		waitKey(0);
		return img2;
	}*/

	void focalsFromHomography(const Mat& H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)
		//H表示单应矩阵
		//f0和f1分别表示单应矩阵H所转换的两幅图像的焦距
		//f0_ok和f1_ok分别表示f0和f1是否评估正确
	{
		//确保H的数据类型和格式正确
		CV_Assert(H.type() == CV_64F && H.size() == Size(3, 3));
		cout << H << endl;
		const double* h = reinterpret_cast<const double*>(H.data);    //赋值指针
		//分别表示式43和式44，或式47和式48的分母
		double d1, d2; // Denominators
		//分别表示式43和式44，或式47和式48
		double v1, v2; // Focal squares value candidates

		f1_ok = true;
		d1 = h[6] * h[7];    //式48的分母
		d2 = (h[7] - h[6]) * (h[7] + h[6]);    //式47的分母
		v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;    //式48
		v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;    //式47
		if (v1 < v2) std::swap(v1, v2);    //使v1不小于v2
		//表示到底选取式47还是式48作为f1
		if (v1 > 0 && v2 > 0) f1 = sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
		else if (v1 > 0) f1 = sqrt(v1);    //v2小于0，则f1一定是v1的平方根
		else f1_ok = false;    //v1和v2都小于0，则没有得到f1

		f0_ok = true;
		d1 = h[0] * h[3] + h[1] * h[4];    //式44的分母
		d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];    //式43的分母
		v1 = -h[2] * h[5] / d1;    //式44
		v2 = (h[5] * h[5] - h[2] * h[2]) / d2;    //式43
		if (v1 < v2) std::swap(v1, v2);    //使v1不小于v2
		//表示到底选取式44还是式43作为f0
		if (v1 > 0 && v2 > 0) f0 = sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
		else if (v1 > 0) f0 = sqrt(v1);    //v2小于0，则f1一定是v1的开根号
		else f0_ok = false;    //v1和v2都小于0，则没有得到f1
	}


protected:
	Ptr<Feature2D> sift;
	BFMatcher matcher;
};

class IdealCylinderStitcher : public CylinderStitcher
{
public:
	IdealCylinderStitcher()
		: CylinderStitcher() {}

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