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


Size imgSize = Size(640, 960);

void testMyMatcher()
{
	Mat img1 = imread("D:/WorkSpace/data/pano/Synthetic/Synthetic/img01.jpg");
	Mat img2 = imread("D:/WorkSpace/data/pano/Synthetic/Synthetic/img02.jpg");
	resize(img1, img1, imgSize);
	resize(img2, img2, imgSize);
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create(1600);
	Mat descriptors1, descriptors2;
	vector<KeyPoint> kps1, kps2;
	Mat imgMatch;
	sift->detectAndCompute(img1, Mat(), kps1, descriptors1);
	sift->detectAndCompute(img2, Mat(), kps2, descriptors2);
	MyMatcher matcher;
	vector<DMatch> matches;
	matcher.KDmatch(descriptors1, descriptors2,matches);
	vector<Point2f> points1, points2;
	for (size_t i = 0; i < matches.size(); i++)
	{
		points1.push_back(kps1[matches[i].queryIdx].pt);
		points2.push_back(kps2[matches[i].trainIdx].pt);
	}
	drawMatches(img1, kps1, img2, kps2, matches, imgMatch);
	imshow("m", imgMatch);
	waitKey(0);
	//matcher.match(descriptors1, descriptors2);

}

int main(int argc, char** argv)
{
	
	//testMyMatcher();
	ifstream files("D:/WorkSpace/data/pano/data2/imgs.txt");	
	//ifstream files("D:/WorkSpace/data/pano/Synthetic/imgs.txt");
	string imgpath;
	vector<Mat> imgs;
	CylinderStitcher cs;
	IdealCylinderStitcher ics;
	
	while (files >> imgpath)
	{
		Mat img = imread(imgpath);
		resize(img, img, imgSize);		
		imgs.push_back(img);
	}
	cout << "read imgs" << endl;	
	ics.stitch(imgs);
	
}
