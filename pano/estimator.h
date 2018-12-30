#pragma once
#include <iostream>
#include <math.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d.hpp>
#include "opencv2/stitching/detail/autocalib.hpp"
#include "opencv2/stitching/detail/blenders.hpp"
#include "opencv2/stitching/detail/camera.hpp"
#include "opencv2/stitching/detail/exposure_compensate.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include "opencv2/stitching/detail/motion_estimators.hpp"
#include "opencv2/stitching/detail/seam_finders.hpp"
#include "opencv2/stitching/detail/util.hpp"
#include "opencv2/stitching/detail/warpers.hpp"
#include "opencv2/stitching/warpers.hpp"
#include "camera.h"
#include "matcher.h"
using namespace cv;
using namespace std;
using namespace detail;
class MyEstimator
{
public:
	MyEstimator()
	{
		sift = xfeatures2d::SIFT::create(1600);
	}

	void estimate2(vector<Mat> &imgs, vector<Camera> &cameras)
	{
		Mat t = Mat::eye(Size(3, 3), CV_64FC1);
		Mat t_inv = Mat::eye(Size(3, 3), CV_64FC1);
		t.at<double>(0, 2) = imgs[0].cols / 2;
		t.at<double>(1, 2) = imgs[0].rows / 2;
		t_inv.at<double>(0, 2) = -imgs[0].cols / 2;
		t_inv.at<double>(1, 2) = -imgs[0].rows / 2;
		double focal;
		vector<Mat> homos;
		Ptr<FeaturesFinder> finder;    //ÌØÕ÷¼ì²â
		finder = new SurfFeaturesFinder();
		vector<ImageFeatures> features(imgs.size());
		for (int i = 0; i < imgs.size(); i++) {
			(*finder)(imgs[i], features[i]);
		}
		vector<MatchesInfo> pairwise_matches;    //ÌØÕ÷Æ¥Åä
		BestOf2NearestMatcher matcher(false, 0.3f, 6, 6);
		matcher(features, pairwise_matches);
		for (int i = 1; i < imgs.size(); i++) {
			MatchesInfo m = pairwise_matches[i * imgs.size() + i - 1];			
			Mat h = m.H;
			cout << h << endl;
			cout << m.src_img_idx << " " << m.dst_img_idx << endl;
			homos.push_back(h);
		}
		focal = 660;
		for (int i = 0; i < imgs.size(); i++) {
			Camera c = Camera();
			c.focal = focal;
			c.setK();
			cameras.push_back(c);
		}
		for (int i = 1; i < imgs.size(); i++) {
			Mat h = homos[i - 1];			
			Mat K = cameras[i].K;

			Mat R = K.inv() * h * K;
			cameras[i].R = cameras[i-1].R * R;
		}
		for (int i = 0; i < imgs.size(); i++) {
			cameras[i].ppx += 0.5 * imgs[0].size().width;
			cameras[i].ppy += 0.5 * imgs[0].size().height;
			cameras[i].setK();
		}
		
	}

	void calR(int from,int to,int num_images,vector<Camera> &cameras, vector<MatchesInfo> &pairwise_matches)
	{
		int pair_idx = from * num_images + to;

		Mat_<double> K_from = Mat::eye(3, 3, CV_64F);
		K_from(0, 0) = cameras[from].focal;
		K_from(1, 1) = cameras[from].focal;
		K_from(0, 2) = cameras[from].ppx;
		K_from(1, 2) = cameras[from].ppy;

		Mat_<double> K_to = Mat::eye(3, 3, CV_64F);
		K_to(0, 0) = cameras[to].focal;
		K_to(1, 1) = cameras[to].focal;
		K_to(0, 2) = cameras[to].ppx;
		K_to(1, 2) = cameras[to].ppy;

		//Mat R = K_from.inv() * pairwise_matches[pair_idx].H.inv() * K_to;
		Mat R = K_from.inv() * pairwise_matches[to * num_images + from].H * K_to;
		cameras[to].R = cameras[from].R * R;
	}

	float getFocal(vector<Mat> &imgs)
	{
		double focal;
		vector<Mat> homos;
		Mat t = Mat::eye(Size(3, 3), CV_64FC1);
		Mat t_inv = Mat::eye(Size(3, 3), CV_64FC1);
		t.at<double>(0, 2) = imgs[0].cols / 2;
		t.at<double>(1, 2) = imgs[0].rows / 2;
		t_inv.at<double>(0, 2) = -imgs[0].cols / 2;
		t_inv.at<double>(1, 2) = -imgs[0].rows / 2;
		for (int i = 1; i < imgs.size(); i++) {
			Mat h = getHomography(imgs[i - 1], imgs[i]);
			Mat th = t_inv * h * t;
			homos.push_back(th);
		}
		estimateFocal(homos, focal);
		return focal;
	}

	void estimate(vector<Mat> &imgs,vector<Camera> &cameras)
	{
		double focal;
		vector<Mat> homos;
		Mat t = Mat::eye(Size(3, 3), CV_64FC1);
		Mat t_inv = Mat::eye(Size(3, 3), CV_64FC1);
		t.at<double>(0, 2) = imgs[0].cols / 2;
		t.at<double>(1, 2) = imgs[0].rows / 2;
		t_inv.at<double>(0, 2) = -imgs[0].cols / 2;
		t_inv.at<double>(1, 2) = -imgs[0].rows / 2;
		for (int i = 1; i < imgs.size(); i++) {			
			Mat h = getHomography(imgs[i - 1], imgs[i]);
			Mat th = t_inv * h * t;
			homos.push_back(th);
		}
		estimateFocal(homos, focal);
		//focal = 660;
		for (int i = 0; i < imgs.size(); i++) {
			Camera c = Camera();
			c.focal = focal;			
			c.setK();
			cameras.push_back(c);
		}
		for (int i = 1; i < imgs.size(); i++) {
			Mat h = homos[i - 1];
			Mat K = cameras[i].K;
			Mat R = K.inv() * h * K;
			//cout << "R from: " << R << endl;
			printf("R from%dto%d \n: ", i - 1, i);
			cout << R << endl;
			cameras[i].R = cameras[i - 1].R * R;
		}
		for (int i = 0; i < imgs.size(); i++) {
			cameras[i].ppx += 0.5 * imgs[0].size().width;
			cameras[i].ppy += 0.5 * imgs[0].size().height;
			cameras[i].setK();
		}
	}

	
	void estimateFocal(vector<Mat> &homos, double &focal)
	{
		vector<double> all_focals;
		double f0, f1;
		bool f0_ok, f1_ok;
		float err[2] = { 0.9, 1.1 };		
		for(int i = 0;i < homos.size();i++) {
			focalsFromHomography(homos[i], f0, f1, f0_ok, f1_ok);
			if (f0 / f1 <= err[1] && f0 / f1 >= err[0]) {
				all_focals.push_back(sqrt(f0*f1));
				cout << f0 << " " << f1 << " " << f0_ok << " " << f1_ok << " " << sqrt(f0*f1) << endl;
			}
		}
		sort(all_focals.begin(), all_focals.end());
		if (all_focals.size() % 2 == 0) {
			focal = all_focals[all_focals.size() / 2];
		}
		else {
			focal = (all_focals[all_focals.size() / 2 + 1] + all_focals[all_focals.size() / 2]) * 0.5;
		}
		cout << focal << endl;
	}

	void focalsFromHomography(const Mat& H, double &f0, double &f1, bool &f0_ok, bool &f1_ok)		
	{
		CV_Assert(H.type() == CV_64F && H.size() == Size(3, 3));
		//cout << H << endl;
		const double* h = reinterpret_cast<const double*>(H.data);    //¸³ÖµÖ¸Õë
		double d1, d2;
		double v1, v2;

		f1_ok = true;
		d1 = h[6] * h[7];    
		d2 = (h[7] - h[6]) * (h[7] + h[6]);   
		v1 = -(h[0] * h[1] + h[3] * h[4]) / d1;    
		v2 = (h[0] * h[0] + h[3] * h[3] - h[1] * h[1] - h[4] * h[4]) / d2;    
		if (v1 < v2) std::swap(v1, v2);    
		if (v1 > 0 && v2 > 0) f1 = sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
		else if (v1 > 0) f1 = sqrt(v1);    
		else f1_ok = false;    

		f0_ok = true;
		d1 = h[0] * h[3] + h[1] * h[4];    
		d2 = h[0] * h[0] + h[1] * h[1] - h[3] * h[3] - h[4] * h[4];    
		v1 = -h[2] * h[5] / d1;   
		v2 = (h[5] * h[5] - h[2] * h[2]) / d2;    
		if (v1 < v2) std::swap(v1, v2);    
		if (v1 > 0 && v2 > 0) f0 = sqrt(std::abs(d1) > std::abs(d2) ? v1 : v2);
		else if (v1 > 0) f0 = sqrt(v1);   
		else f0_ok = false;    
	}

	Mat getHomography(Mat &img1, Mat& img2)
	{
		vector<KeyPoint> kps1, kps2;
		Mat descriptors1, descriptors2;
		Mat imgMatch;;
		sift->detectAndCompute(img1, Mat(), kps1, descriptors1);
		sift->detectAndCompute(img2, Mat(), kps2, descriptors2);
		vector<DMatch> matches;
		MyMatcher mymatcher;
		mymatcher.KDmatch(descriptors1, descriptors2, matches);
		//matcher.match(descriptors1, descriptors2, matches, Mat());
		sort(matches.begin(), matches.end());
		const int numGoodMatches = matches.size() * 0.15;
		//matches.erase(matches.begin() + numGoodMatches, matches.end());
		Mat h;
		std::vector<Point2f> points1, points2;
		for (size_t i = 0; i < matches.size(); i++)
		{
			points1.push_back(kps1[matches[i].queryIdx].pt);
			points2.push_back(kps2[matches[i].trainIdx].pt);
		}		
		vector<uchar> inlineMask;
		vector<DMatch> matches1, matches2;
		int inlineNum = 0;
		float confidence;
		h = findHomography(points2, points1, inlineMask, RANSAC);
		points1.clear();
		points2.clear();
		for (int i = 0; i < inlineMask.size(); i++)
		{
			if (inlineMask[i]) {
				inlineNum++;
				matches1.push_back(matches[i]);				
				points1.push_back(kps1[matches[i].queryIdx].pt);
				points2.push_back(kps2[matches[i].trainIdx].pt);
			}
		}		
		drawMatches(img1, kps1, img2, kps2, matches1, imgMatch);
		/*imshow("match1", imgMatch);
		waitKey(0);*/
		if (inlineMask.size() > 0) {
			confidence = inlineNum / (8 + 0.3 * inlineMask.size());			
		}
		else {
			confidence = 0;
		}
		cout << confidence;	
		inlineMask.clear();
		inlineNum = 0;
		h = findHomography(points2, points1, inlineMask, RANSAC);

		for (int i = 0; i < inlineMask.size(); i++)
		{
			if (inlineMask[i]) {
				matches2.push_back(matches1[i]);
				inlineNum++;
			}
		}
		if (inlineMask.size() > 0) {
			confidence = inlineNum / (8 + 0.3 * inlineMask.size());
		}
		else {
			confidence = 0;
		}
		cout << " " << confidence << endl;
		
		drawMatches(img1, kps1, img2, kps2, matches2, imgMatch);
		/*imshow("match2", imgMatch);
		waitKey(0);*/
		//waitKey(0);
		/*imshow("1", img);
		waitKey(0);*/
		//cout << h << endl;
		return h;
	}
protected:
	Ptr<Feature2D> sift;
	BFMatcher matcher;

};