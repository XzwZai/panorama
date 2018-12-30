#pragma once
#include <opencv2/opencv.hpp>
#include "kdtree.h"
using namespace cv;
class MyMatcher
{
public:	
	MyMatcher()
	{
		
	}

	struct MatchResult
	{
		int queryIdx;
		int trainIdx;
		double distance;

	};

	void KDmatch(Mat des1,Mat des2,vector<DMatch> &matches)
	{
		des1.convertTo(des1, CV_64F);
		des2.convertTo(des2, CV_64F);		
		Kdtree kdtree(des1);
		int dimen = des1.cols;
		for (int i = 0; i < des2.rows; i++)
		{
			feature* feat = (feature*)calloc(1,sizeof(feature));
			feat->descr = (double*)calloc(dimen, sizeof(double));
			for (int j = 0; j < dimen; j++) {
				feat->descr[j] = des2.at<double>(i, j);				
			}
			vector<bfeature> result = kdtree.kdtree_bbf_knn(feat);
			if (result[0].d / result[1].d < 0.49)
			{
				DMatch dm(result[0].feat->id, i, result[0].d);
				matches.push_back(dm);				
			}
			/*cout << i << ": ";
			for (int j = 0; j < 2; j++) {
				cout << result[j].d << " " << result[j].feat->id << ";";
			}
			cout << endl;*/
		}		

	}

	double calDis(double* des1, double* des2,int dimen)
	{
		double dis = 0;
		for (int i = 0; i < dimen; i++)
		{
			double x = des1[i] - des2[i];
			dis += x * x;
		}
		return dis;
	}

	void match(Mat des1, Mat des2)
	{
		des1.convertTo(des1, CV_64F);
		des2.convertTo(des2, CV_64F);
		int dimen = des1.cols;
		vector<feature> fs;
		for (int i = 0; i < des1.rows; i++)
		{
			feature f;
			f.id = i;
			f.descr = (double*)calloc(dimen, sizeof(double));
			for (int j = 0; j < dimen; j++)
			{
				f.descr[j] = des1.at<double>(i, j);
			}
			fs.push_back(f);
		}
		for (int i = 0; i < des2.rows; i++)
		{
			bfeature min1(10000000,NULL), min2(10000000,NULL);

			double minD1 = 100000000, minD2 = 100000000;
			double* descr = (double*)calloc(dimen, sizeof(double));;
			for (int j = 0; j < dimen; j++) {
				descr[j] = des2.at<double>(i, j);
			}
			for (int j = 0; j < fs.size(); j++)
			{
				double dis = calDis(descr, fs[j].descr, dimen);
				if (dis < min2.d)
				{
					min2.d = dis;
					min2.feat = &fs[j];					
				}
				if (min2.d < min1.d)
				{
					bfeature tmp = min2;
					min2 = min1;
					min1 = tmp;					
				}
			}
			cout << min1.d << " " << min1.feat->id << " ; " << min2.d << " " << min2.feat->id << endl;			
		}
	}
};