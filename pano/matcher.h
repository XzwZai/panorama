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

	void KDmatch(Mat des1,Mat des2)
	{
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
			for (int j = 0; j < 2; j++) {
				cout << result[i].d << " " << result[i].feat->id << ";";
			}
			cout << endl;
		}



	}
};