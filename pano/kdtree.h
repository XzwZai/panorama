#pragma once
#include <vector>
#include <algorithm>
#include <iostream>

using namespace std;
struct feature
{
	int id;
	double* descr;
};

struct kd_node
{
	int ki;
	int kv;
	int leaf;
	feature* features;
	int n;
	kd_node* kd_left;
	kd_node* kd_right;
};

struct bbf
{
	double d;
	kd_node* node;
	bbf(double _d, kd_node* _node) { d = _d; node = _node; }
	bool operator<(const bbf& b) const
	{
		return d > b.d;
	}
};

struct bfeature
{
	double d;
	feature* feat;
	bfeature(double _d, feature* _f) { d = _d; feat = _f; }
	bool operator<(const bfeature& b) const
	{
		return d > b.d;
	}
};


class Kdtree
{
public:
	int dimen_cur;
	int D;
	kd_node* kd_root;
	
	Kdtree(Mat mat)
	{
		int id = 0;
		feature* features;
		mat.convertTo(mat, CV_64F);
		D = mat.cols;
		int n = mat.rows;
		features = (feature*)calloc(n, sizeof(feature));
		for (int i = 0; i < n; i++)
		{		
			double *data = (double*)calloc(D, sizeof(double));
			for (int j = 0; j < D; j++)
			{
				data[j] = mat.at<double>(i, j);
			}
			features[i].id = id;
			features[i].descr = data;
			id++;
		}
		/*for (int i = 0; i < n; i++) {
			for (int j = 0; j < D; j++) {
				cout << features[i].descr[j] << " ";
			}
			cout << endl;
		}*/
		kd_root = kdtree_build(features, n);
		
	}



	bool feature_compare(feature &f1, feature &f2)
	{
		return (f1.descr[dimen_cur] < f2.descr[dimen_cur]);
	}

	kd_node* kdtree_build(feature* features, int n)
	{
		if (n == 0) {
			return NULL;
		}
		kd_node* kd_root;
		kd_root = kd_node_init(features, n);
		expand_kd_node_subtree(kd_root);
		return kd_root;
	}

	kd_node* kd_node_init(feature* features, int n)
	{
		kd_node* node;
		node = (kd_node*)calloc(1, sizeof(kd_node));
		node->ki = -1;
		node->n = n;
		node->features = features;
		return node;
	}

	void expand_kd_node_subtree(kd_node* node)
	{
		if (node->n == 1 || node->n == 0) {
			node->leaf = 1;
			return;
		}
		assign_part_key(node);
		partition_features(node);
		if (node->kd_left)
		{
			expand_kd_node_subtree(node->kd_left);
		}
		if (node->kd_right)
		{
			expand_kd_node_subtree(node->kd_right);
		}
	}

	void assign_part_key(kd_node* node)
	{
		feature* features;	
		double kv, x, mean, var, var_max = 0;
		double* tmp;
		int d, n, i, j, ki = 0; //枢轴索引ki

		features = node->features;
		n = node->n;
		d = D;
		
		for (j = 0; j < d; j++)
		{
			mean = var = 0;
			for (i = 0; i < n; i++)
				mean += features[i].descr[j];
			mean /= n;			
			for (i = 0; i < n; i++)
			{
				x = features[i].descr[j] - mean;
				var += x * x;
			}
			var /= n;
			if (var > var_max)
			{
				ki = j;
				var_max = var;
			}
		}
		tmp = (double*)calloc(n, sizeof(double));
		for (i = 0; i < n; i++)
			tmp[i] = features[i].descr[ki];
		kv = median_select(tmp, n);
		free(tmp);
		node->ki = ki;
		node->kv = kv;
	}

	double median_select(double* arr, int n)
	{
		sort(arr, arr + n);
		return arr[(n - 1) / 2];
		//return rank_select(arr, n, (n - 1) / 2);
	}

	void partition_features(struct kd_node* node)
	{
		struct feature* features, tmp;
		double kv;
		int n, ki, p, i, j = -1;

		features = node->features;
		n = node->n;
		//printf("%d\n",n);
		ki = node->ki;
		kv = node->kv;
		for (i = 0; i < n; i++)
		{
			if (features[i].descr[ki] <= kv)
			{
				tmp = features[++j];
				features[j] = features[i];
				features[i] = tmp;
				if (features[j].descr[ki] == kv)
					p = j;
			}
		}
		tmp = features[p];
		features[p] = features[j];
		features[j] = tmp;
		
		if (j == n - 1)
		{
			node->leaf = 1;
			return;
		}

		node->kd_left = kd_node_init(features, j + 1);
		node->kd_right = kd_node_init(features + (j + 1), (n - j - 1));
	}

	double rank_select(double* array, int n, int r)
	{
		double* tmp, med;
		int gr_5, gr_tot, rem_elts, i, j;

		/* base case */
		if (n == 1)
			return array[0];

		//将数组分成5个一组，共gr_tot组
		/* divide array into groups of 5 and sort them */
		gr_5 = n / 5; //组的个数-1，n/5向下取整
		gr_tot = cvCeil(n / 5.0); //组的个数，n/5向上取整
		rem_elts = n % 5;//最后一组中的元素个数
		tmp = array;
		//对每组进行插入排序
		for (i = 0; i < gr_5; i++)
		{
			insertion_sort(tmp, 5);
			tmp += 5;
		}
		//最后一组
		insertion_sort(tmp, rem_elts);

		//找中值的中值
		/* recursively find the median of the medians of the groups of 5 */
		tmp = (double*)calloc(gr_tot, sizeof(double));
		//将每个5元组中的中值(即下标为2,2+5,...的元素)复制到temp数组
		for (i = 0, j = 2; i < gr_5; i++, j += 5)
			tmp[i] = array[j];
		//最后一组的中值
		if (rem_elts)
			tmp[i++] = array[n - 1 - rem_elts / 2];
		//找temp中的中值med，即中值的中值
		med = rank_select(tmp, i, (i - 1) / 2);
		free(tmp);

		//利用中值的中值划分数组，看划分结果是否是第r小的数，若不是则递归调用rank_select重新选择
		/* partition around median of medians and recursively select if necessary */
		j = partition_array(array, n, med);//划分数组，返回med在新数组中的索引
		if (r == j)//结果是第r小的数
			return med;
		else if (r < j)//第r小的数在前半部分
			return rank_select(array, j, r);
		else//第r小的数在后半部分
		{
			array += j + 1;
			return rank_select(array, (n - j - 1), (r - j - 1));
		}
	}
	
	void insertion_sort(double* array, int n)
	{
		double k;
		int i, j;

		for (i = 1; i < n; i++)
		{
			k = array[i];
			j = i - 1;
			while (j >= 0 && array[j] > k)
			{
				array[j + 1] = array[j];
				j -= 1;
			}
			array[j + 1] = k;
		}
	}
	int partition_array(double* array, int n, double pivot)
	{
		double tmp;
		int p, i, j;

		i = -1;
		for (j = 0; j < n; j++)
			if (array[j] <= pivot)
			{
				tmp = array[++i];
				array[i] = array[j];
				array[j] = tmp;
				if (array[i] == pivot)
					p = i;//p保存枢轴的下标
			}
		//将枢轴和最后一个小于枢轴的数对换
		array[p] = array[i];
		array[i] = pivot;

		return i;
	}

	double descr_dist(feature* f1,feature* f2)
	{
		double dis = 0;
		for (int i = 0; i < D; i++)
		{
			double x = f1->descr[i] - f2->descr[i];
			dis += x * x;
		}
		return dis;
	}


	vector<bfeature> kdtree_bbf_knn(feature* feat,int k = 2,int max_nn_chks = 200)
	{
		priority_queue<bbf> pq;
		priority_queue<bfeature> bfs;
		pq.push(bbf(0, kd_root));
		int times = 0;
		while (!pq.empty() && times < max_nn_chks)
		{
			times++;
			bbf tmp = pq.top();
			pq.pop();
			kd_node* leaf = explore_to_leaf(tmp.node, feat, pq);
			for (int i = 0; i < leaf->n; i++)
			{
				bfeature bd(descr_dist(feat, &leaf->features[i]), &leaf->features[i]);
				bfs.push(bd);
			}
		}
		vector<bfeature> result;
		for (int i = 0; i < k; i++)
		{
			result.push_back(bfs.top());
			bfs.pop();
		}
		return result;
	}

	kd_node* explore_to_leaf(kd_node* node, feature* feat, priority_queue<bbf> &pq)
	{
		kd_node* unexp, *exp = node;
		int ki;
		double kv;
		while (exp != NULL && !exp->leaf)
		{
			ki = exp->ki;
			kv = exp->kv;
			if (feat->descr[ki] <= kv) {
				unexp = exp->kd_right;
				exp = exp->kd_left;
			}
			else {
				unexp = exp->kd_left;
				exp = exp->kd_right;
			}
			bbf tmp(abs(feat->descr[ki] - kv), unexp);
			pq.push(tmp);
		}
		return exp;
	}
};
