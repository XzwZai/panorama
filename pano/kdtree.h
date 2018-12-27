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
		int d, n, i, j, ki = 0; //��������ki

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

		//������ֳ�5��һ�飬��gr_tot��
		/* divide array into groups of 5 and sort them */
		gr_5 = n / 5; //��ĸ���-1��n/5����ȡ��
		gr_tot = cvCeil(n / 5.0); //��ĸ�����n/5����ȡ��
		rem_elts = n % 5;//���һ���е�Ԫ�ظ���
		tmp = array;
		//��ÿ����в�������
		for (i = 0; i < gr_5; i++)
		{
			insertion_sort(tmp, 5);
			tmp += 5;
		}
		//���һ��
		insertion_sort(tmp, rem_elts);

		//����ֵ����ֵ
		/* recursively find the median of the medians of the groups of 5 */
		tmp = (double*)calloc(gr_tot, sizeof(double));
		//��ÿ��5Ԫ���е���ֵ(���±�Ϊ2,2+5,...��Ԫ��)���Ƶ�temp����
		for (i = 0, j = 2; i < gr_5; i++, j += 5)
			tmp[i] = array[j];
		//���һ�����ֵ
		if (rem_elts)
			tmp[i++] = array[n - 1 - rem_elts / 2];
		//��temp�е���ֵmed������ֵ����ֵ
		med = rank_select(tmp, i, (i - 1) / 2);
		free(tmp);

		//������ֵ����ֵ�������飬�����ֽ���Ƿ��ǵ�rС��������������ݹ����rank_select����ѡ��
		/* partition around median of medians and recursively select if necessary */
		j = partition_array(array, n, med);//�������飬����med���������е�����
		if (r == j)//����ǵ�rС����
			return med;
		else if (r < j)//��rС������ǰ�벿��
			return rank_select(array, j, r);
		else//��rС�����ں�벿��
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
					p = i;//p����������±�
			}
		//����������һ��С����������Ի�
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
