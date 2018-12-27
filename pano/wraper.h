#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#define PI 3.1415926
using namespace cv;

class MyProjector
{
public:
	float scale;
	float r_kinv[9], k_rinv[9], k[9], rinv[9], kinv[9];
	float t[3];
	void setCameraParams(const Mat &K, const Mat &R)
		//K��ʾ������ڲ���
		//R��ʾ�������ת����
		//T��ʾ�����ƽ����
	{
		//ȷ���������������ȷ
		CV_Assert(K.size() == Size(3, 3) && K.type() == CV_32F);
		CV_Assert(R.size() == Size(3, 3) && R.type() == CV_32F);
		//CV_Assert((T.size() == Size(1, 3) || T.size() == Size(3, 1)) && T.type() == CV_32F);

		Mat_<float> K_(K);    //����
		//�Ѿ�����ʽ��Kת��Ϊ������ʽ��k
		k[0] = K_(0, 0); k[1] = K_(0, 1); k[2] = K_(0, 2);
		k[3] = K_(1, 0); k[4] = K_(1, 1); k[5] = K_(1, 2);
		k[6] = K_(2, 0); k[7] = K_(2, 1); k[8] = K_(2, 2);

		Mat_<float> Kinv = K.inv();    //�õ�r���棬��R-1
		//�õ�������ʽ��rinv
		kinv[0] = Kinv(0, 0); kinv[1] = Kinv(0, 1); kinv[2] = Kinv(0, 2);
		kinv[3] = Kinv(1, 0); kinv[4] = Kinv(1, 1); kinv[5] = Kinv(1, 2);
		kinv[6] = Kinv(2, 0); kinv[7] = Kinv(2, 1); kinv[8] = Kinv(2, 2);

		Mat_<float> Rinv = R.t();    //�õ�r���棬��R-1
		//�õ�������ʽ��rinv
		rinv[0] = Rinv(0, 0); rinv[1] = Rinv(0, 1); rinv[2] = Rinv(0, 2);
		rinv[3] = Rinv(1, 0); rinv[4] = Rinv(1, 1); rinv[5] = Rinv(1, 2);
		rinv[6] = Rinv(2, 0); rinv[7] = Rinv(2, 1); rinv[8] = Rinv(2, 2);

		Mat_<float> R_Kinv = R * K.inv();    //�õ�rK-1����R-1K-1
		//�õ�������ʽ��r_kinv
		r_kinv[0] = R_Kinv(0, 0); r_kinv[1] = R_Kinv(0, 1); r_kinv[2] = R_Kinv(0, 2);
		r_kinv[3] = R_Kinv(1, 0); r_kinv[4] = R_Kinv(1, 1); r_kinv[5] = R_Kinv(1, 2);
		r_kinv[6] = R_Kinv(2, 0); r_kinv[7] = R_Kinv(2, 1); r_kinv[8] = R_Kinv(2, 2);

		Mat_<float> K_Rinv = K * Rinv;    //�õ�Kr-1����KR
		//�õ�������ʽ��k_rinv
		k_rinv[0] = K_Rinv(0, 0); k_rinv[1] = K_Rinv(0, 1); k_rinv[2] = K_Rinv(0, 2);
		k_rinv[3] = K_Rinv(1, 0); k_rinv[4] = K_Rinv(1, 1); k_rinv[5] = K_Rinv(1, 2);
		k_rinv[6] = K_Rinv(2, 0); k_rinv[7] = K_Rinv(2, 1); k_rinv[8] = K_Rinv(2, 2);

		//Mat_<float> T_(T.reshape(0, 3));    //����
		//�Ѿ�����ʽ��Tת��Ϊ������ʽ��t
		//t[0] = T_(0, 0); t[1] = T_(1, 0); t[2] = T_(2, 0);
	}
	void mapForward(float x, float y, float &u, float &v)    //����
	{		
		float oldtan = u / scale;
		//ʽ70
		float x_ = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
		float y_ = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
		float z_ = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];
		//ʽ75	
		float tanxz = atan2f(x_, z_);		
		while (tanxz < oldtan) {
			tanxz += 2 * PI;
		}		
		u = scale * tanxz;
		v = scale * y_ / sqrtf(x_ * x_ + z_ * z_);
	}

	float calFirstU()
	{
		float x = 0, y = 0;
		float x_1 = kinv[0] * x + kinv[1] * y + kinv[2];
		float y_1 = kinv[3] * x + kinv[4] * y + kinv[5];
		float z_1 = kinv[6] * x + kinv[7] * y + kinv[8];
		float x_2 = r_kinv[0] * x + r_kinv[1] * y + r_kinv[2];
		float y_2 = r_kinv[3] * x + r_kinv[4] * y + r_kinv[5];
		float z_2 = r_kinv[6] * x + r_kinv[7] * y + r_kinv[8];
		float tanxz_1 = atan2f(x_1, z_1);
		float tanxz_2 = atan2f(x_2, z_2);
		if (tanxz_2 >= tanxz_1 && tanxz_2 < 0) {
			return scale * tanxz_1;
		}
		else if (tanxz_2 > 0) {
			return 0;
		}
		else {
			return PI;
		}
	}

	void mapBackward(float u, float v, float &x, float &y)    //����
	{
		u /= scale;
		v /= scale;

		float x_ = sinf(u);
		float y_ = v;
		float z_ = cosf(u);
		//ʽ76
		float z;
		x = k_rinv[0] * x_ + k_rinv[1] * y_ + k_rinv[2] * z_;
		y = k_rinv[3] * x_ + k_rinv[4] * y_ + k_rinv[5] * z_;
		z = k_rinv[6] * x_ + k_rinv[7] * y_ + k_rinv[8] * z_;
		//ʽ74
		if (z > 0) { x /= z; y /= z; }
		else x = y = -1;
	}

};

class MyWarper
{	
public:
	float scale;
	MyProjector projector;
	MyWarper(float _scale)
	{
		projector.scale = _scale;
	}
	//Point2f warpPoint(const Point2f &pt, const Mat &K, const Mat &R)
	//	//pt��ʾͶ���Դ��
	//	//K��ʾ������ڲ���
	//	//R��ʾ�������ת����
	//	//�ú�������Ͷ���
	//{
	//	projector.setCameraParams(K, R);    //�����������
	//	Point2f uv;    //��ʾͶ��ӳ���
	//	projector.mapForward(pt.x, pt.y, uv.x, uv.y);    //ǰ��ͶӰ���õ�Ͷ���
	//	return uv;    //����Ͷ���
	//}

	
	Rect buildMaps(Size src_size, const Mat &K, const Mat &R, Mat &xmap, Mat &ymap)
		//src_size��ʾԴͼ������
		//K��ʾ������ڲ���
		//R��ʾ�������ת����
		//xmap��ymap�ֱ��ʾ���غ��������ǰ��ӳ����ٷ���ӳ���ֵ
		//�ú�������ͶӰ�������ߴ�
	{
		projector.setCameraParams(K, R);    //�����������

		Point dst_tl, dst_br;    //��ʾͶӰ��������Ͻ���������½�����
		//�õ�ӳ�������Ͻ�����dst_tl�����½�����dst_br
		detectResultRoi(src_size, dst_tl, dst_br);
		//����xmap��ymap�����С
		xmap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);
		ymap.create(dst_br.y - dst_tl.y + 1, dst_br.x - dst_tl.x + 1, CV_32F);

		float x, y;    //��ʾ����ͶӰӳ����x���y������ֵ
		//����ͶӰ�����ٽ��з���ӳ��
		for (int v = dst_tl.y; v <= dst_br.y; ++v)
		{
			for (int u = dst_tl.x; u <= dst_br.x; ++u)
			{
				//����ͶӰ
				projector.mapBackward(static_cast<float>(u), static_cast<float>(v), x, y);
				xmap.at<float>(v - dst_tl.y, u - dst_tl.x) = x;    //��ֵ
				ymap.at<float>(v - dst_tl.y, u - dst_tl.x) = y;    //��ֵ
			}
		}

		return Rect(dst_tl, dst_br);    //����ͶӰӳ������
	}

	
	Point warp(const Mat &src, const Mat &K, const Mat &R, int interp_mode, int border_mode,
		Mat &dst)
		//src��ʾԴͼ
		//K��ʾ����ڲ���
		//R��ʾ�������ת����
		//interp_mode��ʾ��ֵģʽ
		//border_mode��ʾ�߽�����ģʽ
		//dst��ʾͶӰӳ��ͼ
		//�ú�������ͶӰӳ��ͼ�����Ͻ��ڻ�׼ͼ������ϵ�µ����꣬��ȫ��ͼ������ϵ�µ�����
	{
		Mat xmap, ymap;
		Rect dst_roi = buildMaps(src.size(), K, R, xmap, ymap);    //����buildMaps����

		dst.create(dst_roi.height + 1, dst_roi.width + 1, src.type());    //������С
		//��xmap��ymap��src������ӳ�䣬�õ�dst
		remap(src, dst, xmap, ymap, interp_mode, border_mode);

		return dst_roi.tl();    //�������Ͻ�����
	}

	
	Rect warpRoi(Size src_size, const Mat &K, const Mat &R)
		//src��ʾԴͼ
		//K��ʾ����ڲ���
		//R��ʾ�������ת����
		//����ͶӰ��������
	{
		projector.setCameraParams(K, R);    //�����������

		Point dst_tl, dst_br;
		detectResultRoi(src_size, dst_tl, dst_br);    //�õ�ӳ������

		return Rect(dst_tl, Point(dst_br.x + 1, dst_br.y + 1));    //����ӳ���������
	}

	
	void detectResultRoi(Size src_size, Point &dst_tl, Point &dst_br)
		//src_size��ʾԴͼ������
		//dst_tl��dst_br�ֱ��ʾ���صõ���ͶӰ����������Ͻ���������½�����
	{
		//����4�������ֱ��ʾ���ϽǺ����½�x���y���ֵ
		float tl_uf = std::numeric_limits<float>::max();    //�ȳ�ʼ��Ϊ���ֵ
		float tl_vf = std::numeric_limits<float>::max();    //�ȳ�ʼ��Ϊ���ֵ
		float br_uf = -std::numeric_limits<float>::max();    //�ȳ�ʼ��Ϊ��Сֵ
		float br_vf = -std::numeric_limits<float>::max();    //�ȳ�ʼ��Ϊ��Сֵ

		float u, v;
		float firstU = projector.calFirstU();
		for (int y = 0; y < src_size.height; ++y)    //����Դͼ����
		{
			u = firstU;
			for (int x = 0; x < src_size.width; ++x)
			{
				if (x == 625) {
					x = x;
				}
				//ǰ��ӳ��
				projector.mapForward(static_cast<float>(x), static_cast<float>(y), u, v);
				
				tl_uf = std::min(tl_uf, u); tl_vf = std::min(tl_vf, v);    //�������Ͻ�����
				br_uf = std::max(br_uf, u); br_vf = std::max(br_vf, v);    //�������½�����
			}
		}
		//�õ����յ����ϽǺ����½�����
		dst_tl.x = static_cast<int>(tl_uf);
		dst_tl.y = static_cast<int>(tl_vf);
		dst_br.x = static_cast<int>(br_uf);
		dst_br.y = static_cast<int>(br_vf);
	}
};

