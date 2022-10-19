#include<opencv2/features2d.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include"utils.h"

#define RESIZE_FACTOR Size(400, 400)
#define REFERENCE_FILEPATH "./images/Lecture5/case.jpg"

using namespace cv;
using namespace std;

// ORB settings
int ORB_MAX_KPTS = 1500;
float ORB_SCALE_FACTOR = 1.2;
int ORB_PYRAMID_LEVELS = 4;
float ORB_EDGE_THRESHOLD = 31.0;
int ORB_FIRST_PYRAMID_LEVEL = 0;
int ORB_WTA_K = 2;
int ORB_PATCH_SIZE = 31;

// Some image matching options
float MIN_H_ERROR = 120.50f; // Maximum error in pixels to accept an inlier  <- 얘를 조절하면 ransac 영향을 알 수 있음.
float DRATIO = 0.80f;

void main() {

	Mat img1, img1_32, img2, img2_32; 
	string img_path1, img_path2, homography_path;
	double t1 = 0.0; double t2 = 0.0;

	Mat desc1_orb; vector<KeyPoint> kpts1_orb;



	Ptr<DescriptorMatcher> matcher_l2 = DescriptorMatcher::create("BruteForce");
	Ptr<DescriptorMatcher> matcher_l1 = DescriptorMatcher::create("BruteForce-Hamming");
	img1 = imread(REFERENCE_FILEPATH, IMREAD_GRAYSCALE);
	resize(img1, img1, RESIZE_FACTOR);
	img1.convertTo(img1_32, CV_32F, 1.0 / 255.0, 0);


	Mat img1_rgb_orb = Mat(Size(img1.cols, img1.rows), CV_8UC3);
	Mat img_com_orb = Mat(Size(img1.cols * 2, img1.rows), CV_8UC3);

	Ptr<ORB> orb = ORB::create(ORB_MAX_KPTS, ORB_SCALE_FACTOR, ORB_PYRAMID_LEVELS,
		ORB_EDGE_THRESHOLD, ORB_FIRST_PYRAMID_LEVEL, ORB_WTA_K, ORB::HARRIS_SCORE,
		ORB_PATCH_SIZE);

	orb->detectAndCompute(img1, noArray(), kpts1_orb, desc1_orb, false);

	cvtColor(img1, img1_rgb_orb, COLOR_GRAY2BGR);

	draw_keypoints(img1_rgb_orb, kpts1_orb);//

	Mat frame_gray_scale = Mat(Size(img1.cols, img1.rows), CV_8UC1);
	VideoCapture capture(0);

	if (!capture.isOpened()) {
		cerr << "something wrong...\n";
		return;
	}
	Mat frame;
	while (true) {
		t1 = 0.0, t2 = 0.0;
		Mat desc2_orb;
		vector<KeyPoint> kpts2_orb;
		vector<Point2f> matches_orb, inliers_orb;
		vector<vector<DMatch>> dmatches_orb;
		int nmatches_orb = 0, ninliers_orb = 0, noutliers_orb = 0;
		int nkpts1_orb = 0, nkpts2_orb = 0;
		float ratio_orb = 0.0;

		double torb = 0.0; // Create the L2 and L1 matchers

		capture >> frame;
		resize(frame, frame, RESIZE_FACTOR);

		cvtColor(frame, frame_gray_scale, COLOR_BGR2GRAY);
		frame_gray_scale.convertTo(img2_32, CV_32F, 1.0 / 255.0, 0);

	
		Mat img2_rgb_orb = Mat(Size(frame_gray_scale.cols, img1.rows), CV_8UC3);


		t1 = getTickCount();
		orb->detectAndCompute(frame, noArray(), kpts2_orb, desc2_orb, false);
		
		matcher_l1->knnMatch(desc1_orb, desc2_orb, dmatches_orb, 2);
		matches2points_nndr(kpts1_orb, kpts2_orb, dmatches_orb, matches_orb, DRATIO);
		compute_inliers_ransac(matches_orb, inliers_orb, MIN_H_ERROR, false);
		nkpts1_orb = kpts1_orb.size();
		nkpts2_orb = kpts2_orb.size();
		nmatches_orb = matches_orb.size() / 2;
		ninliers_orb = inliers_orb.size() / 2;
		noutliers_orb = nmatches_orb - ninliers_orb;
		ratio_orb = 100.0 * (float)(ninliers_orb) / (float)(nmatches_orb);
		t2 = cv::getTickCount();
		torb = 1000.0 * (t2 - t1) / cv::getTickFrequency();

		cvtColor(frame_gray_scale, img2_rgb_orb, COLOR_GRAY2BGR);
		draw_keypoints(img2_rgb_orb, kpts2_orb);//
		draw_inliers(img1_rgb_orb, img2_rgb_orb, img_com_orb, inliers_orb, 0);//
		imshow("ORB", img_com_orb);


		cout << "ORB Results" << endl;
		cout << "**************************************" << endl;
		cout << "Number of Keypoints Image 1: " << nkpts1_orb << endl;
		cout << "Number of Keypoints Image 2: " << nkpts2_orb << endl;
		cout << "Number of Matches: " << nmatches_orb << endl;
		cout << "Number of Inliers: " << ninliers_orb << endl;
		cout << "Number of Outliers: " << noutliers_orb << endl;
		cout << "Inliers Ratio: " << ratio_orb << endl;
		cout << "ORB Features Extraction Time (ms): " << torb << endl; cout << endl;

		if (waitKey(30) >= 0) break;
	}
}

/*
		RANSAC : Random Sampling Consensus
				 outlier 의 영향을 줄이기 위해, random sampling 하는 것임.
				 여러 번 random sampling 해서 최고의 consensus 를 뽑는(voting) 방식으로 outlier 에 robust 한 것이 최고 장점!
				 RANSAC 을 돌리지 않으면 Outlier 에 의한 영향을 너무 크게 받아서, 성능 크게 저하될 우려.


*/