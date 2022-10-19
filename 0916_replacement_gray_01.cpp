#define _CRT_SECURE_NO_WARNINGS

// 2022. 09. 16. 전기전자공학부 이재현

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;


int main(int ac, char** av) {

	Mat img = imread("images/Lenna.png", CV_LOAD_IMAGE_COLOR);
	Mat temp = img.clone();

	const int BLK = 70;
	int cx = 0, cy = 0;
	cout << "plz input center position(X,Y)" << '\n';
	cin >> cx >> cy;

	for (int y = cy - BLK; y <= cy + BLK; y++) {
		for (int x = cx - BLK; x <= cx + BLK; x++) {
			if (cx < 0 || cx >= temp.cols || cy < 0 || cy >= temp.rows) continue; // out of bounds
			int gray = (img.at<Vec3b>(y, x)[1] + img.at<Vec3b>(y, x)[2] + img.at<Vec3b>(y, x)[3]) / 3;
			temp.at<Vec3b>(y, x)[0] = gray;
			temp.at<Vec3b>(y, x)[1] = gray;
			temp.at<Vec3b>(y, x)[2] = gray;
		}
	}

	imshow("origin_img", img);
	imshow("result_img", temp);
	waitKey(0);

	return 0;
}
