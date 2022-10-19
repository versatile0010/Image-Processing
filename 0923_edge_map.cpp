#define _CRT_SECURE_NO_WARNINGS

// 2022. 09. 23. 전기전자공학부 이재현

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;

int filter_y[3][3] = {
	{-1, -1, -1},
	{0, 0, 0},
	{1, 1, 1}
};
int filter_x[3][3] = {
	{-1, 0, 1},
	{-1, 0, 1},
	{-1, 0, 1}
};

Mat cvtBGR2Gray(Mat Image) {
	int height, width;
	height = Image.rows, width = Image.cols;
	Mat result(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.at<uchar>(i, j) = (Image.at<Vec3b>(i, j)[0]+ Image.at<Vec3b>(i, j)[1]+ Image.at<Vec3b>(i, j)[2])/3;
		}
	}
	return result;
}
Mat getEdgeMap(Mat imgGray) {
	int height, width;
	height = imgGray.rows, width = imgGray.cols;
	Mat result(height, width, CV_8UC1);
	int xcenter = 3 / 2;
	int ycenter = 3 / 2;
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			double mag = 0.0;
			double fx = 0.0;
			double fy = 0.0;
			for (int m = 0; m < 3; m++) {
				for (int n = 0; n < 3; n++) {
					if (i == 0 || i == height - 1 || j == 0 || j == width - 1) continue;
					else {
						fx += imgGray.at<uchar>(i + m - ycenter, j + n - xcenter) * (double)filter_x[m][n];
						fy += imgGray.at<uchar>(i + m - ycenter, j + n - xcenter) * (double)filter_y[m][n];
					}
				}
			}
			mag = sqrt(fx * fx + fy * fy);
			result.at<uchar>(i, j) = mag;
		}
	}

	return result;
};

void gradient_computation(Mat input) { // 
	int x, y, xx, yy;
	int height = input.rows;
	int width = input.cols;
	float conv_x, conv_y;
	float* mag = (float*)calloc(height * width, sizeof(float));
	int mask_x[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	int mask_y[9] = { -1,0,1,-2,0,2,-1,0,1 };
	
	float min = 1000000, max = -1;

	Mat result(height, width, CV_8UC1);

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			// cur
			conv_x = 0;
			conv_y = 0;

			for (yy = y - 1; yy <= y + 1; yy++) {
				for (xx = x - 1; xx <= x + 1; xx++) { // calc conv_x, conv_y
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						// indexing 에 주의!
						conv_x += input.at<uchar>(yy, xx) * mask_x[(yy - (y - 1)) * 3 + (xx - (x - 1))];
						conv_y += input.at<uchar>(yy, xx) * mask_y[(yy - (y - 1)) * 3 + (xx - (x - 1))];
					}
				}
			}
			conv_x /= 9.0;
			conv_y /= 9.0; // scaling
			mag[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y); // calc magninute
			if (max < mag[y * width + x]) max = mag[y * width + x];
			if (min > mag[y * width + x]) min = mag[y * width + x];
		}
	}

	// for visualization
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			result.at<uchar>(y, x) = 255 - 255 * (mag[y * width + x] - min) / (max - min); // scaling and negative
		}
	}
	imshow("result", result);
}

int main(int ac, char** av) {
	int height, width;
	Mat imgColor = imread("images/Lenna.png", CV_LOAD_IMAGE_COLOR);
	height = imgColor.rows;
	width = imgColor.cols;

	Mat imgGray(height, width, CV_8UC1);
	Mat imgEdge(height, width, CV_8UC1);

	imgGray = cvtBGR2Gray(imgColor);
	imgEdge = getEdgeMap(imgGray);

	imshow("imgColor", imgColor);
	imshow("imgGray", imgGray);
	imshow("EdgeMap", imgEdge);

	gradient_computation(imgGray);
	waitKey(0);

	return 0;
}
