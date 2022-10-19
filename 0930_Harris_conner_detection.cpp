#define _CRT_SECURE_NO_WARNINGS

// 2022. 09. 30. 전기전자공학부 이재현
// reference : https://darkpgmr.tistory.com/131

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;

void getGradientMap(float** X, float** Y, Mat input) {
	int x, y, xx, yy;
	int height = input.rows;
	int width = input.cols;
	float conv_x, conv_y;

	int mask_x[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 };
	int mask_y[9] = { -1,0,1,-2,0,2,-1,0,1 };

	float min = 1000000, max = -1;

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
			X[y][x] = conv_x;
			Y[y][x] = conv_y;
			// mag[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y); // calc magninute
			// if (max < mag[y * width + x]) max = mag[y * width + x];
			// if (min > mag[y * width + x]) min = mag[y * width + x];
		}
	}
}

// 조명 변화에 robust 하지만 scale 변화에 취약
Mat Harris_CornerDetect(Mat img, float k) {
	// img - Input image. It should be grayscale and float32 type.
	// k - Harris detector free parameter in the equation.
	Mat gray_img;
	Mat result = img.clone();
	if (img.channels() == 3) // gray
		cvtColor(img, gray_img, COLOR_BGR2GRAY);
	else
		gray_img = img.clone();
	// todo : gaussian filtering to improve performance.
	int gaussian_mask[9] = { 1,2,1,2,4,2,1,2,1 };
	for (int y = 0; y < gray_img.rows; y++) {
		for (int x = 0; x < gray_img.cols; x++) {
			// cur
			float conv = 0.0;

			for (int yy = y - 1; yy <= y + 1; yy++) {
				for (int xx = x - 1; xx <= x + 1; xx++) { // calc conv_x, conv_y
					if (yy >= 0 && yy < gray_img.rows && xx >= 0 && xx < gray_img.cols) {
						// indexing 에 주의!
						conv += gray_img.at<uchar>(yy, xx) * gaussian_mask[(yy - (y - 1)) * 3 + (xx - (x - 1))];
					}
				}
			}
			gray_img.at<uchar>(y, x) = conv/16.0;
		}
	}



	// calculate gradient
	float** grad_x = (float**)calloc(gray_img.rows, sizeof(float*));
	float** grad_y = (float**)calloc(gray_img.rows, sizeof(float*));
	float** r_map = (float**)calloc(gray_img.rows, sizeof(float*));
	for (int i = 0; i < gray_img.rows; i++) {
		grad_x[i] = (float*)calloc(gray_img.cols, sizeof(float));
		grad_y[i] = (float*)calloc(gray_img.cols, sizeof(float));
		r_map[i] = (float*)calloc(gray_img.cols, sizeof(float));
	}
	getGradientMap(grad_x, grad_y, gray_img);

	//compute R at every pixel position
	float txx, txy, tyy;
	float det = 0.0; float tr = 0.0;
	for (int i = 1; i < gray_img.rows-1; i++) {
		for (int j = 1; j < gray_img.cols-1; j++) {
			txx = 0.0, txy = 0.0, tyy = 0.0;

			for (int y = 0; y < 3; y++) { // use 3x3 window
				for (int x = 0; x < 3; x++) {
					txx += grad_x[i + y - 1][j + x - 1] * grad_x[i + y - 1][j + x - 1];
					txy += grad_x[i + y - 1][j + x - 1] * grad_y[i + y - 1][j + x - 1];
					tyy += grad_y[i + y - 1][j + x - 1] * grad_y[i + y - 1][j + x - 1];
				}
				det = txx * tyy - txy * txy;
				tr = txx + tyy;
				r_map[i][j] = det - k*tr*tr;
			}
		}
	}
	// non - maxima suppression
	for (int i = 1; i < gray_img.rows - 1; i++) {
		for (int j = 1; j < gray_img.cols - 1; j++) {
			// thresholding
			float max = 0.0;
			if (r_map[i][j] < 0) continue;
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					if (max < r_map[x][y])
						max = r_map[x][y];
				}
			}
			for (int x = 0; x < 3; x++) {
				for (int y = 0; y < 3; y++) {
					if (r_map[x][y] < max)
						r_map[x][y] = 0;
				}
			}
		}
	}


	// for visualization
	Scalar c;
	Point pCenter;
	float radius = 0.3;
	c.val[0] = 0, c.val[1] = 0, c.val[2] = 255;
	for (int i = 1; i < gray_img.rows - 1; i++) {
		for (int j = 1; j < gray_img.cols - 1; j++) { 
			// thresholding
			if (r_map[i][j] > 500000) {
				pCenter.x = j;
				pCenter.y = i;
				circle(result, pCenter, 0.3, c, 2, 8, 0);
			}
		}
	}

	free(grad_y);
	free(grad_x);
	free(r_map);
	return result;
}

int main(int ac, char** av) {

	Mat test = imread("images/Lecture4/test.png", CV_LOAD_IMAGE_COLOR);
	Mat ref = imread("images/Lecture4/ref.bmp", CV_LOAD_IMAGE_COLOR);
	Mat tar = imread("images/Lecture4/tar.bmp", CV_LOAD_IMAGE_COLOR);
	Mat cube = imread("images/Lecture4/cube.png", CV_LOAD_IMAGE_COLOR);

	Mat result_test;
	Mat result_ref;
	Mat result_tar;
	Mat result_cube;

	float k = 0.04;
	result_test = Harris_CornerDetect(test, k);
	imshow("result_test", result_test);

	result_ref = Harris_CornerDetect(ref, k);
	imshow("result_ref", result_ref);

	result_tar = Harris_CornerDetect(tar, k);
	imshow("result_tar", result_tar);

	result_cube = Harris_CornerDetect(cube, k);
	imshow("result_cube", result_cube);

	waitKey(0);

	return 0;
}
