#define _CRT_SECURE_NO_WARNINGS

// Assignment
// 2022. 09. 23. 전기전자공학부 이재현

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

#define PI 3.141592

using namespace cv;
using namespace std;

float* get_hog_histogram(Mat input) { // by professor
	int x, y, xx, yy;
	int height = input.rows;
	int width = input.cols;
	float conv_x, conv_y, dir;
	float* mag = (float*)calloc(height * width, sizeof(float));
	float* histogram = (float*)calloc(9, sizeof(float));
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
			dir = atan2(conv_y, conv_x) * 180 / PI ; // calc direction ( radian to degree )
			if (dir < 0) dir += 180;
			histogram[(int)(dir / 20)] += mag[y*width+x];
			if (max < mag[y * width + x]) max = mag[y * width + x];
			if (min > mag[y * width + x]) min = mag[y * width + x];
		}
	}
	// L-2 normalization
	float normalization_sum = 0.0;
	for (int i = 0; i < 9; i++) { normalization_sum += histogram[i]*histogram[i];}
	normalization_sum = sqrt(normalization_sum);
	for (int i = 0; i < 9; i++) histogram[i] /= normalization_sum;


	cout << "HOG histogram result\n";
	for (int i = 0; i < 9; i++)
	{
		cout << histogram[i] << "  ";
	} cout << '\n';

	// for visualization
	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			result.at<uchar>(y, x) = 255 - 255 * (mag[y * width + x] - min) / (max - min); // scaling and negative
		}
	}
	return histogram;
}

int main(int ac, char** av) {
	int height, width;
	float* histo_comp1 = (float*)calloc(9, sizeof(float));
	Mat imgGray = imread("images/Lecture3/compare1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	histo_comp1 = get_hog_histogram(imgGray);
	waitKey(0);

	free(histo_comp1);
	return 0;
}
