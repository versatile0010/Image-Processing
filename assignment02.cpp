#define _CRT_SECURE_NO_WARNINGS

// Assignment
// 2022. 09. 23. 전기전자공학부 이재현

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

#define PI 3.141592
#define BLK 16 // Block size

using namespace cv;
using namespace std;

FILE* fp;

float* get_hog_histogram(Mat input, const char* filename) { 
	int x, y, xx, yy;
	int height = input.rows;
	int width = input.cols;
	float conv_x, conv_y, dir;

	float* mag = (float*)calloc(height * width, sizeof(float));
	float* dir_arr = (float*)calloc(height * width, sizeof(float));
	float* histogram = (float*)calloc(9*(height/(BLK/2) - 1) * (width/(BLK/2) - 1), sizeof(float));
	int mask_x[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 }; // sobel mask
	int mask_y[9] = { -1,0,1,-2,0,2,-1,0,1 };


	float min = 1000000, max = -1;

	Mat result(height, width, CV_8UC1);

	for (y = 0; y < height; y++) { // calculate magnitude and direction
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
			dir = atan2(conv_y, conv_x) * 180.0 / PI; // calc direction ( radian to degree )
			if (dir < 0) dir += 180.0;
			dir_arr[y * width + x] = dir;

			//histogram[(int)(dir / 20)] += mag[y * width + x];
			if (max < mag[y * width + x]) max = mag[y * width + x];
			if (min > mag[y * width + x]) min = mag[y * width + x];
		}
	}

	int idx = 0;
	// block-based hog algorithm
	float* temp_histogram = (float*)calloc(9, sizeof(float));
	for (int i = 0; i <= input.cols - BLK; i += BLK/2) { // block size = 16 , and slicing interval = block_size/2
		for (int j = 0; j <= input.rows - BLK; j += BLK/2) {
			fill(temp_histogram, temp_histogram+9,  0); // set all zero
			double dir_val = 0.0;

			for (int m = 0; m < BLK; m++) { // calculate histogram at each block
				for (int n = 0; n < BLK; n++) {
					dir_val = dir_arr[(i + m) * width + (j + n)];
					temp_histogram[(int)dir_val / 20] += mag[(i + m) * width + (j + n)];
				}
			}
			for (int i = 0; i < 9; i++) {
				histogram[idx++] = temp_histogram[i];
			} 
		}
	}

	// L-2 normalization
	float normalization_sum = 0.0;
	for (int i = 0; i < 9 * (height / (BLK / 2) - 1) * (width / (BLK / 2) - 1); i++) { normalization_sum += histogram[i] * histogram[i]; }
	normalization_sum = sqrt(normalization_sum);
	for (int i = 0; i < 9 * (height / (BLK / 2) - 1) * (width / (BLK / 2) - 1); i++) histogram[i] /= normalization_sum;
	
	fp = fopen(filename, "wt");
	for (int i = 0; i < 9 * (height / (BLK / 2) - 1) * (width / (BLK / 2) - 1); i++) {
		fprintf(fp, "%f\n", histogram[i]);
	}
	fclose(fp);
		
	return histogram; // return histogram
}

float get_similarity(float* obj1, float* obj2, int size) {
	float score = 0.0;
	
	for (int i = 0; i < size; i++) {
		score += (abs(obj1[i] - obj2[i])) * (abs(obj1[i] - obj2[i]));
	}

	score = sqrt(score);
	// use Euclidean distance
	// 더 작을수록 유사도가 높은 것
	return score;
}

int main(int ac, char** av) {
	int height, width;
	float* histo_comp1 = nullptr;
	float* histo_comp2 = nullptr;
	float* histo_lec3 = nullptr;
	float score_comp1_and_lec3 = 0.0;
	float score_comp2_and_lec3 = 0.0;
	int comp1_histo_length, comp2_histo_length, lec3_histo_length;

	Mat comp1 = imread("images/Lecture3/compare1.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat comp2 = imread("images/Lecture3/compare2.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	Mat lecture3 = imread("images/Lecture3/lecture3.bmp", CV_LOAD_IMAGE_GRAYSCALE);
	
	histo_comp1 = get_hog_histogram(comp1, "Assign2_hog/compare1.csv");
	histo_comp2 = get_hog_histogram(comp2, "Assign2_hog/compare2.csv");
	histo_lec3 = get_hog_histogram(lecture3, "Assign2_hog/histo_lec3.csv");

	comp1_histo_length = 9 * (comp1.rows / (BLK / 2) - 1) * (comp1.cols / (BLK / 2) - 1);
	comp2_histo_length = 9 * (comp2.rows / (BLK / 2) - 1) * (comp2.cols / (BLK / 2) - 1);
	lec3_histo_length = 9 * (lecture3.rows / (BLK / 2) - 1) * (lecture3.cols / (BLK / 2) - 1);

	score_comp1_and_lec3 = get_similarity(histo_comp1, histo_lec3, comp1_histo_length);
	score_comp2_and_lec3 = get_similarity(histo_comp2, histo_lec3, comp2_histo_length);

	cout << "Euclidean distance compare1 between lecture3 : " << score_comp1_and_lec3 << '\n';
	cout << "Euclidean distance compare2 between lecture3 : " << score_comp2_and_lec3 << '\n';

	waitKey(0);
	free(histo_comp1);
	free(histo_comp2);
	free(histo_lec3);
	return 0;
}
