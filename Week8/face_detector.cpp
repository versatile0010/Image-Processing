#define _CRT_SECURE_NO_WARNINGS

// Assignment04 : HOG based Face Detector
// 2022. 11. 03. 전기전자공학부 이재현

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

#define BLK 4 // Block Size
#define PI 3.1415926535897932384626433832795028841971693 // Pi
#define BIN 9 // # of bins
#define WIN 36

using namespace cv;
using namespace std;

/**Convert BGR channel to GRAY*/
Mat cvtBGR2Gray(Mat Image) { 
	int height, width;
	height = Image.rows, width = Image.cols;
	Mat result(height, width, CV_8UC1);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.at<uchar>(i, j) = (Image.at<Vec3b>(i, j)[0] + Image.at<Vec3b>(i, j)[1] + Image.at<Vec3b>(i, j)[2]) / 3;
		}
	}
	return result;
}
/**Convert GRAY channel to BGR*/
Mat cvtGray2BGR(Mat Image) {
	int height, width;
	height = Image.rows, width = Image.cols;
	Mat result(height, width, CV_8UC3);
	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++) {
			result.at<Vec3b>(i, j)[0] = Image.at<uchar>(i, j);
			result.at<Vec3b>(i, j)[1] = Image.at<uchar>(i, j);
			result.at<Vec3b>(i, j)[2] = Image.at<uchar>(i, j);
		}
	}
	return result;
}
/**Get cosine similarity*/
float get_cosine_sim(float* obj1, float* obj2, int size) {
	float dot = 0.0;
	float denom_of_obj1 = 0.0;
	float denom_of_obj2 = 0.0;

	for (int i = 0; i < size; i++) {
		dot += obj1[i] * obj2[i];
		denom_of_obj1 += obj1[i] * obj1[i];
		denom_of_obj2 += obj2[i] * obj2[i];
	}
	float sim = dot / (sqrt(denom_of_obj1) * sqrt(denom_of_obj2));
	return sim;
}

/**Get Gradient Map*/
void getGradientMap(float* mag, float* dir_arr, Mat input) {
	int x, y, xx, yy;
	int height = input.rows;
	int width = input.cols;
	float conv_x, conv_y;
	float dir = 0.0;

	int mask_x[9] = { -1, -2, -1, 0, 0, 0, 1, 2, 1 }; // sobel mask
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

			mag[y * width + x] = sqrt(conv_x * conv_x + conv_y * conv_y); // calc magninute

			dir = atan2(conv_y, conv_x) * 180.0 / PI; // calc direction ( radian to degree )
			if (dir < 0) dir += 180.0;
			dir_arr[y * width + x] = dir;

			if (max < mag[y * width + x]) max = mag[y * width + x];
			if (min > mag[y * width + x]) min = mag[y * width + x];
		}
	}
}

void get_hog_histogram(Mat input, float* histogram, float* dir_arr, float* mag, int* idx = 0) {
	int x, y, xx, yy;
	int height = WIN;
	int width = WIN;
	const int REF_BLK = width / 3;

	float* temp_histogram = (float*)calloc(BIN, sizeof(float));

	 int cnt = 0;
	for (int i = 0; i <= height - REF_BLK; i += REF_BLK / 2) {
		for (int j = 0; j <= width - REF_BLK; j += REF_BLK / 2) {
			fill(temp_histogram, temp_histogram + 9, 0); // set all zero
			double dir_val = 0.0;

			for (int m = i; m < REF_BLK +i; m++) { // calculate histogram at each block
				for (int n = j; n < REF_BLK +j; n++) {
					dir_val = dir_arr[m * width + n];
					temp_histogram[(int)dir_val / 20] += mag[m * width + n];
				}
			}

			// L-2 normalization
			float normalization_sum = 0.0;
			for (int i = 0; i < BIN; i++) {
				normalization_sum += temp_histogram[i] * temp_histogram[i]; 
			}
			normalization_sum = sqrt(normalization_sum);
			for (int i = 0; i < BIN; i++) {
				if (normalization_sum == 0) {
					histogram[cnt * BIN + i] = 0;
					continue;
				}
				histogram[cnt * BIN + i] = temp_histogram[i] / normalization_sum;
			} 
			cnt++;
		}
	}
	*idx = cnt;
	free(temp_histogram);
}

Mat face_detection(Mat tar, Mat ref) {
	Mat result = tar.clone();
	Mat face_result = cvtGray2BGR(tar);
	int height = tar.rows; int width = tar.cols; int idx = 0; int cnt = 0;

	float max = -1.0; float min = 1;
	// allocation
	float* ref_mag = (float*)calloc(ref.rows * ref.cols, sizeof(float));
	float* ref_dir = (float*)calloc(ref.rows * ref.cols, sizeof(float));
	float* ref_histo = (float*)calloc(BIN * 25, sizeof(float));

	float* tar_mag = (float*)calloc(tar.rows * tar.cols, sizeof(float));
	float* tar_dir = (float*)calloc(tar.rows * tar.cols, sizeof(float));
	float* tar_histo = (float*)calloc(BIN * 25, sizeof(float));

	float* test_mag = (float*)calloc(WIN * WIN, sizeof(float));
	float* test_dir = (float*)calloc(WIN * WIN, sizeof(float));

	float* similarity_map = (float*)calloc(height * width, sizeof(float));

	// get magnitude, direction map
	getGradientMap(ref_mag, ref_dir, ref);
	getGradientMap(tar_mag, tar_dir, tar);

	// get reference histogram
	get_hog_histogram(ref, ref_histo, ref_dir, ref_mag, &cnt);

	// find face
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			idx = 0; 
			for (int yy = y - WIN / 2; yy < y + WIN / 2; yy++) {
				for (int xx = x - WIN / 2; xx < x + WIN / 2; xx++) {
					if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
						test_mag[idx] = tar_mag[yy * width + xx];
						test_dir[idx] = tar_dir[yy * width + xx];
						idx++;
					}
				}
			} 
			get_hog_histogram(tar, tar_histo, test_dir, test_mag, &cnt);
			similarity_map[y * width + x] = get_cosine_sim(ref_histo, tar_histo, BIN * cnt);
			if (similarity_map[y * width + x] < min) min = similarity_map[y * width + x];
			if (similarity_map[y * width + x] > max) max = similarity_map[y * width + x];

		}
	}
	// non - maxima suppression
	for (int i = 1; i < height-1; i++) {
		for (int j = 1; j < width-1; j++) {
			// thresholding
			float temp_max = 0.0;
			for (int y = 0; y < 9; y++) {
				for (int x = 0; x < 9; x++) {
					if ((y + i) < 0 || (y + i) >= height || (x + j) < 0 || (x + j) >= width) continue;
					if (temp_max < similarity_map[(y+i) * width + x+j])
						temp_max = similarity_map[(y + i) * width + x + j];
				} 
			}
			for (int y = 0; y < 9; y++) {
				for (int x = 0; x < 9; x++) {
					if ((y + i) < 0 || (y + i) >= height || (x + j) < 0 || (x + j) >= width) continue;
					if (similarity_map[(y + i) * width + x + j] < temp_max)
						similarity_map[(y + i) * width + x + j] = 0;
	
				}
			}
		}
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (similarity_map[y * width + x] > 0.955 * max) {
				rectangle(face_result, Rect(Point(x-16, y-16), Point(x + 16, y + 16)), Scalar(0, 255, 0), 1, 8, 0);
			}
		}
	}

	// de-allocation
	free(ref_mag); free(ref_dir); free(tar_mag); free(tar_dir);
	free(ref_histo); free(tar_histo);
	free(test_mag); free(test_dir); free(similarity_map);

	return face_result;
}

Mat Histogram_equalization(Mat &img) {
	Mat result(img.rows, img.cols, CV_8UC1);
	int sum = 0;
	float* histogram = (float*)calloc(256, sizeof(float));
	float* cdf = (float*)calloc(256, sizeof(float));
	float* normalized_histo = (float*)calloc(256, sizeof(float));
	float* matched_histo = (float*)calloc(256, sizeof(float));

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			histogram[img.at<uchar>(i, j)]++; // vote
		}
	}
	cdf[0] = histogram[0];

	for (int i = 1; i < 256; i++) { // calculate Cumulative Distribution Function
		sum += histogram[i];
		cdf[i] = sum;
	}

	for (int i = 0; i < 256; i++) { // nomalization
		normalized_histo[i] = cdf[i] / (float)(img.rows*img.cols);
	}

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) { // matching
			result.at<uchar>(i, j) = normalized_histo[img.at<uchar>(i, j)]*255;
		}
	}

	free(histogram); free(cdf); free(normalized_histo); free(matched_histo);
	return result;
}

void main() {
	Mat tar = imread("images/Lecture7/face_tar.bmp", IMREAD_GRAYSCALE); // image read
	Mat ref = imread("images/Lecture7/face_ref.bmp", IMREAD_GRAYSCALE); // image read
	imshow("bef ref", ref);
	ref = Histogram_equalization(ref);
	Mat result = face_detection(tar, ref);

	imshow("tar", tar);
	imshow("ref", ref);
	imshow("result", result);
	waitKey(0);
}
