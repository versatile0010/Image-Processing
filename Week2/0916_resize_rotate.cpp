#define _CRT_SECURE_NO_WARNINGS

// 2022. 09. 16. 전기전자공학부 이재현
// 09.16 Assignment : bilinear 구현하고, nn method, average method 랑 성능 비교 분석하는 레포트 작성하기

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>

using namespace cv;
using namespace std;

#define PI 3.1415926535897932384626433832795028841971693

int limit(int num) {
	if (num > 255) return 255;
	else return num;
}

Mat ImageResize(Mat input, float scale, String type) {
	int x, y;
	int height, width;
	int re_height, re_width;
	float pos_x, pos_y;
	int sx, sy;

	height = input.rows;
	width = input.cols;
	re_height = scale * height;
	re_width = scale * width;

	Mat result(re_height, re_width, CV_8UC3);

	if (type == "NN") {
		// Nearest Neighbor method
		for (y = 0; y < re_height; y++) {
			for (x = 0; x < re_width; x++) {
				pos_x = (1.0 / scale) * x;
				pos_y = (1.0 / scale) * y;

				sx = (int)(pos_x + 0.5);
				sy = (int)(pos_y + 0.5);

				result.at<Vec3b>(y, x)[0] = input.at<Vec3b>(sy, sx)[0];
				result.at<Vec3b>(y, x)[1] = input.at<Vec3b>(sy, sx)[1];
				result.at<Vec3b>(y, x)[2] = input.at<Vec3b>(sy, sx)[2];
			}
		}
		return result;
	}
	else if (type == "AVERAGE") {
		// Averaging method
		for (y = 0; y < re_height; y++) {
			for (x = 0; x < re_width; x++) {
				pos_x = (1.0 / scale) * x;
				pos_y = (1.0 / scale) * y;

				sx = (int)(pos_x + 0.5);
				sy = (int)(pos_y + 0.5);

				result.at<Vec3b>(y, x)[0] = (input.at<Vec3b>(sy, sx)[0] + input.at<Vec3b>(sy + 1, sx)[0] +
					input.at<Vec3b>(sy, sx+1)[0] + input.at<Vec3b>(sy + 1, sx + 1)[0]) / 4;
				result.at<Vec3b>(y, x)[1] = (input.at<Vec3b>(sy, sx)[1] + input.at<Vec3b>(sy + 1, sx)[1] +
					input.at<Vec3b>(sy, sx + 1)[1] + input.at<Vec3b>(sy + 1, sx + 1)[1]) / 4;
				result.at<Vec3b>(y, x)[2] = (input.at<Vec3b>(sy, sx)[2] + input.at<Vec3b>(sy + 1, sx)[2] +
					input.at<Vec3b>(sy, sx + 1)[2] + input.at<Vec3b>(sy + 1, sx + 1)[2]) / 4;
			}
		}
		return result;
	}
	else if (type == "BILINEAR") {
		// Bilinear method
		double rx, ry, p, q, b,g,r;
		int x1, x2, y1, y2;
		for (y = 0; y < re_height; y++) {
			for (x = 0; x < re_width; x++) {
				rx = static_cast<double>(width - 1) * x / (re_width - 1);
				ry = static_cast<double>(height - 1) * y / (re_height - 1);

				x1 = static_cast<int>(rx); y1 = static_cast<int>(ry);
				x2 = x1 + 1; y2 = y1 + 1;
				if (x2 == width) x2 = width - 1;
				if (y2 == height) y2 = height - 1;

				p = rx - x1; q = ry - y1;

				b = (1.0 - p) * (1.0 - q) * input.at<Vec3b>(y1, x1)[0] + p * (1.0 - q) * input.at<Vec3b>(y1, x2)[0] +
					(1.0 - p) * q * input.at<Vec3b>(y2, x1)[0] + p * q * input.at<Vec3b>(y2, x2)[0];
				g = (1.0 - p) * (1.0 - q) * input.at<Vec3b>(y1, x1)[1] + p * (1.0 - q) * input.at<Vec3b>(y1, x2)[1] +
					(1.0 - p) * q * input.at<Vec3b>(y2, x1)[1] + p * q * input.at<Vec3b>(y2, x2)[1];
				r = (1.0 - p) * (1.0 - q) * input.at<Vec3b>(y1, x1)[2] + p * (1.0 - q) * input.at<Vec3b>(y1, x2)[2] +
					(1.0 - p) * q * input.at<Vec3b>(y2, x1)[2] + p * q * input.at<Vec3b>(y2, x2)[2];

				result.at<Vec3b>(y, x)[0] = static_cast<uchar>(limit(b + 0.5));
				result.at<Vec3b>(y, x)[1] = static_cast<uchar>(limit(g + 0.5));
				result.at<Vec3b>(y, x)[2] = static_cast<uchar>(limit(r + 0.5));
			}
		}
		return result;
	}
	else {
		cerr << "error!\n";
		return result;
	}
}

Mat ImageRotate(Mat input, float degree, string type) {
	int x, y;
	int sx, sy;
	int height, width;
	height = input.rows;
	width = input.cols;
	Mat result(height, width, CV_8UC3);
	float pos_x, pos_y;
	float radian  = degree * PI / 180.0;
	float R[2][2] = {
		{cos(radian), sin(radian)},
		{-sin(radian), cos(radian)}
	}; // rotate matrix
	double rx, ry, p, q, b, g, r;
	int x1, x2, y1, y2;

	for (y = 0; y < height; y++) {
		for (x = 0; x < width; x++) {
			pos_x = R[0][0] * (x - width / 2) + R[0][1] * (y - height / 2);
			pos_y = R[1][0] * (x - width / 2) + R[1][1] * (y - height / 2);

			pos_x += width / 2;
			pos_y += height / 2;

			sx = (int)(pos_x + 0.5);
			sy = (int)(pos_y + 0.5);

			if (type == "AVERAGE") {
				// average method
				if (sx >= 0 && sx < width && sy >= 0 && sy < height) // check boundary!
				{
					result.at<Vec3b>(y, x)[0] = (input.at<Vec3b>(sy, sx)[0] + input.at<Vec3b>(sy + 1, sx)[0] +
						input.at<Vec3b>(sy, sx + 1)[0] + input.at<Vec3b>(sy + 1, sx + 1)[0]) / 4;
					result.at<Vec3b>(y, x)[1] = (input.at<Vec3b>(sy, sx)[1] + input.at<Vec3b>(sy + 1, sx)[1] +
						input.at<Vec3b>(sy, sx + 1)[1] + input.at<Vec3b>(sy + 1, sx + 1)[1]) / 4;
					result.at<Vec3b>(y, x)[2] = (input.at<Vec3b>(sy, sx)[2] + input.at<Vec3b>(sy + 1, sx)[2] +
						input.at<Vec3b>(sy, sx + 1)[2] + input.at<Vec3b>(sy + 1, sx + 1)[2]) / 4;
				}
				else
				{
					result.at<Vec3b>(y, x)[0] = 255;
					result.at<Vec3b>(y, x)[1] = 255;
					result.at<Vec3b>(y, x)[2] = 255;
				}
			}
			else if (type == "BILINEAR") {
				// bilinear method
				//cout << "bilinear!!\n";
				rx = pos_x + 0.5;
				ry = pos_y + 0.5;
				if (rx >= 0 && rx < width && ry >= 0 && ry < height) // check boundary!
				{
					x1 = static_cast<int>(rx); y1 = static_cast<int>(ry);
					x2 = x1 + 1; y2 = y1 + 1;
					if (x2 == width) x2 = width - 1;
					if (y2 == height) y2 = height - 1;

					p = rx - x1; q = ry - y1;
					b = (1.0 - p) * (1.0 - q) * input.at<Vec3b>(y1, x1)[0] + p * (1.0 - q) * input.at<Vec3b>(y1, x2)[0] +
						(1.0 - p) * q * input.at<Vec3b>(y2, x1)[0] + p * q * input.at<Vec3b>(y2, x2)[0];
					g = (1.0 - p) * (1.0 - q) * input.at<Vec3b>(y1, x1)[1] + p * (1.0 - q) * input.at<Vec3b>(y1, x2)[1] +
						(1.0 - p) * q * input.at<Vec3b>(y2, x1)[1] + p * q * input.at<Vec3b>(y2, x2)[1];
					r = (1.0 - p) * (1.0 - q) * input.at<Vec3b>(y1, x1)[2] + p * (1.0 - q) * input.at<Vec3b>(y1, x2)[2] +
						(1.0 - p) * q * input.at<Vec3b>(y2, x1)[2] + p * q * input.at<Vec3b>(y2, x2)[2];

					result.at<Vec3b>(y, x)[0] = static_cast<uchar>(limit(b + 0.5));
					result.at<Vec3b>(y, x)[1] = static_cast<uchar>(limit(g + 0.5));
					result.at<Vec3b>(y, x)[2] = static_cast<uchar>(limit(r + 0.5));
				}
				else
				{
					result.at<Vec3b>(y, x)[0] = 255;
					result.at<Vec3b>(y, x)[1] = 255;
					result.at<Vec3b>(y, x)[2] = 255;
				}
			}
			else {
				// default : nearest method
				if (sx >= 0 && sx < width && sy >= 0 && sy < height) // check boundary!
				{
					result.at<Vec3b>(y, x)[0] = input.at<Vec3b>(sy, sx)[0];
					result.at<Vec3b>(y, x)[1] = input.at<Vec3b>(sy, sx)[1];
					result.at<Vec3b>(y, x)[2] = input.at<Vec3b>(sy, sx)[2];
				}
				else
				{
					result.at<Vec3b>(y, x)[0] = 255;
					result.at<Vec3b>(y, x)[1] = 255;
					result.at<Vec3b>(y, x)[2] = 255;
				}
			}
		}
	}
	return result;
}

int main(int ac, char** av) {

	Mat img = imread("images/Lenna.png", CV_LOAD_IMAGE_COLOR);

	double fac = 0.3;
	//cout << "Input your scale factor : " << '\n';
	//cin >> fac;
	//Mat result(img.rows * fac, img.cols * fac, CV_8UC3);
	//result = ImageResize(img, fac, "NN");

	Mat result(img.rows, img.cols, CV_8UC3);
	result = ImageRotate(img, 145, "AVERAGE");
	imshow("origin_img", img);
	imshow("result_img", result);
	imwrite("rotate_BILINEAR.bmp", result);
	waitKey(0);

	return 0;
}
