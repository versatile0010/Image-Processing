/*

    전기전자심화설계및소프트웨어실습
    Assignment05 : Face Verification Algorithm
    2022. 11. 06. 전기전자공학부 이재현

*/

#include<opencv2/imgproc.hpp>
#include<opencv2/highgui.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include <vector>
#include <stdio.h>
#include <Windows.h>

#include <chrono>

using namespace cv;
using namespace std;

#define WIN 128

const int BLK = WIN / 4;


const int uniform_lookup[256] =
{
	0,1,2,3,4,58,5,6,7,58,58,58,8,58,9,10,11,58,58,58,58,58,58,58,12,58,58,58,13,58,
	14,15,16,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,17,58,58,58,58,58,58,58,18,
	58,58,58,19,58,20,21,22,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,23,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,58,24,58,58,58,58,58,58,58,25,58,58,58,26,58,27,28,29,30,58,31,58,58,58,32,58,
	58,58,58,58,58,58,33,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,34,58,58,58,58,
	58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,58,
	58,35,36,37,58,38,58,58,58,39,58,58,58,58,58,58,58,40,58,58,58,58,58,58,58,58,58,
	58,58,58,58,58,58,41,42,43,58,44,58,58,58,45,58,58,58,58,58,58,58,46,47,48,58,49,
	58,58,58,50,51,52,58,53,54,55,56,57
};

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

/**Fill elements*/
template <typename T>
void fill_zeros(T* arr, int size) {
	for (int i = 0; i < size; i++) {
		arr[i] = 0;
	}
}

/** Uniform LBP Descriptor */
void calc_lbp(Mat src, int sx, int sy, int face_height, int face_width, float* lbp_histo) {
	int binary_table[8] = { 1, 2, 4, 8, 16, 32, 64, 128};
	float* histo = (float*)calloc(59* (WIN / (BLK / 2) - 1) * (WIN / (BLK / 2) - 1), sizeof(float));
	Mat img = src.clone();
	Mat LBPimg(face_height, face_width, CV_8UC1);
	Mat LBPimg_resize(face_height, face_width, CV_8UC1);

	img = cvtBGR2Gray(img);
	int idx = 0; int result = 0; int center_pixel = 0; int val = 0; int sum = 0;
	for (int y = sy; y < face_height + sy; y++) {
		for (int x = sx; x < face_width + sx; x++) {
			if (x < 0 || x >= img.cols || y < 0 || y >= img.rows) continue;
			val = 0;
			result = 0; center_pixel = img.at<uchar>(y, x);
			
			if (center_pixel < img.at<uchar>(y - 1, x)) val += binary_table[0];
			if (center_pixel < img.at<uchar>(y - 1, x + 1)) val += binary_table[1];
			if (center_pixel < img.at<uchar>(y, x + 1)) val += binary_table[2];
			if (center_pixel < img.at<uchar>(y + 1, x + 1)) val += binary_table[3];
			if (center_pixel < img.at<uchar>(y + 1, x)) val += binary_table[4];
			if (center_pixel < img.at<uchar>(y + 1, x - 1)) val += binary_table[5];
			if (center_pixel < img.at<uchar>(y, x - 1)) val += binary_table[6];
			if (center_pixel < img.at<uchar>(y - 1, x - 1))val += binary_table[7];

			LBPimg.at<uchar>(y-sy, x-sx) = uniform_lookup[val];
		}
	}
	resize(LBPimg, LBPimg_resize, Size(WIN, WIN), 1);
	int cnt = 0;
	// generate LBP histogram
	for (int y = 0; y <= WIN - BLK; y += BLK / 2) {
		for (int x = 0; x <= WIN - BLK; x += BLK / 2) {


			fill_zeros(histo, 59);
			for (int yy = y; yy < y + BLK; yy++) {
				for (int xx = x; xx < x + BLK; xx++) {
					histo[LBPimg_resize.at<uchar>(yy, xx)]++;
				}
			}
			// normalization 
			sum = 0;
			for (int i = 0; i < 59; i++) {
				sum += histo[i] * histo[i];
			}
			sum = sqrt(sum);
			for (int i = 0; i < 59; i++) {
				if (sum == 0)
					lbp_histo[59 * cnt + i] = 0;
				else
					lbp_histo[59 * cnt + i] = histo[i] / sum;
			}
			cnt++;
		}
	}
	free(histo);
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


void main(int ac, char** av) {
	int cnt = 0; float total = 0.0; float avg_time = 0.0;
	Mat frame;
	CascadeClassifier cascade;
	cascade.load("C:\\opencv\\sources\\data\\lbpcascades\\lbpcascade_frontalface.xml");
	float* target_histo = (float*)calloc(59 * (WIN / (BLK / 2) - 1) * (WIN / (BLK / 2) - 1), sizeof(float));
	float* ref_histo = (float*)calloc(59 * (WIN / (BLK / 2) - 1) * (WIN / (BLK / 2) - 1), sizeof(float));
	bool registered = false;
	float t = 0.85;
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cerr << "Couldn't open the web camera...\n";
		return;
	}
	while (true) {
		capture >> frame; // we can get an image frame.

		vector<Rect> faces;
		cascade.detectMultiScale(frame, faces, 1.1, 4, 0 | CASCADE_SCALE_IMAGE, Size(100, 100));

		if (faces.size() == 1 && registered == false) {
			Point lb(faces[0].x + faces[0].width, faces[0].y + faces[0].height);
			Point tr(faces[0].x, faces[0].y);
			rectangle(frame, lb, tr, Scalar(0, 255, 255), 3, 8, 0);

			calc_lbp(frame, faces[0].x, faces[0].y, faces[0].height, faces[0].width, ref_histo);

			putText(frame, "registered", Point(faces[0].x, faces[0].y - 10), 0, 1, Scalar(0, 255, 255), 3, 8);
			cout << "registerd\n";
			registered = true;
		
		}
		if (faces.size() > 0 && registered == true) {
			for (int i = 0; i < faces.size(); i++) {
				float score = 0.0;
				fill_zeros(target_histo, 59 * (WIN / (BLK / 2) - 1) * (WIN / (BLK / 2) - 1));
				Point lb(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
				Point tr(faces[i].x, faces[i].y);
				rectangle(frame, lb, tr, Scalar(0, 255, 0), 1, 8, 0);

				chrono::system_clock::time_point start_time = chrono::system_clock::now();
				calc_lbp(frame, faces[i].x, faces[i].y, faces[i].height, faces[i].width, target_histo);
				chrono::system_clock::time_point end_time = chrono::system_clock::now();
				chrono::microseconds micro = chrono::duration_cast<chrono::microseconds>(end_time - start_time);

				score = get_cosine_sim(target_histo, ref_histo, 59 * (WIN / (BLK / 2) - 1) * (WIN / (BLK / 2) - 1));
				cout << "score = " << score << "     execution time : " << micro.count() << " microseconds" << '\n';
				total += micro.count(); cnt++;
				if (cnt == 500) {
					avg_time = total / cnt;
					cout << "=============================== 500 회 평균 수행시간 : " << avg_time << '\n';
				}
				if (score > t) {
					rectangle(frame, lb, tr, Scalar(0, 255, 0), 3, 8, 0);
					putText(frame, "valid", Point(faces[i].x, faces[i].y - 10), 0, 1, Scalar(0, 255, 0), 3, 8);
				}
				else {
					rectangle(frame, lb, tr, Scalar(0, 0, 255), 3, 8, 0);
					putText(frame, "invalid", Point(faces[i].x, faces[i].y - 10), 0, 1, Scalar(0, 0, 255), 3, 8);
				}
			}
		}

		imshow("Video", frame);
		if (waitKey(30) >= 0) break;
	}

	free(ref_histo);
	free(target_histo);
	waitKey(0);
}
