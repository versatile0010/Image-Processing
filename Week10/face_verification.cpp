
/*
	전기전자심화설계및소프트웨어실습
	Assignment06 : Face Verification Algorithm(Landmark)
	2022. 11. 13. 전기전자공학부 이재현

*/

#include <vector>
#include <iostream>
#include <fstream>

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"

#include "include/ldmarkmodel.h"

using namespace std;
using namespace cv;

#define LANDMARK_NUM 51
#define FACIAL_STARTING_POINT 17
#define BIN 59
#define WIN 128

const int BLK = WIN / 4;


static int uniform_lookup[256] =
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

/**Landmark point 에서 lbp histogram 을 계산한 temp_histo 생성*/
void gen_histo_on_landmarks(Mat &color_frame, float* temp_histo, cv::Point pt) { // <------------ 이부분 체크
	Mat gray_frame = color_frame.clone();
	gray_frame = cvtBGR2Gray(color_frame);
	int sx = pt.x; int sy = pt.y; int window_size = 16;
	int val = 0; int result = 0; int center_pixel = 0;
	int binary_table[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
	Mat LBPimg(gray_frame.rows, gray_frame.cols, CV_8UC1);
	
	for (int y = 0; y < gray_frame.rows; y++) {
		for (int x = 0; x < gray_frame.cols; x++) {
			if (x < 0 || x >= gray_frame.cols || y < 0 || y >= gray_frame.rows) continue; // out of bounds exception
			val = 0;
			center_pixel = gray_frame.at<uchar>(y, x);

			if (center_pixel < gray_frame.at<uchar>(y - 1, x))	   val += binary_table[0];
			if (center_pixel < gray_frame.at<uchar>(y - 1, x + 1)) val += binary_table[1];
			if (center_pixel < gray_frame.at<uchar>(y, x + 1))     val += binary_table[2];
			if (center_pixel < gray_frame.at<uchar>(y + 1, x + 1)) val += binary_table[3];
			if (center_pixel < gray_frame.at<uchar>(y + 1, x))     val += binary_table[4];
			if (center_pixel < gray_frame.at<uchar>(y + 1, x - 1)) val += binary_table[5];
			if (center_pixel < gray_frame.at<uchar>(y, x - 1))     val += binary_table[6];
			if (center_pixel < gray_frame.at<uchar>(y - 1, x - 1)) val += binary_table[7];

			LBPimg.at<uchar>(y, x) = uniform_lookup[val];
		}
	}
	int cnt = 0; int sum = 0;
	

	fill_zeros(temp_histo, BIN);

	//cout << "before : " << temp_histo[0] << '\t';
	for (int i = sy - window_size / 2; i < sy + window_size / 2; i++) {
		for (int j = sx - window_size / 2; j < sx + window_size / 2; j++) {
			temp_histo[LBPimg.at<uchar>(i, j)]++;
		}
	}

	for (int i = 0; i < BIN; i++) {
		sum += temp_histo[i] * temp_histo[i];
	}
	sum = sqrt(sum);
	for (int i = 0; i < BIN; i++) {
		if (sum == 0)
			temp_histo[i] = 0;
		else
			temp_histo[i] = temp_histo[i] / sum;
	}
}
/**Concatenate histogram*/
void concat_histo(float* src, float* dst, int& cnt, int size) {
	for (int i = 0; i < size; i++) {
		dst[cnt * size + i] = src[i];
	}
}
int main()
{
    ldmarkmodel modelt;
    std::string modelFilePath = "roboman-landmark-model.bin";
    while (!load_ldmarkmodel(modelFilePath, modelt)) {
        std::cerr << "File open error..." << std::endl;
        std::cin >> modelFilePath;
    }

    cv::VideoCapture mCamera(0);
    if (!mCamera.isOpened()) {
        std::cerr << "Camera opening failed..." << std::endl;
        system("pause");
        return 0;
    }
    cv::Mat Image;
    cv::Mat current_shape;
	cv::Mat ref_Image;
	cv::Mat ref_current_shape;

	float* ref_histo = (float*)calloc(LANDMARK_NUM * BIN, sizeof(float));
	float* tar_histo = (float*)calloc(LANDMARK_NUM * BIN, sizeof(float));
	float* temp_histo = (float*)calloc(BIN, sizeof(float));
	int cnt = 0;
	float sim = 0.0;

	// generating ref histogram
	ref_Image = imread("images/iu.jpg", CV_LOAD_IMAGE_COLOR);
	modelt.track(ref_Image, ref_current_shape);
	cv::Vec3d ref_eav;
	modelt.EstimateHeadPose(ref_current_shape, ref_eav);

	for (int j = 0; j < ref_current_shape.cols / 2; j++) {
		// 68 개의 point 중에서 (j : 0~16) 까지 턱선에 해당함.
		// 턱선을 제외한 눈썹, 눈, 코, 입에 해당하는 point 는 17~67 (0-index) 임.
		// 그러므로 total histogram 의 bin 개수를 51 * 9 으로 설정함.
		int x = ref_current_shape.at<float>(j);
		int y = ref_current_shape.at<float>(j + ref_current_shape.cols / 2);
		std::stringstream ss;
		ss << j;
		if (j >= FACIAL_STARTING_POINT && j < ref_current_shape.cols / 2) // 눈 코 입에 대하여
		{
			cnt = j - FACIAL_STARTING_POINT;
			gen_histo_on_landmarks(ref_Image, temp_histo, cv::Point(x, y)); // landmark 별로 histogram 생성
			concat_histo(temp_histo, ref_histo, cnt, BIN);
		}

		cv::putText(ref_Image, ss.str(), cv::Point(x, y), 0.25, 0.25, cv::Scalar(0, 255, 0));
		cv::circle(ref_Image, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);
	}

	// real-time processing
    for (;;) {
        mCamera >> Image;
		//Image = imread("images/ye.jpg", CV_LOAD_IMAGE_COLOR);
		
        modelt.track(Image, current_shape);
        cv::Vec3d eav;
        modelt.EstimateHeadPose(current_shape, eav);
        //modelt.drawPose(Image, current_shape, 50);
        int numLandmarks = current_shape.cols / 2;

		cnt = 0;
        for (int j = 0; j < numLandmarks; j++) {
            // 68 개의 point 중에서 (j : 0~16) 까지 턱선에 해당함.
			// 턱선을 제외한 눈썹, 눈, 코, 입에 해당하는 point 는 17~67 (0-index) 임.
			// 그러므로 total histogram 의 bin 개수를 51 * 9 으로 설정함.
            int x = current_shape.at<float>(j);
            int y = current_shape.at<float>(j + numLandmarks);
            std::stringstream ss;
            ss << j;
			if (j >= FACIAL_STARTING_POINT && j < numLandmarks) // 눈 코 입에 대하여
			{
				cnt = j - FACIAL_STARTING_POINT;
				gen_histo_on_landmarks(Image, temp_histo, cv::Point(x, y)); // landmark 별로 histogram 생성
				concat_histo(temp_histo, tar_histo, cnt, BIN);
			}
			
            cv::putText(Image, ss.str(), cv::Point(x, y), 0.25, 0.25, cv::Scalar(0, 255, 0));
            cv::circle(Image, cv::Point(x, y), 1, cv::Scalar(255, 0, 0), -1);	
        }
		sim = get_cosine_sim(ref_histo, tar_histo, BIN * cnt);
		

		if (sim > 0.9) {
			putText(Image, "similarity : "+to_string(sim), Point(Image.cols / 3, Image.rows*0.8), 0, 1, Scalar(0, 255, 0), 2, 8);
			putText(Image, "valid", Point(Image.cols/2, Image.rows*0.9), 0, 1, Scalar(0, 255, 0), 3, 8);
		}
		else
		{
			putText(Image, "similarity : " + to_string(sim), Point(Image.cols / 3, Image.rows*0.8), 0, 1, Scalar(0, 0, 255), 2, 8);
			putText(Image, "invalid", Point(Image.cols / 2, Image.rows*0.9), 0, 1, Scalar(0, 0, 255), 3, 8);
		}

        cv::imshow("Camera", Image);
        cv::imshow("ref", ref_Image);
        if (27 == cv::waitKey(5)) {
            mCamera.release();
            cv::destroyAllWindows();
            break;
        }
    }
	free(ref_histo);
	free(tar_histo);
	free(temp_histo);

    system("pause");
    return 0;
}

