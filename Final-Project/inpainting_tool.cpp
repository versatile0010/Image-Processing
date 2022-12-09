#define _CRT_SECURE_NO_WARNINGS

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/photo.hpp>

#include <iostream>
#include <stdio.h>
#include <windows.h>
#include <tchar.h>
#include <opencv2/photo.hpp>
#include <unordered_set>
#include <filesystem>

using namespace cv;
using namespace std;

string main_window_name = "MAIN";

struct ITEMS {
	Mat frame;
	Point p1;
	Point p2;
	Rect roi_box;
	bool drag_flag = false;
	bool update_flag = false;
};

template <class T1, class T2>
float get_distance(T1* obj1, T2* obj2, String type) {
	//RGB pixel 에 대한 distance 를 계산함
	float dist = 0;
	for (int i = 0; i < 3; i++) {
		if (type == "L2") {
			dist += (obj1[i] - obj2[i]) * (obj1[i] - obj2[i]);
		}
		if (type == "L1") {
			dist += abs(obj1[i] - obj2[i]);
		}
	}
	if (type == "L2") return sqrt(dist);
	else return dist;
}

void K_means_clustering(Mat& input, Mat& result, int k, string Distance_Metric) {
	int width = input.cols;
	int height = input.rows;
	vector<Point>* st = new std::vector<Point>[k]; // clustered pixel point 를 담을 배열
	srand(time(NULL)); // for random initialization

	float** k_centroid_arr = (float**)calloc(k, sizeof(float*));
	float** k_centroid_arr_save = (float**)calloc(k, sizeof(float*));
	for (int i = 0; i < k; i++) {
		k_centroid_arr[i] = (float*)calloc(3, sizeof(float)); // bgr channel
		k_centroid_arr_save[i] = (float*)calloc(3, sizeof(float));
	}

	// random initialization
	for (int i = 0; i < k; i++) {
		int pos_x = rand() % width;
		int pos_y = rand() % height;

		k_centroid_arr[i][0] = input.at<Vec3b>(pos_y, pos_x)[0];
		k_centroid_arr[i][1] = input.at<Vec3b>(pos_y, pos_x)[1];
		k_centroid_arr[i][2] = input.at<Vec3b>(pos_y, pos_x)[2];
	}

	bool have_to_update = true;
	while (have_to_update) {
		have_to_update = false;
		for (int i = 0; i < k; i++)
			st[i].clear();
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < 3; j++) {
				k_centroid_arr_save[i][j] = k_centroid_arr[i][j]; // 현재(업데이트 이전) center 정보를 저장함
				k_centroid_arr[i][j] = 0;
			}
		}

		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				int present_pixel_arr[3]; // (x,y) 에서의 r,g,b 값을 담을 배열
				present_pixel_arr[0] = input.at<Vec3b>(y, x)[0];
				present_pixel_arr[1] = input.at<Vec3b>(y, x)[1];
				present_pixel_arr[2] = input.at<Vec3b>(y, x)[2];

				float min_distance = INT_MAX;
				int cluster_idx = 0;
				for (int i = 0; i < k; i++) {
					float dist = get_distance(k_centroid_arr_save[i], present_pixel_arr, Distance_Metric);
					if (dist < min_distance) {
						min_distance = dist;
						cluster_idx = i;
					}
				}
				// 최소거리를 가지는 cluster 을 찾았으면 해당 cluster 에 좌표를 저장함.
				st[cluster_idx].push_back(Point(x, y));
				// 그리고 픽셀 rgb 값을 누적시킴.
				k_centroid_arr[cluster_idx][0] += present_pixel_arr[0];
				k_centroid_arr[cluster_idx][1] += present_pixel_arr[1];
				k_centroid_arr[cluster_idx][2] += present_pixel_arr[2];
			}
		}
		// 위 단계를 지나면, clustering 이 한번 된 것.
		// 그러면 centroid 를 updating
		for (int i = 0; i < k; i++) {
			int size = st[i].size(); // 각각의 cluster 로 분류된 point 개수를 셈.
			if (size == 0) continue; // 해당 cluster 로 분류된 point 가 하나도 없으면 continue;
			k_centroid_arr[i][0] /= size;
			k_centroid_arr[i][1] /= size;
			k_centroid_arr[i][2] /= size;
		}
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < 3; j++) {
				if (k_centroid_arr[i][j] != k_centroid_arr_save[i][j]) {
					// 이전 centroid 의 값과 다르면, updating 을 계속함
					// 만약 같으면 optimized 되었다고 판단.
					have_to_update = true;
					break;
				}
			}
		}
	}
	// Clustering Complete!
	// Mapping
	for (int i = 0; i < k; i++) {
		int size = st[i].size();
		for (int j = 0; j < size; j++) {
			int x = st[i][j].x;
			int y = st[i][j].y;
			result.at<Vec3b>(y, x)[0] = k_centroid_arr[i][0];
			result.at<Vec3b>(y, x)[1] = k_centroid_arr[i][1];
			result.at<Vec3b>(y, x)[2] = k_centroid_arr[i][2];
		}
	}

	for (int i = 0; i < k; i++) {
		free(k_centroid_arr[i]);
		free(k_centroid_arr_save[i]);
	}
	free(k_centroid_arr);
	free(k_centroid_arr_save);

	delete[] st;
}

void SetROI(int event, int x, int y, int flags, void* param)
{
	ITEMS* p = (ITEMS*)param;

	if (event == CV_EVENT_LBUTTONDOWN)
	{
		p->p1.x = x;
		p->p1.y = y;
		p->p2 = p->p2;
		p->drag_flag = true;
	}
	if (event == CV_EVENT_LBUTTONUP)
	{
		int w = x - p->p1.x;
		int h = y - p->p1.y;

		p->roi_box.x = p->p1.x;
		p->roi_box.y = p->p1.y;
		p->roi_box.width = w;
		p->roi_box.height = h;
		p->drag_flag = false;

		if (w >= 10 && h >= 10)
		{
			p->update_flag = true;
			cout << "updated!\n";
		}
	}
	if (p->drag_flag && event == CV_EVENT_MOUSEMOVE)
	{
		if (p->p2.x != x || p->p2.y != y)
		{
			Mat img = p->frame.clone();
			p->p2.x = x;
			p->p2.y = y;
			rectangle(img, p->p1, p->p2, Scalar(0, 255, 255), 2);
			imshow(main_window_name, img);
		}
	}
}

int int_cmp(const void* a, const void* b) {
	return (*(int*)a - *(int*)b);
}
void gen_insert_sort(void* base, int nelem, int width, int(*fcmp)(const void*, const void*), void* fre_arr) {
	int i, j;
	void* t = malloc(width);
	for (i = 0; i < nelem; i++) {
		memcpy(t, (char*)base + i * width, width);
		j = i;
		while (j > 0 && fcmp((char*)fre_arr + (j - 1) * width, t) > 0) {
			memcpy((char*)base + (j)*width, (char*)base + (j - 1) * width, width);
			memcpy((char*)fre_arr + (j)*width, (char*)fre_arr + (j - 1) * width, width);
			j--;
		}
		memcpy((char*)base + j * width, t, width);
		memcpy((char*)fre_arr + j * width, t, width);
	}
}
void insert_sort(int* a, int N) {
	int i, j, t;
	for (i = 0; i < N; i++) {
		t = a[i];
		j = i;
		while (j > 0 && a[j - 1] > t) {
			a[j] = a[j - 1];
			j--;
		}
		a[j] = t;
	}
}

Mat get_mask(Mat& origin, Mat& clusterd_img, Rect roi, int k=0) {
	Mat hsv_img = clusterd_img.clone();
	Mat roi_cluster = clusterd_img.clone();
	cvtColor(clusterd_img, hsv_img, CV_BGR2HSV);
	Mat mask(origin.rows, origin.cols, CV_8UC1);
	mask = Scalar::all(0);
	imshow("clusterd_img", clusterd_img);


	int frequency_table[256] = { 0 };
	for (int i = roi.y; i < roi.y + roi.height; i++) {
		for (int j = roi.x; j < roi.x + roi.width; j++) {
			int idx = (hsv_img.at<Vec3b>(i, j)[2]);
			cout << idx << '\n';
			frequency_table[idx]++;
		}
	}

	int mode_label = 0; int max_fre = 0;

	int label_arr[256] = { 0 };
	for (int i = 0; i < 256; i++)
		label_arr[i] = i;

	vector<pair<int, int>> vr;
	for (int i = 0; i < 256; i++) {
		vr.push_back({ label_arr[i], frequency_table[i] });
	}
	sort(vr.begin(), vr.end(), [](pair<int, int> a, pair<int, int> b) {
		return a.second > b.second;
		});

	for (int i = 0; i < 256; i++) {
		if (frequency_table[i] > max_fre) {
			max_fre = frequency_table[i];
			mode_label = i;
		}
	}
	mode_label = vr[k].first;
	// roi 내에서 가장 많은 cluster label 은 mode_label 이다.
	//cout << "mode _ label : " << mode_label << " vr[0].frist = " << vr[0].first << '\n';
	for (int i = roi.y; i < roi.y + roi.height; i++) {
		for (int j = roi.x; j < roi.x + roi.width; j++) {
			if (i < 0 || i >= hsv_img.rows || j < 0 || j >= hsv_img.cols)
				continue;
			if (hsv_img.at<Vec3b>(i, j)[2] == mode_label) {
				mask.at<uchar>(i, j) = 255;
				roi_cluster.at<Vec3b>(i, j)[0] = 0;
				roi_cluster.at<Vec3b>(i, j)[1] = 255;
				roi_cluster.at<Vec3b>(i, j)[2] = 255;
			}
		}
	}

	cv::imshow("ROI", roi_cluster);
	return mask;
}

void dilation(Mat& input, Mat& result) {
	int height = input.rows; int width = input.cols;
	int size = 3;
	int structuring_element[9] = {  0,1,0,
									1,1,1,
									0,1,0 };
	int ele_size = 3;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			bool escape_double_loop_flag = false;
			bool fill_flag = false;
			for (int yy = y-1; yy <= y + 1; yy++) {
				for (int xx = x-1; xx <= x + 1; xx++) {
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue;
					if (input.at<uchar>(yy, xx) == 255 && structuring_element[(yy - (y - 1)) * ele_size + (xx - (x - 1))] == 1) {
						escape_double_loop_flag = true;
						fill_flag = true;
						break;
					}
				}
				if (escape_double_loop_flag == true) break;
			}
			if (fill_flag == true) {
				for (int yy = y; yy < y + size; yy++) {
					for (int xx = x; xx < x + size; xx++) {
						if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue;
						result.at<uchar>(yy, xx) = 255;
					}
				}
			}
		}
	}
}

void erosion(Mat& input, Mat& result) {
	int height = input.rows; int width = input.cols;
	int size = 3;
	int structuring_element[9] = { 0,1,0,
									1,1,1,
									0,1,0 };
	int ele_size = 3;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			bool escape_double_loop_flag = true;
			bool fill_flag = true;
			for (int yy = y - 1; yy <= y + 1; yy++) {
				for (int xx = x - 1; xx <= x + 1; xx++) {
					if (yy < 0 || yy >= height || xx < 0 || xx >= width) continue;
					if (input.at<uchar>(yy, xx) == 255 != structuring_element[(yy - (y - 1)) * ele_size + (xx - (x - 1))] == 1) {
						escape_double_loop_flag = false;
						fill_flag = false;
						break;
					}
				}
				if (escape_double_loop_flag == true) break;
			}
			if (fill_flag == true) {
				result.at<uchar>(y, x) = 255;
			}
		}
	}
}

Mat get_negative_mask(Mat& input, Rect roi) {
	Mat output(input.rows, input.cols, CV_8UC1);
	output = Scalar::all(0);
	for (int i = roi.y; i < roi.y+roi.height; i++) {
		for (int j = roi.x; j < roi.x+roi.width; j++) {
			output.at<uchar>(i, j) = 255 - input.at<uchar>(i, j);
		}
	}
	imshow("ooo", output);
	return output;
}

void gaussian_filtering(Mat& img, Mat& output, Rect roi) {
	output = img.clone();
	int gaussian_mask[9] = { 1,2,1,2,4,2,1,2,1 };
	for (int y = roi.y-10; y < roi.y+roi.height+10; y++) {
		for (int x = roi.x-10; x < roi.x+roi.width+10; x++) {
			if (y < 0 || y >= img.rows || x < 0 || x >= img.cols) continue;
			float conv_b = 0.0;
			float conv_g = 0.0;
			float conv_r = 0.0;

			for (int yy = y - 1; yy <= y + 1; yy++) {
				for (int xx = x - 1; xx <= x + 1; xx++) { // calc conv_x, conv_y
					if (yy >= 0 && yy < img.rows && xx >= 0 && xx < img.cols) {
						// indexing 에 주의!
						conv_b += img.at<Vec3b>(yy, xx)[0] * gaussian_mask[(yy - (y - 1)) * 3 + (xx - (x - 1))];
						conv_g += img.at<Vec3b>(yy, xx)[1] * gaussian_mask[(yy - (y - 1)) * 3 + (xx - (x - 1))];
						conv_r += img.at<Vec3b>(yy, xx)[2] * gaussian_mask[(yy - (y - 1)) * 3 + (xx - (x - 1))];
					}
				}
			}
			output.at<Vec3b>(y, x)[0] = conv_b / 16.0;
			output.at<Vec3b>(y, x)[1] = conv_g / 16.0;
			output.at<Vec3b>(y, x)[2] = conv_r / 16.0;
		}
	}
}

int main(int ac, char** av) {
	system("mode con cols=80 lines=20 | title K - means clustering based Inpainting tool");
	int k = 0;
	cout << "+===========================================================================+\n";
	cout << "|                   K - means clustering based Inpainting tool              |\n";
	cout << "|                  전기전자심화설계및SW실습(2022) Final project             |\n";
	cout << "|                         전기전자공학부 201810909 이재현                   |\n";
	cout << "+===========================================================================+\n";

	vector<string> file_list;
	std::filesystem::directory_iterator iter(std::filesystem::current_path() / "images/test");
	while (iter != std::filesystem::end(iter)) {
		const std::filesystem::directory_entry& entry = *iter;
		string ele = entry.path().string();
		file_list.push_back(ele);
		iter++;
	}
	int file_idx = 0; int path_idx = 0;
	cout << "+-----+---------------------------------------------------------------------+\n";
	cout << "| idx |                          file path                                  |\n";
	cout << "+-----+---------------------------------------------------------------------+\n";
	for (auto ele : file_list) {
		cout << "[" << file_idx++ << "] - ";
		cout << ele << '\n';
	}
	cout << "+---------------------------------------------------------------------------+\n";
	cout << "select input file (0~" << file_idx-1 << ")\n";
	cin >> path_idx;
	if (path_idx < 0 || path_idx >= file_idx) {
		cerr << "wrong input ...\n";
		return 0;
	}
	cin.ignore();

	string input_path = file_list[path_idx];

	system("cls");
	cout << "Please enter k value... (for k - means clustering) \n>>";
	cout << "[2<=k<=9 is recommended.] \n>>";
	cin >> k;
	cin.ignore();

	Mat frame = imread(input_path);
	resize(frame, frame, Size(600, 400));
	Mat mask(frame.rows, frame.cols, CV_8UC1);
	Mat final_mask(frame.rows, frame.cols, CV_8UC1);
	Mat final_result(frame.rows, frame.cols, CV_8UC3);
	mask = Scalar::all(0);
	imshow(main_window_name, frame);
	ITEMS param;
	param.frame = frame;
	param.drag_flag = false;
	param.update_flag = false;
	bool proceesing_flag = false;
	setMouseCallback(main_window_name, SetROI, &param);
	Mat k_means_result(frame.rows, frame.cols, CV_8UC3);
	K_means_clustering(frame, k_means_result, k, "L2");
	cout << "Image segmentation complete! \n";
	cout << "check your ROI region! \n";

	bool mask_setting = false;
	while (!mask_setting) {
		while (!param.update_flag) {
			imshow("k_means_result", k_means_result);
			imshow(main_window_name, frame);
			if (waitKey(10) >= 0) break;
		}
		char cluster_num = -1;
		while (cluster_num == -1) {
			system("cls");
			cout << "select cluster[0~k]\n";
			cout << "Type the \'q\' or \'Q\' key to continue!\n";
			cluster_num = waitKey(0);
			if (cluster_num == 113 || cluster_num == 124) { // q or Q
				cout << "Mask setup complete\n";
				mask_setting = true;
				break;
			}
		}
		if (!mask_setting){
			mask = get_mask(frame, k_means_result, param.roi_box, cluster_num - 48);
			imshow("mask", mask);
		}
		if(mask_setting){
			break;
		}

	}
	final_mask = Scalar::all(0);
	dilation(mask, final_mask);
	erosion(final_mask, final_mask);
	inpaint(frame, final_mask, final_result, 3, INPAINT_TELEA);
	gaussian_filtering(final_result, final_result, param.roi_box);
	gaussian_filtering(final_result, final_result, param.roi_box);

	imshow("final_mask", final_mask);
	imshow("mask", final_mask);
	imshow("final_result", final_result);
	cout << "Result image has been saved to \"images/result/out.jpg\"\n.";
	imwrite("images/result/out.jpg", final_result);

	cvWaitKey(0);

	return 0;
}
