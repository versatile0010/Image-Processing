#define _CRT_SECURE_NO_WARNINGS

// 2022. 10. 07. 전기전자공학부 이재현

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

#include <iostream>
#include <stdio.h>


using namespace cv;
using namespace std;


int main(int ac, char** av) {
	Mat frame;
	VideoCapture capture(0);
	if (!capture.isOpened()) {
		cerr << "Couldn't open the web camera...\n";
		return 0;
	}
	char filename[100];
	int frameNum = 0;

	while (true) {
		capture >> frame; // we can get an image frame.

		imshow("Video", frame);
		sprintf(filename, "./video_capture/frame%d.bmp", frameNum++);
		cout << filename << '\n';
		imwrite(filename, frame);
		if (waitKey(30) >= 0) break;
	}

	waitKey(0);

	return 0;
}
