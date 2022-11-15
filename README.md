#  :earth_americas: openCV-cpp
* This is a Course Project of ***ADVANCED ELECTRICAL AND ELECTRONIC DESIGN LABORATORY AND SOFTWARE PRACTICE(2022-Fall)*,** Konkuk Univ.

|Week|Topic|Description|
|------|---|---|
|Week 2.|[Fundamentals of image processing](https://github.com/versatile0010/openCV-cpp/tree/main/Week2)|Implementation of ImageResize, ImageRotate, Simple interpolation|
|Week 3.|[Edge-based feature extraction](https://github.com/versatile0010/openCV-cpp/tree/main/Week3)|Generate gradient map, Implementation of HOG Descriptor|
|Week 4.|[Corner-based feature extraction(1)](https://github.com/versatile0010/openCV-cpp/tree/main/Week4)|Implementation of Harris Corner detector|
|Week 5.|[Corner-based feature extraction(2)](https://github.com/versatile0010/openCV-cpp/tree/main/Week5)|ORB|
|Week 6.|[Optical flow](https://github.com/versatile0010/openCV-cpp/tree/main/Week6)|Lucas-Kanade Algorithm, Gunnar-farneback|
|Week 8.|[Face detection](https://github.com/versatile0010/openCV-cpp/tree/main/Week8)|Implementation of Face Detector with HOG Descriptor|
|Week 9.|[Face verification](https://github.com/versatile0010/openCV-cpp/tree/main/Week9)|Implementation of Face verification with LBP Descriptor|
|Week 10.|Facial landmark tracking|Implementation of Face Verification with Facial Landmarks Tracking|
|Week 11.|Image segmentation (1)|.|
|Week 12.|Image segmentation (2)|.|
|Week 13.|Image inpainting|.|
|Week 14.|Presentation of projects|.|
  

---
## Week 4. Feature matching with Harris Corner detector

![image](https://user-images.githubusercontent.com/96612168/200473264-63022518-f221-48f0-b009-d1a3f7db8e07.png)
![image](https://user-images.githubusercontent.com/96612168/200473312-3da226cf-bf7c-4ea9-8349-13c946a9fd68.png)
![image](https://user-images.githubusercontent.com/96612168/200473324-49c0fdc0-9c41-497c-b8c8-9afbd452da2b.png)
![image](https://user-images.githubusercontent.com/96612168/200473348-b79c8b7d-f9ea-4240-9ff3-1eb58d06651e.png)


Harris corner detector 으로 검출한 corner point 에 대해서
HOG descriptor 으로 ref.bmp 와 tar.bmp 의 유사도가 가장 좋은 corner point
를 matching 하는 프로그램을 구현

## Week 8. Face detector with HOG Descriptor
![image](https://user-images.githubusercontent.com/96612168/200472759-af1fd569-c05b-4015-8535-704b328f1dfa.png)

  HOG Descriptor 는 image 의 gradient 으로부터 maginitude 를 orientation 에 따라 vote 한 histogram 으로 feature 
을 정의한다. 두 이미지가 존재할 때, 각각의 HOG Descriptor 로부터 얻어진 histogram 을 비교
하여 유사도를 계산할 수 있다는 것이다. 
위 그림과 같이 reference image (Given model) 로 부터 HOG histogram 을 계산한 뒤, Window 
를 Terget image 에서 sliding 하며 HOG histogram similarity 를 계산한다. 이때 similarity 는
cosine similarity 를 사용하여 계산한다. 이때 중복 검출을 최대한 피하기 위해 non-maximum 
suppression 을 수행한 뒤, similarity 를 thresholding 하여 face detection 을 수행한다.

![image](https://user-images.githubusercontent.com/96612168/200472784-fc035784-d904-4127-b915-61297bfff43f.png)


## Week 9. Face Verification with LBP Descriptor
![image](https://user-images.githubusercontent.com/96612168/200472979-85936a86-0f57-49a4-8942-81a468925543.png)

  LBP Descriptor 을 활용한 Face verification 프로그램을 구현하였다.
reference face 에 대한 LBP histogram 을 저장한 다음, real-time 으로 detect 되는 face window
에 대하여 block 단위로 sliding 하는데, block 내의 center pixel 에 대하여 8 – neighbors 에
대하여 LBP 를 생성하는 방식을 통하여 target LBP histogram 을 구하고 Cosine similarity 를
thresholding 하여 face verification 을 진행하였다.
![image](https://user-images.githubusercontent.com/96612168/200472995-715e7069-bf3d-4fdf-a446-e0f7989e7dcf.png)
