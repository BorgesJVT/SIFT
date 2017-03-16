// Program to illustrate ORB keypoint and descriptor extraction, and matching using FLANN-LSH
// Author: Samarth Manoj Brahmbhatt, University of Pennsylvania
 
#include <opencv2/opencv.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <stdio.h>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;
 
int main() {
	cv::ocl::setUseOpenCL(true);
	//UMat img_object = imread("data/train.jpg").getUMat( ACCESS_READ );
	Mat img_object = imread("data/train.jpg");
	cvtColor(img_object, img_object, CV_BGR2GRAY);
	 
	//detect SIFT keypoints and extract descriptors in the train image
	int minHessian = 400;
    Ptr<SIFT> detector = SIFT::create( minHessian );
    std::vector<KeyPoint> keypoints_object;
    UMat descriptors_object;
	detector->detectAndCompute( img_object, Mat(), keypoints_object, descriptors_object );

	cout << "Descriptor depth " << descriptors_object.depth() << endl;
	
	// VideoCapture object
	VideoCapture cap(0);
	cap.set(CV_CAP_PROP_FPS, 30);
 
	unsigned int frame_count = 0;

	while(char(waitKey(1)) != 'q') {
		double t0 = getTickCount();
		UMat img_scene;
		cap >> img_scene;
		if(img_scene.empty()) continue;
		 
		cvtColor(img_scene, img_scene, CV_BGR2GRAY);
		 
		//detect SIFT keypoints and extract descriptors in the test image
		std::vector<KeyPoint> keypoints_scene;
    	UMat descriptors_scene;
		detector->detectAndCompute( img_scene, Mat(), keypoints_scene, descriptors_scene );

		// match train and test descriptors, getting 2 nearest neighbors for all test descriptors
		FlannBasedMatcher matcher;
	    std::vector< DMatch > matches;
		matcher.match( descriptors_object, descriptors_scene, matches );

		double max_dist = 0; double min_dist = 100;
	    //-- Quick calculation of max and min distances between keypoints
	    for( int i = 0; i < descriptors_object.rows; i++ )
	    { double dist = matches[i].distance;
			if( dist < min_dist ) min_dist = dist;
			if( dist > max_dist ) max_dist = dist;
	    }
	    //-- Draw only "good" matches (i.e. whose distance is less than 2*min_dist,
	    //-- or a small arbitary value ( 0.02 ) in the event that min_dist is very
	    //-- small)
	    //-- PS.- radiusMatch can also be used here.
	    std::vector< DMatch > good_matches;
	    for( int i = 0; i < descriptors_object.rows; i++ )
	    { if( matches[i].distance <= max(2*min_dist, 0.02) )
			{ good_matches.push_back( matches[i]); }
	    }

		Mat img_show;
  		drawMatches( img_object, keypoints_object, img_scene, keypoints_scene,
               good_matches, img_show, Scalar::all(-1), Scalar::all(-1),
               vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

  		imshow( "Good Matches", img_show );
		 
		cout << "Frame rate = " << getTickFrequency() / (getTickCount() - t0) << endl;
	}
	 
	return 0;
}

