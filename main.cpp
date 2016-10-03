/* OpenCV */
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/opencv.hpp"

/* librealsense library */
#include <librealsense/rs.hpp>

#include <iostream>
#include <ctype.h>

using namespace cv;
using namespace std;

/* arrowed LINE */
static void arrowedLine(InputOutputArray img, Point pt1, Point pt2, const Scalar& color, int thickness, int line_type, int shift, double tipLength) {
    const double tipSize = norm(pt1-pt2)*tipLength; // Factor to normalize the size of the tip depending on the length of the arrow
    line(img, pt1, pt2, color, thickness, line_type, shift);
    const double angle = atan2( (double) pt1.y - pt2.y, (double) pt1.x - pt2.x );
    Point p(cvRound(pt2.x + tipSize * cos(angle + CV_PI / 4)),
    cvRound(pt2.y + tipSize * sin(angle + CV_PI / 4)));
    line(img, p, pt2, color, thickness, line_type, shift);
    p.x = cvRound(pt2.x + tipSize * cos(angle - CV_PI / 4));
    p.y = cvRound(pt2.y + tipSize * sin(angle - CV_PI / 4));
    line(img, p, pt2, color, thickness, line_type, shift);
}


/* Creates the optFlow map*/
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, Mat& aux, int step, double, const Scalar& color) {

    for(int y = 0; y < cflowmap.rows; y += step) {
        for(int x = 0; x < cflowmap.cols; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x);
	    // Line is KEY!
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(255,0,0));
            circle(cflowmap, Point(x,y), 2, color, -1);

	    if( (fabs(fxy.x)>8) && fabs(fxy.x<15) && ( (fabs(fxy.y)>8 && fabs(fxy.y)<15) )) {		
		//line(aux, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(255,0,0));
		arrowedLine(aux, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), Scalar(255,255,255));

		cout << "You moved!" << endl;
		if (fxy.x < 0) cout << "right!" << endl; 
		else cout << "left!" << endl;
		if (fxy.y < 0) cout << "up!" << endl; 
		else cout << "down!" << endl;

	    }
        }
    }
}


/* Creates the HSV map*/
static void drawHsvMap(const Mat& flow, Mat& sflowmap) {

	cv::Mat xy[2];
	split(flow, xy);

	//calculate angle and magnitude
	Mat magnitude, angle;
	cartToPolar(xy[0], xy[1], magnitude, angle, true);

	//translate magnitude to range [0:1]
	double mag_max;
	minMaxLoc(magnitude, 0, &mag_max);
	magnitude.convertTo(magnitude, -1, 1.0/mag_max);

	//build hsv image
	Mat _hsv[3], hsv;
	_hsv[0] = angle;
	_hsv[1] = Mat::ones(angle.size(), CV_32F);
	_hsv[2] = magnitude;
	merge(_hsv, 3, hsv);
	// TODO check
	blur(hsv, hsv, Size(5,5));

	//convert to BGR and show
	Mat bgr;//CV_32FC3 matrix
	cvtColor(hsv, bgr, COLOR_HSV2BGR);
	bgr.copyTo(sflowmap);
}


/* Main */
int main( int argc, char** argv )
try {

	// Turn on logging.
	rs::log_to_console(rs::log_severity::warn);
	std::cout << "Starting..." << std::endl;

	// realsense contextremap
	rs::context ctx;
        cout << "There are " << ctx.get_device_count() << " connected RealSense devices." << endl << endl;

	// exit if not device is already connected
        if (ctx.get_device_count() == 0) return EXIT_FAILURE;

	// rs defining device to be used
	rs::device * dev = ctx.get_device(0);

	// configure RGB to run at 60 frames per second
	dev->enable_stream(rs::stream::color, 640, 480, rs::format::rgb8, 60);
	// configure DEPTH to run at 60 frames per second
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 60);

	// start the device
	dev->start();

	// OpenCV frame definition.
	cv::Mat img, gray, flow, sflow, shsv, aux;
        // some faster than mat image container
 	UMat prevgray;

	// capture first 50 frames to allow camera to stabilize
	for (int i = 0; i < 50; ++i) dev->wait_for_frames();


	// loop -- DATA ACQUISITION
	while (1) {

		// wait for new frame data
		dev->wait_for_frames();

		// RGB data acquisition
		uchar *rgb = (uchar *) dev->get_frame_data(rs::stream::color);
		// DEPTH data acquisition
		uchar *depth = (uchar *) dev->get_frame_data(rs::stream::depth);

		// data acquisition into opencv::Mat
		// RGB
		const uint8_t * rgb_frame = reinterpret_cast<const uint8_t *>(rgb);
		img = cv::Mat(480, 640, CV_8UC3, (void*) rgb_frame);
		//cvtColor(img, img, CV_BGR2RGB); // saving data into cv::mat container img

		// just make current frame gray
   		cvtColor(img, gray, COLOR_BGR2GRAY);

		if (!prevgray.empty()) {

			flow = Mat(gray.size(), CV_32FC2);
			sflow = Mat(gray.size(), CV_8UC3);

			// sflow to gray
			cvtColor(prevgray, sflow, COLOR_GRAY2BGR);
			// shsv to gray
			cvtColor(prevgray, shsv, COLOR_GRAY2BGR);

			// applying FlowFarneback
			calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);

			aux = Mat::ones(flow.size(), CV_8U);
			drawOptFlowMap(flow, sflow, aux, 16, 1.5, Scalar(0, 255, 0));
            imshow("flow", sflow);
			imshow("bw rows", aux);

			//drawHsvMap(flow, shsv);
			//imshow("hsv", shsv);
	

			// movingPoints vector
			std::vector<cv::Point2f> movingPoints;
			
			for(int y = 0; y < flow.rows; y += 16) {
				for(int x = 0; x < flow.cols; x += 16) {
					const Point2f& f = flow.at<Point2f>(y, x);
					// condition to take points in account
					if(fabs(f.x)>8 && fabs(f.y)>8) movingPoints.push_back(cv::Point2f(x, y));
				}
			}

			int K = 2;
			Mat centers, labels, res;
			if(movingPoints.size() >= K) {
				cv::Mat movingPointsMatrix(movingPoints, false); //second param for data copy (here, data are not duplicated!)

				//cout << "movingPointsMatrix.size(): " << movingPointsMatrix.size() << endl << endl;
				//cout << "movingPointsMatrix: " << movingPointsMatrix << endl << endl;

				cv::kmeans(movingPointsMatrix, K, labels, TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10, 1.0), 3, KMEANS_PP_CENTERS, centers);

				int colors[K];
				for (int i = 0; i < K; i++) {
					colors[i] = 250.0 + (float) 250 / (i + 1);
				}

				Mat clustered = Mat(flow.rows, flow.cols, CV_32F); 

				for (int i = 0; i < K; i++) {
					Point2f center(centers.at<float>(i,0), centers.at<float>(i,1));
					cout << "center: "<< center << endl;
					circle(clustered, center, 2, Scalar(colors[i], 0, 0), -1);
				}

				clustered.convertTo(clustered, CV_8U);
				imshow("Kmeans", clustered);
			}

			//  updateMotionHistory(silh, mhi, timestamp, MHI_DURATION);
			//  calcMotionGradient(mhi, mask, orient, MAX_TIME_DELTA, MIN_TIME_DELTA, 3);
			//  segmentMotion(mhi, segmask, regions, timestamp, MAX_TIME_DELTA);
		} else {
		         // update previus image
	    		 gray.copyTo(prevgray); 
		}


		if( (char)waitKey(10) == 27 )
		    break;

		// update previous image
		gray.copyTo(prevgray);
    }

    return 0;
} catch (const rs::error & e) {
	// method calls against librealsense objects may throw exceptions of type rs::error
	cout << "rs::error was thrown when calling " << e.get_failed_function().c_str()
	<< "-" << e.get_failed_args().c_str() << "-:     " << e.what() << endl;
	return EXIT_FAILURE;
}
