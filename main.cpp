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


/* Creates the optFlow map*/
static void drawOptFlowMap(const Mat& flow, Mat& cflowmap, int step, double, const Scalar& color) {
    for(int y = 0; y < cflowmap.rows; y += step) {
        for(int x = 0; x < cflowmap.cols; x += step) {
            const Point2f& fxy = flow.at<Point2f>(y, x)*10;
            line(cflowmap, Point(x,y), Point(cvRound(x+fxy.x), cvRound(y+fxy.y)), color);
            circle(cflowmap, Point(x,y), 2, color, -1);
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
	//cout << "mag: " << magnitude << endl;

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
	//imshow("flow", bgr);
	bgr.copyTo(sflowmap);
}

/* Main */
int main( int argc, char** argv )
try {

	// Turn on logging.
	rs::log_to_console(rs::log_severity::warn);
	std::cout << "Starting..." << std::endl;

	// realsense context
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
	cv::Mat img, gray, flow, frame, sflow, shsv;
        // some faster than mat image container
 	UMat flowUmat, prevgray;

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
		// TODO check conversions
		//cv::Mat rgb_ = cv::Mat(480, 640, CV_8UC3, (void*) rgb_frame);
		//cvtColor(rgb_, img, CV_BGR2RGB); // saving data into cv::mat container img

		// just make current frame gray
   		cvtColor(img, gray, COLOR_BGR2GRAY);

		if (!prevgray.empty()) {
			// applying FlowFareback
			calcOpticalFlowFarneback(prevgray, gray, flowUmat, 0.5, 3, 15, 3, 5, 1.2, 0);
			// sflow to gray
			cvtColor(prevgray, sflow, COLOR_GRAY2BGR);
			// shsv to gray TODO check initial value in SHSV
			cvtColor(prevgray, shsv, COLOR_GRAY2BGR);
    			// copy Umat container to standard Mat
    			flowUmat.copyTo(flow);

			drawOptFlowMap(flow, sflow, 16, 1.5, Scalar(0, 255, 0));
            		imshow("flow", sflow);

			drawHsvMap(flow, shsv);
			imshow("hsv", shsv);


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
