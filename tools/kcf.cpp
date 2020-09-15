#include <opencv2/core/utility.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <cstring>
 
using namespace std;
using namespace cv;
 
int main()
{
    Rect2d roi;
    Mat frame;
    Ptr<TrackerKCF> tracker = TrackerKCF::create();
    string video = "./@20200818125523_20200818125735_5984.mp4";

    VideoCapture cap(video);
    if (!cap.isOpened())
    {
        return 0;
    }
    cout << "press c to leap current Image" << endl;
    cout << "press q to slect current Image" << endl;
    cout << "press empty key to start track RIO Object" << endl;
 
    cap >> frame;
    while (1)
    {
        char key = waitKey(1);
        if (key == 'c')  // 按c键跳帧
        {
            cap >> frame;
        }
        if (key == 'q')  // 按q键退出跳帧
        {
            break;
        }
        imshow("first", frame);
    }
 
    cv::destroyWindow("first");
 
    roi = selectROI("tracker", frame);
 
    if (roi.width == 0 || roi.height == 0)
        return 0;
 
    tracker->init(frame, roi);
 
    // perform the tracking process
    printf("Start the tracking process\n");
    for (;; )
    {
        // get frame from the video
        cap >> frame;
 
        // stop the program if no more images
        if (frame.rows == 0 || frame.cols == 0) {
            cv::destroyWindow("tracker");
            break;
        }
 
        // update the tracking result
 
        tracker->update(frame, roi);
 
        // draw the tracked object
        rectangle(frame, roi, Scalar(255, 0, 0), 2, 1);
 
        // show image with the tracked object
        imshow("tracker", frame);
 
        //quit on ESC button
        if (char(waitKey(1)) == 'q') {
            cv::destroyWindow("tracker");
            break;
        }
    }
    return 0;
}