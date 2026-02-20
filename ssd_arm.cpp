#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

int main()
{
    // -----------------------------
    // 1. Load Model
    // -----------------------------
    string model = "mobilenet-ssd.caffemodel";
    string config = "mobilenet-ssd.prototxt";

    Net net = readNetFromCaffe(config, model);

    if (net.empty()) {
        cout << "Error: Could not load DNN model!" << endl;
        return -1;
    }

    cout << "Model loaded successfully." << endl;

    // -----------------------------
    // 2. Open Camera
    // -----------------------------
    VideoCapture cap(0);  // Use default camera

    if (!cap.isOpened()) {
        cout << "Error: Could not open camera!" << endl;
        return -1;
    }

    cout << "Camera opened successfully." << endl;

    // VOC Class Names
    vector<string> classNames = {
        "background","aeroplane","bicycle","bird","boat",
        "bottle","bus","car","cat","chair",
        "cow","diningtable","dog","horse",
        "motorbike","person","pottedplant",
        "sheep","sofa","train","tvmonitor"
    };

    const float confidenceThreshold = 0.5;

    Mat frame;

    // -----------------------------
    // 3. Detection Loop
    // -----------------------------
    while (true)
    {
        cap >> frame;
        if (frame.empty())
            break;

        double start = (double)getTickCount();

        // Create blob
        Mat blob = blobFromImage(frame,
                                 0.007843,
                                 Size(300,300),
                                 Scalar(127.5,127.5,127.5),
                                 false);

        net.setInput(blob);
        Mat detections = net.forward();

        Mat detectionMat(detections.size[2],
                         detections.size[3],
                         CV_32F,
                         detections.ptr<float>());

        int personCount = 0;
        int carCount = 0;

        for (int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i,2);

            if (confidence > confidenceThreshold)
            {
                int classId = (int)detectionMat.at<float>(i,1);

                int x1 = int(detectionMat.at<float>(i,3) * frame.cols);
                int y1 = int(detectionMat.at<float>(i,4) * frame.rows);
                int x2 = int(detectionMat.at<float>(i,5) * frame.cols);
                int y2 = int(detectionMat.at<float>(i,6) * frame.rows);

                // Boundary check
                x1 = max(0, x1);
                y1 = max(0, y1);
                x2 = min(frame.cols - 1, x2);
                y2 = min(frame.rows - 1, y2);

                Rect box(Point(x1,y1), Point(x2,y2));

                // Count persons and cars
                if (classNames[classId] == "person")
                    personCount++;

                if (classNames[classId] == "car")
                    carCount++;

                rectangle(frame, box, Scalar(0,255,0), 2);

                string label = classNames[classId] +
                               " : " + to_string(confidence).substr(0,4);

                putText(frame,
                        label,
                        Point(x1, y1 - 5),
                        FONT_HERSHEY_SIMPLEX,
                        0.5,
                        Scalar(0,255,0),
                        2);
            }
        }

        // Calculate FPS
        double time = ((double)getTickCount() - start) / getTickFrequency();
        double fps = 1.0 / time;

        putText(frame,
                "FPS: " + to_string((int)fps),
                Point(10,30),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(0,0,255),
                2);

        putText(frame,
                "Persons: " + to_string(personCount),
                Point(10,60),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(255,0,0),
                2);

        putText(frame,
                "Cars: " + to_string(carCount),
                Point(10,90),
                FONT_HERSHEY_SIMPLEX,
                0.8,
                Scalar(255,255,0),
                2);

        imshow("MobileNet-SSD Detection", frame);

        if (waitKey(1) == 27) // ESC
            break;
    }

    cap.release();
    destroyAllWindows();

    return 0;
}
