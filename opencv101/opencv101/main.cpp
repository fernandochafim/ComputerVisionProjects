#include <opencv2/opencv.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

void detectAndDisplay(Mat frame);

CascadeClassifier faceCascade;
CascadeClassifier eyeCascade;

int main(int argc, const char** argv)
{
	
	CommandLineParser parser(argc, argv,
		"{help h||}"
		"{faceCascade|data/haarcascades/haarcascade_frontalface_alt.xml|Path to face cascade.}"
		"{eyeCascade|data/haarcascades/haarcascade_eye_tree_eyeglasses.xml|Path to eyes cascade.}"
		"{camera|0|Camera device number.}");

	parser.about("\nThis program demonstrates using the cv::CascadeClassifier class to detect objects (Face + eyes) in a video stream.\n"
		"You can use Haar or LBP features.\n\n");
	parser.printMessage();

	String faceCascade_name = samples::findFile(parser.get<String>("faceCascade"));
	String eyeCascade_name = samples::findFile(parser.get<String>("eyeCascade"));

	//-- 1. Load the cascades
	if (!faceCascade.load(faceCascade_name))
	{
		cout << "--(!)Error loading face cascade\n";
		return -1;
	};
	if (!eyeCascade.load(eyeCascade_name))
	{
		cout << "--(!)Error loading eye cascade\n";
		return -1;
	};

	// temp
	//faceCascade.load("D:\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml");
	//eyeCascade.load("D:\\Downloads\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml");
	
	
	int camera_device = parser.get<int>("camera");
	VideoCapture capture;
	//VideoCapture capture(0);
	//-- 2. Read the video stream
	capture.open(camera_device);

	if (!capture.isOpened())
	{
		cout << "--(!)Error opening video capture\n";
		return -1;
	}

	Mat frame;
	while (capture.read(frame))
	{
		if (frame.empty())
		{
			cout << "--(!) No captured frame -- Break!\n";
			break;
		}

		//-- 3. Apply the classifier to the frame
		detectAndDisplay(frame);
		if (waitKey(10) == 27)
		{
			break; // escape
		}
	}

	return 0;
}


void detectAndDisplay(Mat frame)
{
	Mat grayscale;
	//-- Convert to frame to Grayscale
	cvtColor(frame, grayscale, COLOR_BGR2GRAY);
	
	//-- Equalize histograms of images
	equalizeHist(grayscale, grayscale);

	//-- Detect faces
	vector<Rect> faces;
	faceCascade.detectMultiScale(grayscale, faces);
	
	double scale = 1.0;

	for (size_t i = 0; i < faces.size(); i++)
	{
		Rect facesArea = cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);

		// Draw rectangle in face
		Scalar drawBlueColor = Scalar(255, 0, 0);
		Scalar drawRedColor = Scalar(0, 0, 255);
		
		Point centerx11(cvRound(facesArea.x * scale), cvRound(facesArea.y * scale));
		Point centerx22(cvRound((facesArea.x + facesArea.width - 1) * scale), cvRound((facesArea.y + facesArea.height - 1) * scale));
		
		rectangle(frame, centerx11, centerx22, drawBlueColor);

		
		Mat faceROI = grayscale(facesArea);
		//-- In each face, detect eyes
		vector<Rect> eyes;
		eyeCascade.detectMultiScale(faceROI, eyes);
		for (size_t j = 0; j < eyes.size(); j++)
		{
			Rect eyesArea = cv::Rect(eyes[j].x, eyes[j].y, eyes[j].width, eyes[j].height);
		
			Point eyesCenter(facesArea.x + eyesArea.x + eyesArea.width / 2, facesArea.y + eyesArea.y + eyesArea.height / 2);
			int radius = cvRound((eyesArea.width + eyesArea.height)*0.25);
			circle(frame, eyesCenter, radius, drawRedColor, 4);
		}
	}
	//-- Show what you got
	imshow("OpenCV4 - Face and eye detection", frame);
}