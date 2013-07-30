#pragma once

#include "ofMain.h"
#include "ofxOpenCv.h"


class testApp : public ofBaseApp{
	public:
		void setup();
		void update();
		void draw();
		
		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y);
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);
        ofVideoGrabber 		vidGrabber;
        ofxCvColorImage			colorImg;
        ofxCvGrayscaleImage 	grayImage;
        ofImage PatternRead;
        ofxCvColorImage	PatternColor;
        ofxCvGrayscaleImage PatternGray;
        IplImage * TargetCV;
        IplImage *  InputCV;  // opencv image for input
        IplImage * ColorInputCV;// opencv image needed for Hue segmentation

        double G_value;
        ofTrueTypeFont  Fuente;
    IplImage * G_LevelsImage;
    int G_DetectMode; // contours detection mode
    cv::Mat G_DestinyForWarp;
  
  };
