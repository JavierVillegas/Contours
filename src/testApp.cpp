#include "testApp.h"

#define Nx 640
#define Ny 480

int GrayLevels = 6;

const int FFTSize = 64;
 ofxCvColorImage Mirona;
// best contours
vector <ofVec2f> BestMatches;
CvSeq* G_TargContour;
CvMemStorage* G_storage;
   CvMat *MagniPattern;
CvMat *Parapintar;
double HuFunction(CvSeq* TheConts, double minCuanto,int mode);
double FFTFunction(CvSeq* TheConts, double minCuanto, int mode);
CvBox2D BestBox;
int G_PointOffset;
bool G_edges = false;
bool G_smooth = false;
int G_input =0;
//--------------------------------------------------------------
void testApp::setup(){
    
    vidGrabber.setVerbose(true);
    vidGrabber.initGrabber(Nx,Ny);
    colorImg.allocate(Nx,Ny);
	grayImage.allocate(Nx,Ny);
    PatternRead.loadImage("two.png");
    PatternColor.allocate(PatternRead.width, PatternRead.height);
    PatternColor.setFromPixels(PatternRead.getPixels(),PatternRead.width,PatternRead.height);
    PatternGray.allocate(PatternRead.width, PatternRead.height);
    PatternGray = PatternColor;
  //  TargetCV.create(PatternRead.width, PatternRead.height, CV_8U);
    TargetCV = cvCreateImage(cvSize(PatternRead.width, PatternRead.height), IPL_DEPTH_8U, 1);
    InputCV = cvCreateImage(cvSize(Nx, Ny), IPL_DEPTH_8U, 1);
    G_LevelsImage = cvCreateImage(cvSize(Nx, Ny), IPL_DEPTH_8U, 1);
    TargetCV = PatternGray.getCvImage();

   
    G_storage = cvCreateMemStorage(0);
    G_TargContour =0;
    cvNot(TargetCV, TargetCV);
    
    cvFindContours(TargetCV,G_storage,&G_TargContour,sizeof(CvContour),
                   CV_RETR_EXTERNAL,CV_CHAIN_APPROX_NONE);
    
  //  for( ; G_TargContour != 0; G_TargContour = G_TargContour->h_next )
  //  {
    CvMat* PatternFFT;
    
    Parapintar = cvCreateMat(FFTSize, 1, CV_32FC1);
        PatternFFT = cvCreateMat(FFTSize, 1, CV_32FC2);
      // temporat mats
    CvMat *TempReal;
    CvMat *TempImag;
 
    
   TempReal =  cvCreateMat(FFTSize, 1, CV_32FC1);
   TempImag =  cvCreateMat(FFTSize, 1, CV_32FC1);
   MagniPattern =  cvCreateMat(FFTSize, 1, CV_32FC1);
    
    
    if (G_TargContour->total>20){
            cvDrawContours( TargetCV, G_TargContour, CV_RGB(100,100,100),
                           CV_RGB(100,100,100), 0, 3, 8 );
  
            //Getting The FFT of the input contour.
            
            CvPoint2D32f * OtroArray;
            OtroArray = new CvPoint2D32f[G_TargContour->total];
                    for(int q=0;q<G_TargContour->total;q++){
                       CvPoint* p = (CvPoint*)cvGetSeqElem( G_TargContour, q );
                       OtroArray[q].x = (float)p->x;
                       OtroArray[q].y=(float)p->y;
                    }
            CvMat* CountArray;
            CountArray=cvCreateMatHeader(G_TargContour->total,1,CV_32FC2);
            cvInitMatHeader(CountArray,G_TargContour->total,1,CV_32FC2,OtroArray);
            cvResize(CountArray, PatternFFT);
            // FFT
            cvDFT(PatternFFT, PatternFFT, CV_DXT_FORWARD);
            // Real imag parts
            cvSplit(PatternFFT, TempReal, TempImag, NULL, NULL);
            cvCartToPolar(TempReal,TempImag,MagniPattern,NULL);
            // eliminating first element
            cvSet1D(MagniPattern, 0, cvScalarAll(0));
            // normalizing with the first
            CvScalar C1Scalar;
            C1Scalar = cvGet1D(MagniPattern, 1);
        cvSetZero(TempImag);
        if (C1Scalar.val[0]!=0){
            cvScaleAdd(MagniPattern, cvScalarAll(1/(C1Scalar.val[0])), TempImag, MagniPattern);
        }
  // Normalizing with the Energy
 //       double NormFac;
 //       NormFac = cvNorm(MagniPattern,NULL,CV_L2);
//        cvSetZero(TempImag);
//        if (NormFac!=0){
//            cvScaleAdd(MagniPattern, cvScalarAll(1/(NormFac)), TempImag, MagniPattern);
//            
//        }
            cvReleaseMat(&CountArray);
        cvReleaseMat(&TempReal);
        cvReleaseMat(&TempImag);
        cvReleaseMat(&PatternFFT);
        
            delete [] OtroArray;
        }
   // }
    
    
    Fuente.loadFont("helvetica.ttf", 32);
    
    G_DetectMode =5;  //Hu  moments modes 0-2
                      // Fourier Descriptors 3-8

    BestBox.center =cvPoint2D32f(100, 100);
    BestBox.angle = 0;
    BestBox.size =cvSize2D32f(50, 50);
    
}

//--------------------------------------------------------------
void testApp::update(){
    bool bNewFrame = false;
    
    
    vidGrabber.update();
    bNewFrame = vidGrabber.isFrameNew();
    
	if (bNewFrame){
        
        
        colorImg.setFromPixels(vidGrabber.getPixels(), Nx,Ny);
        grayImage = colorImg;
        ColorInputCV = colorImg.getCvImage();
        //TheInput = colorImg.getCvImage();
        InputCV = grayImage.getCvImage();
        
        

        CvMemStorage* storage2;
        CvSeq* contodos;
        
        storage2 = cvCreateMemStorage(0);
        contodos = 0;
        IplImage * BINARY;
        IplImage * COLORFULONE;
        IplImage * AuxPaint;
        
        int countcont;
        COLORFULONE= cvCreateImage(cvGetSize(InputCV),IPL_DEPTH_8U,3);
        BINARY = cvCreateImage(cvGetSize(InputCV),IPL_DEPTH_8U,1);
         AuxPaint = cvCreateImage(cvGetSize(InputCV),IPL_DEPTH_8U,1);
        
        // whiteImage
        cvSet( AuxPaint, cvScalarAll(255));
        cvSet(G_LevelsImage, cvScalarAll(255));
       
        
        if(G_input==0){ // using grayscale values
       
            // gray levels
            cvCvtColor(InputCV,COLORFULONE,CV_GRAY2BGR);
        }
        else{ // using the hue plane
            cvCvtColor(ColorInputCV,COLORFULONE,CV_RGB2HSV);
            cvSplit(COLORFULONE, InputCV, NULL, NULL, NULL);
            cvCopy(ColorInputCV, COLORFULONE);
        }
        
        

        double minCuanto=100000000000;
        
        for (int k=1; k< GrayLevels; k++) {
      
            
            if (G_smooth){
                cvSmooth(InputCV, InputCV,CV_GAUSSIAN,5);
            
            }
            
            
            if (G_edges){
                cvCanny(InputCV, BINARY, 50, 100);
               cvSetZero(AuxPaint);
            }
            else{
                cvInRangeS(InputCV,cvScalarAll((k-1)*255/(float)GrayLevels),cvScalarAll(k*255/(float)GrayLevels),BINARY);
            cvSet(AuxPaint, cvScalarAll((k-1)*255/(float)GrayLevels));
            }
            //   cvNot(BINARY, BINARY);
         // cvThreshold(InputCV, BINARY, 100, 255, CV_THRESH_BINARY_INV);
         //  cvSmooth(BINARY,BINARY,CV_MEDIAN,3);
            
            cvCopy(AuxPaint, G_LevelsImage,BINARY);
        

            countcont = cvFindContours(BINARY,storage2,&contodos,sizeof(CvContour),
                           CV_RETR_LIST,CV_CHAIN_APPROX_NONE );
       
            
            
            for( ; contodos != 0; contodos = contodos->h_next )
            {
                if ((contodos->total>10)&&(contodos->total<800)){
                    if (G_DetectMode <3) {
                     minCuanto=HuFunction(contodos,minCuanto, G_DetectMode);
                    }
                    else{
                      minCuanto=FFTFunction(contodos,minCuanto, G_DetectMode);
                    
                    }
                }// ent if contour is long enought
            } // end for countours
            
           contodos = 0;
            
            
        } // end for gray levels
        
       
        G_value = minCuanto;

        
        Mirona.allocate(COLORFULONE->width, COLORFULONE->height);
        
        Mirona = COLORFULONE;
        
        if (contodos!=NULL){cvClearSeq(contodos);}
        if (storage2!=NULL){cvReleaseMemStorage(&storage2);}
        
        cvReleaseImage(&BINARY);
        cvReleaseImage(&COLORFULONE);
        cvReleaseImage(&AuxPaint);
    }

}




//--------------------------------------------------------------
void testApp::draw(){
	ofSetHexColor(0xffffff);

    
    
    ofxCvGrayscaleImage AuxGrayImage;
    AuxGrayImage.allocate(TargetCV->width, TargetCV->height);
    AuxGrayImage = TargetCV;
    AuxGrayImage.draw(0, 0);
    Mirona.draw(AuxGrayImage.width,0);

    ofxCvGrayscaleImage AuxGrayImage2;
    AuxGrayImage2.allocate(G_LevelsImage->width, G_LevelsImage->height);
    AuxGrayImage2 = G_LevelsImage;
    AuxGrayImage2.draw(AuxGrayImage.width, Mirona.height);

    
    ofSetColor(200, 0, 0);
    glBegin(GL_LINE_STRIP);
    
    for (int k = 0; k < BestMatches.size();k++){
        glVertex2f(BestMatches[k].x+AuxGrayImage.width, BestMatches[k].y);
    }
     glEnd();
 
    glBegin(GL_LINE_STRIP);
    
    for (int k = 0; k < BestMatches.size();k++){
        glVertex2f(BestMatches[k].x+AuxGrayImage.width, BestMatches[k].y + Mirona.height);
    }
    glEnd();
    
   // Fuente.drawString(ofToString(G_value), AuxGrayImage.width+Mirona.width+40, 40);
   Fuente.drawString(ofToString(BestMatches.size()), AuxGrayImage.width+Mirona.width+40, 40);
    
    Fuente.drawString("Mode: "+ ofToString(G_DetectMode), AuxGrayImage.width+Mirona.width+40, 80);
    
    
    cv::RotatedRect ThePlace(BestBox);
    cv::Point2f vertiPlot[4];
    ThePlace.points(vertiPlot);
    ofSetColor(0, 200, 0);
    ofNoFill();
    ofLine(AuxGrayImage.width+vertiPlot[0].x, vertiPlot[0].y, AuxGrayImage.width+vertiPlot[1].x, vertiPlot[1].y);
    ofLine(AuxGrayImage.width+vertiPlot[1].x, vertiPlot[1].y, AuxGrayImage.width+vertiPlot[2].x, vertiPlot[2].y);
    ofLine(AuxGrayImage.width+vertiPlot[2].x, vertiPlot[2].y, AuxGrayImage.width+vertiPlot[3].x, vertiPlot[3].y);
    ofLine(AuxGrayImage.width+vertiPlot[3].x, vertiPlot[3].y, AuxGrayImage.width+vertiPlot[0].x, vertiPlot[0].y);
    
   
    
    vector<cv::Point2f> scene_corners(4);
    scene_corners[0] =vertiPlot[(0+G_PointOffset)%4];
    scene_corners[1] =vertiPlot[(1+G_PointOffset)%4];
    scene_corners[2] =vertiPlot[(2+G_PointOffset)%4];
    scene_corners[3] =vertiPlot[(3+G_PointOffset)%4];
    
    vector<cv::Point2f> Rewarp_corners(4);
    Rewarp_corners[0] = cvPoint(350,350);
    Rewarp_corners[1] = cvPoint(450,350);
    Rewarp_corners[2] = cvPoint(450,450);
    Rewarp_corners[3] = cvPoint(350,450);
    
    cv::Mat newH =findHomography( scene_corners, Rewarp_corners, CV_RANSAC );
    
    // the source image
    
    cv::Mat InputMAT(InputCV);
    cv::warpPerspective(InputMAT, G_DestinyForWarp, newH, cvSize(800, 800));
    
    
    
    ofSetColor(255, 255, 255);
    ofxCvGrayscaleImage ReWarp;

    ReWarp.allocate(G_DestinyForWarp.cols, G_DestinyForWarp.rows);
    ReWarp = G_DestinyForWarp.data;
    ReWarp.draw(AuxGrayImage.width+Mirona.width+40,150,800,800);
    
    ofSetColor(0, 200, 0);
    ofLine(AuxGrayImage.width+Mirona.width+40+Rewarp_corners[0].x, Rewarp_corners[0].y+150,
           AuxGrayImage.width+Mirona.width+40+Rewarp_corners[1].x, Rewarp_corners[1].y+150);
    ofLine(AuxGrayImage.width+Mirona.width+40+Rewarp_corners[1].x, Rewarp_corners[1].y+150,
           AuxGrayImage.width+Mirona.width+40+Rewarp_corners[2].x, Rewarp_corners[2].y+150);
    ofLine(AuxGrayImage.width+Mirona.width+40+Rewarp_corners[2].x, Rewarp_corners[2].y+150,
           AuxGrayImage.width+Mirona.width+40+Rewarp_corners[3].x, Rewarp_corners[3].y+150);
    ofLine(AuxGrayImage.width+Mirona.width+40+Rewarp_corners[3].x, Rewarp_corners[3].y+150,
           AuxGrayImage.width+Mirona.width+40+Rewarp_corners[0].x, Rewarp_corners[0].y+150);
        
    
    
    
//    // draw the FFT bins
//
//    float YmaxA =100;
//    float yOffset =200;
//    float Binwidth = 800/FFTSize;
//    CvScalar Tempscalar1,Tempscalar2;
//    for (int k = 0; k < FFTSize ; k++){
//        Tempscalar1 = cvGet1D(MagniPattern, k);
//        ofSetColor(200, 0, 0);
//        ofNoFill();
//        ofRect(AuxGrayImage.width+Mirona.width+40 + k*Binwidth,
//               yOffset+YmaxA -Tempscalar1.val[0]*50,Binwidth,Tempscalar1.val[0]*50);
//        if (G_DetectMode>2){
//            ofSetColor(0, 0, 200);
//            Tempscalar2 = cvGet1D(Parapintar, k);
//            ofRect(AuxGrayImage.width+Mirona.width+40 + k*Binwidth,
//                   yOffset+YmaxA -Tempscalar2.val[0]*50,Binwidth,Tempscalar2.val[0]*50);
//       
//            // Ploting The Error
//            ofSetColor(0, 200, 0);
//            ofFill();
//            ofRect(AuxGrayImage.width+Mirona.width+40 + k*Binwidth,
//                   200+yOffset+YmaxA -abs(Tempscalar2.val[0]*50 -Tempscalar1.val[0]*50)
//                                          ,Binwidth,abs(Tempscalar2.val[0]*50 -Tempscalar1.val[0]*50));
//        
//        }
//
//    
//   }
}




// Detection functions

double HuFunction(CvSeq* TheConts, double minCuanto, int mode){
    double cuanto;
    if (mode==0){
        cuanto=cvMatchShapes(G_TargContour,TheConts,CV_CONTOURS_MATCH_I1,0);
    }
    else if (mode==1){
        cuanto=cvMatchShapes(G_TargContour,TheConts,CV_CONTOURS_MATCH_I2,0);
    }
    else if (mode==2){
        cuanto=cvMatchShapes(G_TargContour,TheConts,CV_CONTOURS_MATCH_I3,0);
    }
    if (cuanto < minCuanto){
        BestMatches.clear();
        minCuanto = cuanto;
        BestBox = cvFitEllipse2(TheConts);
        for(int q=0;q<TheConts->total;q++){
            CvPoint* p = (CvPoint*)cvGetSeqElem( TheConts, q );
            ofVec2f tempopoint;
            tempopoint.x = (float)p->x;
            tempopoint.y = (float)p->y;
            BestMatches.push_back(tempopoint);
            
        }
    }// end if cuant <minCuanto
    
    return minCuanto;
}

//////////FFT Function///////



double FFTFunction(CvSeq* TheConts, double minCuanto, int mode){
    double cuanto;
    // getting The FFT of the input contour
    CvMat* LocalFFT;
    CvMat *TempReal;
    CvMat *TempImag;
    CvMat *LocalMag;
    
    TempReal =  cvCreateMat(FFTSize, 1, CV_32FC1);
    TempImag =  cvCreateMat(FFTSize, 1, CV_32FC1);
    LocalMag =  cvCreateMat(FFTSize, 1, CV_32FC1);
    
    
    CvPoint2D32f * OtroArray;
    LocalFFT = cvCreateMat(FFTSize, 1, CV_32FC2);
    OtroArray = new CvPoint2D32f[TheConts->total];
    for(int q=0;q<TheConts->total;q++){
        CvPoint* p = (CvPoint*)cvGetSeqElem( TheConts, q );
        OtroArray[q].x = (float)p->x;
        OtroArray[q].y=(float)p->y;
    }
    CvMat* CountArray;
    
    CountArray=cvCreateMatHeader(TheConts->total,1,CV_32FC2);
    cvInitMatHeader(CountArray,TheConts->total,1,CV_32FC2,OtroArray);
    cvResize(CountArray, LocalFFT);
    cvDFT(LocalFFT, LocalFFT, CV_DXT_FORWARD);
    
     // Real imag parts
    cvSplit(LocalFFT, TempReal, TempImag, NULL, NULL);
    cvCartToPolar(TempReal,TempImag,LocalMag,NULL);
    // eliminating first element
    cvSet1D(LocalMag, 0, cvScalarAll(0));
    // normalizing with the first
    CvScalar C1Scalar;
    C1Scalar = cvGet1D(LocalMag, 1);
    
    cvSetZero(TempImag);
    if (C1Scalar.val[0]!=0){
        cvScaleAdd(LocalMag, cvScalarAll(1/(C1Scalar.val[0])), TempImag, LocalMag);
    }
        // Normalicing to the vectors energy
//    double NormFac;
//    NormFac = cvNorm(LocalMag,NULL,CV_L2);
//    cvSetZero(TempImag);
//    if (NormFac!=0){
//        cvScaleAdd(LocalMag, cvScalarAll(1/(NormFac)), TempImag, LocalMag);
//
//    }
    
    cvCopy(LocalMag, Parapintar);
    
    // calculating the distance between contours

    if (mode==3){
        cuanto=cvNorm(LocalMag,MagniPattern,CV_C);
            }
    else if (mode==4){
        cuanto=cvNorm(LocalMag,MagniPattern,CV_L1);
    }
    else if (mode==5){
        cuanto=cvNorm(LocalMag,MagniPattern,CV_L2);
    }
    else if (mode==6){
        cuanto=cvNorm(MagniPattern,LocalMag,CV_RELATIVE_C);
    }
    else if (mode==7){
        cuanto=cvNorm(MagniPattern,LocalMag,CV_RELATIVE_L1);
    }
    else if (mode==8){
        cuanto=cvNorm(MagniPattern,LocalMag,CV_RELATIVE_L2);
    }
    cvReleaseMat(&TempReal);
    cvReleaseMat(&TempImag);
    cvReleaseMat(&LocalMag);
    cvReleaseMat(&LocalFFT);  
    
    cvReleaseMat(&CountArray);
    delete [] OtroArray;
  
    

    if (cuanto < minCuanto){
        BestMatches.clear();
        minCuanto = cuanto;
        BestBox = cvFitEllipse2(TheConts);
        for(int q=0;q<TheConts->total;q++){
            CvPoint* p = (CvPoint*)cvGetSeqElem( TheConts, q );
            ofVec2f tempopoint;
            tempopoint.x = (float)p->x;
            tempopoint.y = (float)p->y;
            BestMatches.push_back(tempopoint);
            
        }
    }// end if cuant <minCuanto
    
    return minCuanto;
}








//--------------------------------------------------------------
void testApp::keyPressed(int key){

    switch (key) {
        case 'm':
      
            
            G_DetectMode++;
            if (G_DetectMode>8){G_DetectMode=0;}
            break;
     
        case 's':
            G_smooth = !G_smooth;
            break;
            
        case 'i':
            G_input = (G_input==0)?1:0;
            break;
            
        case OF_KEY_RIGHT:
            G_PointOffset++;
            if(G_PointOffset >3){G_PointOffset =0;}
            break;
        case OF_KEY_LEFT:
            G_PointOffset--;
            if(G_PointOffset <0){G_PointOffset =3;}
            
            break;
        case 'e':
            G_edges =!G_edges;
            break;
            
    case 'g':
            GrayLevels++;
            if(GrayLevels>32){GrayLevels=32;}
            break;
        case 'b':
            GrayLevels--;
            if(GrayLevels<2){GrayLevels=2;}
            break;
           default:  
            break;
    }
    
    
}

//--------------------------------------------------------------
void testApp::keyReleased(int key){

}

//--------------------------------------------------------------
void testApp::mouseMoved(int x, int y){

}

//--------------------------------------------------------------
void testApp::mouseDragged(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mousePressed(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::mouseReleased(int x, int y, int button){

}

//--------------------------------------------------------------
void testApp::windowResized(int w, int h){

}

//--------------------------------------------------------------
void testApp::gotMessage(ofMessage msg){

}

//--------------------------------------------------------------
void testApp::dragEvent(ofDragInfo dragInfo){ 

}