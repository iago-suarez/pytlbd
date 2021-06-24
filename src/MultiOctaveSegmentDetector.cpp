
#include "multiscale/MultiOctaveSegmentDetector.h"

namespace eth {

MultiOctaveSegmentDetector::MultiOctaveSegmentDetector(eth::SegmentsDetectorPtr detector, int ksize, int numOfOctaves)
    : ksize(ksize), numOfOctaves(numOfOctaves) {
  assert(detector);
  octaveSegDetectors.resize(numOfOctaves);
  for (unsigned int i = 0; i < numOfOctaves; i++) {
    auto copy = detector->clone();
    auto copy_cast = std::dynamic_pointer_cast<eth::OctaveKeyLineDetector>(copy);
    octaveSegDetectors[i] = copy_cast ? copy_cast : std::make_shared<eth::StateOctaveKeyLineDetector>(copy);
  }
  // If the detector doesn't contain a smooth step, smooth the image before processing
  smoothOctaveImg = !octaveSegDetectors.front()->doesSmooth();
}

struct OctaveLine {
  unsigned int octaveCount;//the octave which this line is detected
  unsigned int lineIDInOctave;//the line ID in that octave image
  unsigned int lineIDInScaleLineVec;//the line ID in Scale line vector
  float lineLength; //the length of line in original image scale
};

std::vector<std::vector<cv::line_descriptor::KeyLine>> MultiOctaveSegmentDetector::octaveKeyLines(const cv::Mat &image) {
  if (image.type() != CV_8UC1) {
    std::cerr << "Error: The image should have type CV_8UC1" << std::endl;
    throw std::invalid_argument("Error: The image should have type CV_8UC1");
  }

  // The down sample factor between connective two octave images
  float factor = M_SQRT2;
  std::vector<cv::Mat> pyramid = buildGaussianPyramid(image, factor);

  unsigned int numOfFinalLine = 0;
  for (int o = 0; o < pyramid.size(); o++) {
    const eth::Segments &detectedSegments = octaveSegDetectors[o]->detect(pyramid[o]);
    numOfFinalLine += detectedSegments.size();
  }

  /*lines which correspond to the same line in the octave images will be stored in the same element of ScaleLines.*/
  // Store the lines in OctaveLine structure
  std::vector<OctaveLine> octaveLines(numOfFinalLine);
  // Store the number of finally accepted lines in ScaleLines
  numOfFinalLine = 0;
  unsigned int lineIDInScaleLineVec = 0;
  float dx, dy;
  // add all line detected in the original image
  for (unsigned int lineCurId = 0; lineCurId < octaveSegDetectors[0]->getDetectedSegments().size(); lineCurId++) {
    octaveLines[numOfFinalLine].octaveCount = 0;
    octaveLines[numOfFinalLine].lineIDInOctave = lineCurId;
    octaveLines[numOfFinalLine].lineIDInScaleLineVec = lineIDInScaleLineVec;
    const cv::Vec4f &endpoints = octaveSegDetectors[0]->getDetectedSegments()[lineCurId];
    dx = fabs(endpoints[0] - endpoints[2]);  // x1-x2
    dy = fabs(endpoints[1] - endpoints[3]);  // y1-y2
    octaveLines[numOfFinalLine].lineLength = sqrt(dx * dx + dy * dy);
    numOfFinalLine++;
    lineIDInScaleLineVec++;
  }

  std::vector<float> scale(numOfOctaves);
  scale[0] = 1;
  for (unsigned int octaveCount = 1; octaveCount < numOfOctaves; octaveCount++) {
    scale[octaveCount] = factor * scale[octaveCount - 1];
  }

  float rho1, rho2, tempValue;
  float direction, near, length;
  unsigned int octaveID, lineIDInOctave;
  /*more than one octave image, organize lines in scale space.
   *lines corresponding to the same line in octave images should have the same index in the ScaleLineVec */
  if (numOfOctaves > 1) {
    float twoPI = 2 * M_PI;
    unsigned int closeLineID;
    float endPointDis, minEndPointDis, minLocalDis, maxLocalDis;
    float lp0, lp1, lp2, lp3, np0, np1, np2, np3;
    for (unsigned int octaveCount = 1; octaveCount < numOfOctaves; octaveCount++) {
      const std::vector<cv::Vec4f> &octaveEndpoints = octaveSegDetectors[octaveCount]->getDetectedSegments();

      /*for each line in current octave image, find their corresponding lines in the octaveLines,
       *give them the same value of lineIDInScaleLineVec*/
      for (unsigned int lineCurId = 0; lineCurId < octaveEndpoints.size(); lineCurId++) {
        rho1 = scale[octaveCount] * fabs(octaveSegDetectors[octaveCount]->getLineEquations()[lineCurId][2]);
        /*nearThreshold depends on the distance of the image coordinate origin to current line.
         *so nearThreshold = rho1 * nearThresholdRatio, where nearThresholdRatio = 1-cos(10*pi/180) = 0.0152*/
        tempValue = rho1 * 0.0152;
        float nearThreshold = (tempValue > 6) ? (tempValue) : 6;
        nearThreshold = (nearThreshold < 12) ? nearThreshold : 12;
        dx = fabs(octaveEndpoints[lineCurId][0] - octaveEndpoints[lineCurId][2]);//x1-x2
        dy = fabs(octaveEndpoints[lineCurId][1] - octaveEndpoints[lineCurId][3]);//y1-y2
        length = scale[octaveCount] * sqrt(dx * dx + dy * dy);
        minEndPointDis = 12;
        for (unsigned int lineNextId = 0; lineNextId < numOfFinalLine; lineNextId++) {
          octaveID = octaveLines[lineNextId].octaveCount;
          if (octaveID == octaveCount) {//lines in the same layer of octave image should not be compared.
            break;
          }
          lineIDInOctave = octaveLines[lineNextId].lineIDInOctave;
          /*first check whether current line and next line are parallel.
           *If line1:a1*x+b1*y+c1=0 and line2:a2*x+b2*y+c2=0 are parallel, then
           *-a1/b1=-a2/b2, i.e., a1b2=b1a2.
           *we define parallel=fabs(a1b2-b1a2)
           *note that, in EDLine class, we have normalized the line equations to make a1^2+ b1^2 = a2^2+ b2^2 = 1*/
          direction = fabs(octaveSegDetectors[octaveCount]->getSegmentsDirection()[lineCurId] -
              octaveSegDetectors[octaveID]->getSegmentsDirection()[lineIDInOctave]);
          if (direction > 0.1745 && (twoPI - direction > 0.1745)) {
            continue;//the angle between two lines are larger than 10degrees(i.e. 10*pi/180=0.1745), they are not close to parallel.
          }
          /*now check whether current line and next line are near to each other.
           *If line1:a1*x+b1*y+c1=0 and line2:a2*x+b2*y+c2=0 are near in image, then
           *rho1 = |a1*0+b1*0+c1|/sqrt(a1^2+b1^2) and rho2 = |a2*0+b2*0+c2|/sqrt(a2^2+b2^2) should close.
           *In our case, rho1 = |c1| and rho2 = |c2|, because sqrt(a1^2+b1^2) = sqrt(a2^2+b2^2) = 1;
           *note that, lines are in different octave images, so we define near =  fabs(scale*rho1 - rho2) or
           *where scale is the scale factor between to octave images*/
          rho2 = scale[octaveID] * fabs(octaveSegDetectors[octaveID]->getLineEquations()[lineIDInOctave][2]);
          near = fabs(rho1 - rho2);
          if (near > nearThreshold) {
            continue;//two line are not near in the image
          }
          /*now check the end points distance between two lines, the scale of  distance is in the original image size.
           * find the minimal and maximal end points distance*/
          lp0 = scale[octaveCount] * octaveEndpoints[lineCurId][0];
          lp1 = scale[octaveCount] * octaveEndpoints[lineCurId][1];
          lp2 = scale[octaveCount] * octaveEndpoints[lineCurId][2];
          lp3 = scale[octaveCount] * octaveEndpoints[lineCurId][3];
          const cv::Vec4f &lineIDInOctaveEndPts = octaveSegDetectors[octaveID]->getDetectedSegments()[lineIDInOctave];
          np0 = scale[octaveID] * lineIDInOctaveEndPts[0];
          np1 = scale[octaveID] * lineIDInOctaveEndPts[1];
          np2 = scale[octaveID] * lineIDInOctaveEndPts[2];
          np3 = scale[octaveID] * lineIDInOctaveEndPts[3];
          //L1(0,1)<->L2(0,1)
          dx = lp0 - np0;
          dy = lp1 - np1;
          endPointDis = sqrt(dx * dx + dy * dy);
          minLocalDis = endPointDis;
          maxLocalDis = endPointDis;
          //L1(2,3)<->L2(2,3)
          dx = lp2 - np2;
          dy = lp3 - np3;
          endPointDis = sqrt(dx * dx + dy * dy);
          minLocalDis = (endPointDis < minLocalDis) ? endPointDis : minLocalDis;
          maxLocalDis = (endPointDis > maxLocalDis) ? endPointDis : maxLocalDis;
          //L1(0,1)<->L2(2,3)
          dx = lp0 - np2;
          dy = lp1 - np3;
          endPointDis = sqrt(dx * dx + dy * dy);
          minLocalDis = (endPointDis < minLocalDis) ? endPointDis : minLocalDis;
          maxLocalDis = (endPointDis > maxLocalDis) ? endPointDis : maxLocalDis;
          //L1(2,3)<->L2(0,1)
          dx = lp2 - np0;
          dy = lp3 - np1;
          endPointDis = sqrt(dx * dx + dy * dy);
          minLocalDis = (endPointDis < minLocalDis) ? endPointDis : minLocalDis;
          maxLocalDis = (endPointDis > maxLocalDis) ? endPointDis : maxLocalDis;

          if ((maxLocalDis < 0.8 * (length + octaveLines[lineNextId].lineLength))
              && (minLocalDis < minEndPointDis)) {//keep the closest line
            minEndPointDis = minLocalDis;
            closeLineID = lineNextId;
          }
        }
        //add current line into octaveLines
        if (minEndPointDis < 12) {
          octaveLines[numOfFinalLine].lineIDInScaleLineVec = octaveLines[closeLineID].lineIDInScaleLineVec;
        } else {
          octaveLines[numOfFinalLine].lineIDInScaleLineVec = lineIDInScaleLineVec;
          lineIDInScaleLineVec++;
        }
        octaveLines[numOfFinalLine].octaveCount = octaveCount;
        octaveLines[numOfFinalLine].lineIDInOctave = lineCurId;
        octaveLines[numOfFinalLine].lineLength = length;
        numOfFinalLine++;
      }
    }  // end for(unsigned int octave = 1; octave<numOfOctave_; octave++)
  }  // end if(numOfOctave_>1)

  ////////////////////////////////////
  // Reorganize the detected lines into keyLines

  std::vector<std::vector<cv::line_descriptor::KeyLine>> keyLines(lineIDInScaleLineVec);
  unsigned int tempID;
  float s1, e1, s2, e2;
  bool shouldChange;
  cv::line_descriptor::KeyLine singleLine;
  for (unsigned int lineID = 0; lineID < numOfFinalLine; lineID++) {
    lineIDInOctave = octaveLines[lineID].lineIDInOctave;
    octaveID = octaveLines[lineID].octaveCount;
    direction = octaveSegDetectors[octaveID]->getSegmentsDirection()[lineIDInOctave];
    singleLine.octave = octaveID;
    singleLine.angle = direction;
    singleLine.lineLength = octaveLines[lineID].lineLength;
    singleLine.response = octaveSegDetectors[octaveID]->getSegmentsSalience()[lineIDInOctave];
    singleLine.numOfPixels = octaveSegDetectors[octaveID]->getNumberOfPixels(lineIDInOctave);
    // Decide the start point and end point
    const cv::Vec4f &lineIDInOctaveEndPts = octaveSegDetectors[octaveID]->getDetectedSegments()[lineIDInOctave];
    s1 = lineIDInOctaveEndPts[0];//sx
    s2 = lineIDInOctaveEndPts[1];//sy
    e1 = lineIDInOctaveEndPts[2];//ex
    e2 = lineIDInOctaveEndPts[3];//ey
    dx = e1 - s1;//ex-sx
    dy = e2 - s2;//ey-sy
    // Position the points in such a way that its atan2 produce the correct angle aligned with the image gradient
    shouldChange = (dx * std::cos(direction) + dy * std::sin(direction)) < 0;

    tempValue = scale[octaveID];
    if (shouldChange) {
      singleLine.sPointInOctaveX = e1;
      singleLine.sPointInOctaveY = e2;
      singleLine.ePointInOctaveX = s1;
      singleLine.ePointInOctaveY = s2;
      singleLine.startPointX = tempValue * e1;
      singleLine.startPointY = tempValue * e2;
      singleLine.endPointX = tempValue * s1;
      singleLine.endPointY = tempValue * s2;
    } else {
      singleLine.sPointInOctaveX = s1;
      singleLine.sPointInOctaveY = s2;
      singleLine.ePointInOctaveX = e1;
      singleLine.ePointInOctaveY = e2;
      singleLine.startPointX = tempValue * s1;
      singleLine.startPointY = tempValue * s2;
      singleLine.endPointX = tempValue * e1;
      singleLine.endPointY = tempValue * e2;
    }

    dx = singleLine.endPointX - singleLine.startPointX;
    dy = singleLine.endPointY - singleLine.startPointY;
//    float dot = dx * std::cos(singleLine.angle) + dy * std::sin(singleLine.angle);
//    assert(dot > 0);
    singleLine.angle = std::atan2(dy, dx);

    tempID = octaveLines[lineID].lineIDInScaleLineVec;
    keyLines[tempID].push_back(singleLine);
  }

  return keyLines;
}

std::vector<cv::Mat> MultiOctaveSegmentDetector::buildGaussianPyramid(const cv::Mat &initialImage, float factor) {
  cv::Mat octaveImage = initialImage.clone();
  float preSigma2 = 0;//orignal image is not blurred, has zero sigma;
  float curSigma2 = 1.0;//[sqrt(2)]^0=1;
  std::vector<cv::Mat> pyramid;
  for (unsigned int octaveCount = 0; octaveCount < numOfOctaves; octaveCount++) {
    /* Form each level by adding incremental blur from previous level.
     * curSigma = [sqrt(2)]^octave;
     * increaseSigma^2 = curSigma^2 - preSigma^2 */
    float increaseSigma = sqrt(curSigma2 - preSigma2);

    cv::Mat blurred;
    cv::GaussianBlur(octaveImage,
                     blurred,
                     cv::Size(ksize, ksize),
                     increaseSigma,
                     increaseSigma,
                     cv::BORDER_REPLICATE);

    // If the detector contains a shothing step, set the raw image to it, otherwise sent the smooth image
    cv::Mat imageToProcess = smoothOctaveImg ? blurred : octaveImage;
    pyramid.push_back(imageToProcess);

    //down sample the current octave image to get the next octave image
    cv::Size newSize(int(octaveImage.cols / factor), int(octaveImage.rows / factor));
    octaveImage.release();
    cv::resize(blurred, octaveImage, newSize, 0, 0, cv::INTER_NEAREST);
    preSigma2 = curSigma2;
    curSigma2 = curSigma2 * 2;
  }

  return pyramid;
}
}