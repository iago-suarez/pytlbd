#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/line_descriptor.hpp>
#include <random>
#include "LineBandDescriptor.h"
#include <PairwiseLineMatching.h>
#include <EDLineDetector.h>

void draw_matches(const cv::Mat &cvLeftImage, const cv::Mat &cvRightImage,
                  const eth::ScaleLines &linesInLeft, const eth::ScaleLines &linesInRight,
                  const std::vector<std::pair<uint32_t, uint32_t>> &matchResult) {
  cv::Point startPoint;
  cv::Point endPoint;

  cv::Mat cvLeftColorImage, cvRightColorImage;
  cv::cvtColor(cvLeftImage, cvLeftColorImage, cv::COLOR_GRAY2BGR);
  cv::cvtColor(cvRightImage, cvRightColorImage, cv::COLOR_GRAY2BGR);

  int w = cvLeftImage.cols, h = cvLeftImage.rows;
  int lowest = 100, highest = 255;
  int range = (highest - lowest) + 1;
  unsigned int r, g, b; //the color of lines
  for (auto &lines_vec : linesInLeft) {
    r = lowest + int(rand() % range);
    g = lowest + int(rand() % range);
    b = lowest + int(rand() % range);
    startPoint = cv::Point(int(lines_vec[0].startPointX), int(lines_vec[0].startPointY));
    endPoint = cv::Point(int(lines_vec[0].endPointX), int(lines_vec[0].endPointY));
    cv::line(cvLeftColorImage, startPoint, endPoint, CV_RGB(r, g, b));
  }
  cv::imshow("Left", cvLeftColorImage);

  for (auto &lines_vec : linesInRight) {
    r = lowest + int(rand() % range);
    g = lowest + int(rand() % range);
    b = lowest + int(rand() % range);
    startPoint = cv::Point(int(lines_vec[0].startPointX), int(lines_vec[0].startPointY));
    endPoint = cv::Point(int(lines_vec[0].endPointX), int(lines_vec[0].endPointY));
    cv::line(cvRightColorImage, startPoint, endPoint, CV_RGB(r, g, b));
  }
  cv::imshow("Right", cvRightColorImage);

  ///////////####################################################################

  //store the matching results of the first and second images into a single image
  int lineIDLeft, lineIDRight;
  cv::cvtColor(cvLeftImage, cvLeftColorImage, cv::COLOR_GRAY2RGB);
  cv::cvtColor(cvRightImage, cvRightColorImage, cv::COLOR_GRAY2RGB);
  int lowest1 = 0, highest1 = 255;
  int range1 = (highest1 - lowest1) + 1;
  std::vector<unsigned int> r1(matchResult.size() / 2), g1(matchResult.size() / 2),
      b1(matchResult.size() / 2); //the color of lines
  for (unsigned int pair = 0; pair < matchResult.size() / 2; pair++) {
    r1[pair] = lowest1 + int(rand() % range1);
    g1[pair] = lowest1 + int(rand() % range1);
    b1[pair] = 255 - r1[pair];
    lineIDLeft = matchResult[pair].first;
    lineIDRight = matchResult[pair].second;
    startPoint.x = linesInLeft[lineIDLeft][0].startPointX;
    startPoint.y = linesInLeft[lineIDLeft][0].startPointY;
    endPoint.x = linesInLeft[lineIDLeft][0].endPointX;
    endPoint.y = linesInLeft[lineIDLeft][0].endPointY;
    cv::line(cvLeftColorImage,
             startPoint,
             endPoint,
             CV_RGB(r1[pair], g1[pair], b1[pair]),
             4,
             cv::LINE_AA);
    startPoint.x = linesInRight[lineIDRight][0].startPointX;
    startPoint.y = linesInRight[lineIDRight][0].startPointY;
    endPoint.x = linesInRight[lineIDRight][0].endPointX;
    endPoint.y = linesInRight[lineIDRight][0].endPointY;

    cv::line(cvRightColorImage,
             startPoint,
             endPoint,
             CV_RGB(r1[pair], g1[pair], b1[pair]),
             4,
             cv::LINE_AA);
  }

  cv::Mat cvResultColorImage1(h, w * 2, CV_8UC3);
  cv::Mat cvResultColorImage2, cvResultColorImage;

  cv::Mat out1 = cvResultColorImage1(cv::Rect(0, 0, w, h));
  cvLeftColorImage.copyTo(out1);
  cv::Mat out2 = cvResultColorImage1(cv::Rect(w, 0, w, h));
  cvRightColorImage.copyTo(out2);

  cvResultColorImage2 = cvResultColorImage1.clone();
  for (unsigned int pair = 0; pair < matchResult.size() / 2; pair++) {
    lineIDLeft = matchResult[pair].first;
    lineIDRight = matchResult[pair].second;
    startPoint.x = linesInLeft[lineIDLeft][0].startPointX;
    startPoint.y = linesInLeft[lineIDLeft][0].startPointY;
    endPoint.x = linesInRight[lineIDRight][0].startPointX + w;
    endPoint.y = linesInRight[lineIDRight][0].startPointY;
    cv::line(cvResultColorImage2,
             startPoint,
             endPoint,
             CV_RGB(r1[pair], g1[pair], b1[pair]),
             2,
             cv::LINE_AA);
  }
  cv::addWeighted(cvResultColorImage1, 0.5, cvResultColorImage2, 0.5, 0.0, cvResultColorImage);

  std::cout << "number of total matches = " << matchResult.size() / 2 << std::endl;
  cv::imshow("LBDSG", cvResultColorImage);
  cv::waitKey();
}

void line_matching_example_leuven() {

  //load first image from file
  cv::Mat cvLeftImage = cv::imread("../resources/leuven1.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat cvRightImage = cv::imread("../resources/leuven2.jpg", cv::IMREAD_GRAYSCALE);

  // 2.1. Detecting lines in the scale space
  eth::MultiOctaveSegmentDetector detector(std::make_shared<eth::EDLineDetector>());
  eth::ScaleLines linesInLeft = detector.octaveKeyLines(cvLeftImage);
  eth::ScaleLines linesInRight = detector.octaveKeyLines(cvRightImage);

  // 2.2. The band representation of the line support region & 2.3. The construction of the Line Band Descriptor
  eth::LineBandDescriptor lineDesc;
  std::vector<std::vector<cv::Mat>> descriptorsLeft, descriptorsRight;
  lineDesc.compute(cvLeftImage, linesInLeft, descriptorsLeft);
  lineDesc.compute(cvRightImage, linesInRight, descriptorsRight);

  // 3. Graph matching using spectral technique
  std::vector<std::pair<uint32_t, uint32_t>> matchResult;
  eth::PairwiseLineMatching lineMatch;
  lineMatch.matchLines(linesInLeft, linesInRight, descriptorsLeft, descriptorsRight, matchResult);

  // Show the result
  draw_matches(cvLeftImage, cvRightImage, linesInLeft, linesInRight, matchResult);

  std::vector<std::pair<uint32_t, uint32_t>> expectedMatch = {
      {45, 60}, {44, 49}, {47, 62}, {162, 182}, {80, 91}, {0, 1}, {3, 4}, {79, 90}, {180, 54}, {173, 193}, {125, 134},
      {2, 3}, {9, 18}, {352, 431}, {128, 145}, {24, 0}, {8, 17}, {272, 324}, {130, 146}, {146, 166}, {48, 65},
      {245, 289}, {83, 92}, {390, 295}, {249, 296}, {15, 16}, {13, 5}, {416, 505}, {331, 162}, {141, 136}, {84, 93},
      {254, 301}, {56, 44}, {12, 375}, {46, 61}, {43, 48}, {164, 184}, {179, 53}, {385, 473}, {267, 307}, {33, 34},
      {163, 183}, {221, 283}, {223, 285}, {354, 433}, {107, 20}, {193, 211}, {136, 153}, {184, 200}, {200, 222},
      {215, 252}, {204, 228}, {177, 58}, {319, 33}, {110, 116}, {116, 124}, {145, 165}, {99, 115}, {360, 440},
      {407, 497}, {284, 335}, {121, 130}, {214, 240}, {373, 448}, {94, 103}, {150, 170}, {95, 107}, {346, 247},
      {246, 290}, {336, 396}, {240, 280}, {87, 99}, {126, 141}, {191, 405}, {174, 194}, {106, 119}, {159, 179},
      {348, 423}, {93, 102}, {342, 410}, {138, 157}, {289, 338}, {168, 394}, {36, 374}, {96, 110}, {182, 197},
      {339, 406}, {255, 298}, {98, 111}, {257, 341}, {89, 104}, {119, 129}, {65, 78}, {269, 319}, {288, 337},
      {227, 263}, {356, 438}, {90, 105}, {194, 212}, {205, 229}, {270, 320}, {132, 138}, {127, 142}, {185, 401},
      {39, 45}, {285, 342}, {134, 144}, {404, 495}, {161, 181}, {186, 403}, {190, 204}, {369, 447}, {306, 353},
      {100, 382}, {395, 484}, {129, 140}, {103, 120}, {394, 480}, {417, 506}, {50, 42}, {72, 82}, {35, 43}, {213, 239},
      {69, 95}, {303, 351}, {368, 445}, {361, 441}, {337, 400}, {171, 52}, {247, 291}, {256, 303}, {228, 264},
      {187, 207}, {335, 186}, {393, 481}, {178, 195}, {42, 47}, {92, 106}, {62, 75}, {156, 176}, {137, 155}, {330, 383},
      {382, 391}, {353, 432}, {318, 11}, {243, 279}, {135, 143}, {198, 213}, {386, 299}, {147, 167}, {160, 180},
      {268, 318}, {242, 274}, {68, 94}, {291, 352}, {264, 312}, {123, 135}, {341, 407}, {53, 66}, {1, 2}, {387, 476},
      {409, 485}, {114, 123}, {109, 118}, {124, 131}, {222, 284}, {388, 302}, {73, 83}, {183, 198}, {355, 474},
      {49, 64}, {151, 178}, {154, 171}, {71, 69}, {327, 378}, {149, 169}, {188, 208}, {181, 196}, {211, 244},
      {277, 334}, {196, 217}, {167, 132}, {349, 425}, {278, 331}, {82, 97}, {201, 203}, {237, 268}, {260, 306},
      {199, 219}, {340, 464}, {365, 443}, {271, 317}, {77, 86}, {231, 267}, {104, 122}, {263, 311}, {265, 313},
      {64, 77}, {244, 278}, {61, 377}, {326, 121}, {383, 465}, {158, 177}, {253, 300}, {157, 175}, {279, 333},
      {143, 172}, {113, 126}, {322, 369}, {74, 81}, {351, 430}, {389, 483}, {307, 354}, {232, 243}, {313, 359},
      {281, 325}, {381, 459}, {343, 411}, {75, 80}, {26, 24}, {38, 46}, {296, 349}, {378, 381}, {308, 330}, {414, 51},
      {274, 314}, {212, 494}, {91, 100}, {224, 286}, {266, 358}, {397, 488}, {176, 57}, {295, 350}, {405, 496},
      {41, 39}, {148, 168}, {166, 189}, {367, 329}, {85, 89}, {292, 343}, {81, 379}, {333, 461}, {239, 272}, {170, 133},
      {117, 125}, {59, 72}, {197, 220}, {248, 434}, {11, 19}, {10, 21}, {258, 309}, {51, 41}, {86, 101}, {216, 251},
      {208, 235}, {131, 147}, {101, 117}, {251, 437}, {241, 416}, {18, 27}, {261, 439}, {236, 271}, {338, 205},
      {359, 310}, {320, 26}, {286, 346}, {7, 29}, {40, 35}, {207, 233}, {217, 419}, {54, 30}, {31, 36}, {406, 498},
      {398, 487}, {175, 50},
  };

//  std::cout << "matchResult: " << matchResult << std::endl;

  assert(275 == matchResult.size());
  for (int i = 0; i < matchResult.size(); i++) {
//    std::cout << "{" << matchResult[i].first << ", " << matchResult[i].second << "}," << std::endl;
    assert(expectedMatch[i].first == matchResult[i].first);
    assert(expectedMatch[i].second == matchResult[i].second);
  }
}

void line_matching_example_boat() {

  //load first image from file
  cv::Mat cvLeftImage = cv::imread("../resources/boat1.jpg", cv::IMREAD_GRAYSCALE);
  cv::Mat cvRightImage = cv::imread("../resources/boat3.jpg", cv::IMREAD_GRAYSCALE);


  // 2.1. Detecting lines in the scale space
  eth::MultiOctaveSegmentDetector detector(std::make_shared<eth::EDLineDetector>());
  eth::ScaleLines linesInLeft = detector.octaveKeyLines(cvLeftImage);
  eth::ScaleLines linesInRight = detector.octaveKeyLines(cvRightImage);

  // Description: 2.2. The band representation of the line support region & 2.3. The construction of the Line Band Descriptor
  eth::LineBandDescriptor lineDesc;
  std::vector<std::vector<cv::Mat>> descriptorsLeft, descriptorsRight;
  lineDesc.compute(cvLeftImage, linesInLeft, descriptorsLeft);
  lineDesc.compute(cvRightImage, linesInRight, descriptorsRight);

  // 3. Graph matching using spectral technique
  std::vector<std::pair<uint32_t, uint32_t>> matchResult;
  eth::PairwiseLineMatching lineMatch;
  lineMatch.matchLines(linesInLeft, linesInRight, descriptorsLeft, descriptorsRight, matchResult);

  // Show the result
  draw_matches(cvLeftImage, cvRightImage, linesInLeft, linesInRight, matchResult);

  std::vector<std::pair<uint32_t, uint32_t>> expectedMatch = {
      {369, 118}, {236, 147}, {227, 144}, {335, 101}, {699, 395}, {349, 143}, {163, 63}, {58, 114}, {641, 310},
      {123, 132}, {137, 43}, {425, 232}, {293, 182}, {184, 91}, {533, 239}, {924, 467}, {764, 220}, {4, 80}, {326, 141},
      {112, 98}, {551, 267}, {452, 289}, {758, 178}, {707, 107}, {757, 151}, {0, 57}, {692, 95}, {75, 94}, {881, 276},
      {258, 212}, {905, 457}, {531, 277}, {285, 162}, {64, 99}, {853, 485}, {466, 245}, {827, 240}, {277, 214},
      {51, 60}, {419, 198}, {376, 236}, {212, 157}, {747, 129}, {626, 241}, {712, 222}, {441, 286}, {508, 268},
      {790, 264}, {892, 433}, {761, 235}, {731, 123}, {325, 216}, {746, 218}, {732, 170}, {79, 93}, {774, 199},
      {196, 155}, {575, 260}, {931, 448}, {177, 175}, {218, 111}, {856, 70}, {152, 104}, {59, 149}, {653, 473},
      {812, 327}, {744, 213}, {548, 292}, {418, 197}, {880, 470}, {215, 125}, {294, 181}, {781, 279}, {756, 161},
      {379, 414}, {360, 164}, {316, 183}, {484, 265}, {290, 184}, {127, 177}, {815, 281}, {156, 39}, {942, 455},
      {673, 303}, {738, 412}, {69, 116}, {197, 201}, {725, 131}, {848, 134}, {34, 21}, {595, 417}, {305, 180},
      {57, 113}, {553, 284}, {298, 255}, {191, 392}, {805, 259}, {927, 422}, {367, 409}, {921, 269}, {403, 211},
      {913, 464}, {879, 280}, {745, 185}, {713, 223}, {407, 221}, {554, 261}, {814, 318}, {22, 74}, {465, 204},
      {907, 483}, {329, 126}, {174, 174}, {802, 306}, {9, 58}, {320, 193}, {331, 225}, {564, 253}, {940, 487},
      {283, 192}, {926, 466}, {337, 224}, {308, 228}, {280, 458}, {60, 121}, {556, 305}, {327, 160}, {883, 461},
      {109, 122}, {874, 431}, {547, 304}, {44, 62}, {780, 282}, {810, 301}, {446, 249}, {141, 44}, {241, 130},
      {460, 416}, {302, 153}, {275, 187}, {317, 169}, {68, 83}, {87, 135}, {153, 92}, {734, 454}, {318, 195}, {89, 137},
      {825, 312}, {496, 246}, {950, 490}, {71, 96}, {769, 258}, {378, 205}, {914, 179}, {29, 82}, {786, 202},
      {442, 426}, {656, 298}, {205, 450}, {400, 152}, {512, 244}, {684, 49}, {767, 315}, {552, 285}, {529, 293},
      {11, 12}, {28, 46}, {35, 391}, {231, 154}, {27, 85}, {580, 326}, {313, 115}, {252, 120}, {111, 150}, {948, 489},
      {813, 317}, {385, 252}, {776, 425}, {248, 190}, {348, 102}, {768, 270}, {451, 295}, {651, 233}, {138, 108},
      {493, 203}, {779, 291}, {364, 89}, {520, 247}, {603, 320}, {709, 140}, {833, 272}, {2, 71}, {192, 90}, {701, 397},
      {749, 163}, {370, 119}, {63, 100}, {799, 307}, {198, 69}, {787, 158}, {288, 243}, {264, 139}, {203, 146},
      {819, 328}, {806, 283}, {822, 313}, {729, 165}, {350, 226}, {121, 208}, {824, 430}, {821, 334}, {689, 447},
      {649, 273}, {304, 128}, {735, 127}, {573, 321}, {759, 117}, {300, 256}, {172, 40}, {644, 230}, {538, 302},
      {504, 262}, {676, 325}, {698, 97}, {159, 79}, {532, 278}, {254, 234}, {447, 254}, {655, 238}, {632, 299},
      {642, 421}, {594, 266}, {15, 7}, {737, 173}, {481, 300}, {841, 482},
  };

  assert(236 == matchResult.size());
  for (int i = 0; i < matchResult.size(); i++) {
//    std::cout << "{" << matchResult[i].first << ", " << matchResult[i].second << "}," << std::endl;
    assert(expectedMatch[i].first == matchResult[i].first);
    assert(expectedMatch[i].second == matchResult[i].second);
  }
}

int main() {
  std::cout << "**********************************************" << std::endl;
  std::cout << "****************** TLBD Test *****************" << std::endl;
  std::cout << "**********************************************" << std::endl;

  line_matching_example_leuven();
  line_matching_example_boat();
}