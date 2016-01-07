#include <iostream>
#include <cstdlib>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

void print_help() {
  std::cerr << "Usage: gp [image file name]\n";
}

int main(int argc, char* argv[]) {
  if (2 != argc) {
    print_help();
    return EXIT_FAILURE;
  }

  cv::Mat original_image = cv::imread(argv[1], CV_LOAD_IMAGE_COLOR);

  if (original_image.empty()) {
    std::cerr << "Could not read image \"" << argv[1] << "\"\n";
    return EXIT_FAILURE;
  }

  {
    const std::string kOriginalImageWindowName = "Original image";
    cv::namedWindow(kOriginalImageWindowName, CV_GUI_EXPANDED);
    cv::imshow(kOriginalImageWindowName, original_image);
  }

  cv::Mat generated_image = original_image;
  {
    const std::string kGeneratedImageWindowName = "Generated image";
    cv::namedWindow(kGeneratedImageWindowName, CV_GUI_EXPANDED);
    cv::imshow(kGeneratedImageWindowName, generated_image);
  }

  while(true) {
    cv::waitKey(10);
  };

  cv::destroyAllWindows();
};
