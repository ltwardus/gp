#include <iostream>
#include <cstdlib>
#include <random>
#include <limits>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

void print_help() {
  std::cerr << "Usage: gp [pixel scale] [input image] [output image]\n";
  std::cerr << "pixel scale - value in range (0, 1>, it is used to calculate the output image pixel size:\n";
  std::cerr << "              pixel size = ceil(image_shorter_dimension * pixel_scale))\n";
}

int main(int argc, char* argv[]) {
  if (4 != argc) {
    print_help();
    return EXIT_FAILURE;
  }

  const float kPixelScale = std::max(std::numeric_limits<float>::min(), std::min(1.0f, std::stof(argv[1])));
  const std::string kInputImageFilename = argv[2];
  const std::string kOutputImageFilename = argv[3];

  cv::Mat original_image = cv::imread(kInputImageFilename, CV_LOAD_IMAGE_COLOR);
  cv::Mat original_image_lab;
  cv::cvtColor(original_image, original_image_lab, CV_BGR2Lab);

  if (original_image.empty()) {
    std::cerr << "Could not read image \"" << kInputImageFilename << "\"\n";
    return EXIT_FAILURE;
  }

  std::cout << "Press ESC to save output image and exit ..." << std::endl;

  {
    const std::string kOriginalImageWindowName = "Original image";
    cv::namedWindow(kOriginalImageWindowName, CV_GUI_EXPANDED);
    cv::imshow(kOriginalImageWindowName, original_image);
  }

  {
    cv::Mat generated_image = original_image.clone();
    generated_image = cv::Scalar();

    const std::string kGeneratedImageWindowName = "Generated image";
    cv::namedWindow(kGeneratedImageWindowName, CV_GUI_EXPANDED);
    cv::imshow(kGeneratedImageWindowName, generated_image);

    const cv::Size kGeneratedImageSize = generated_image.size();
    const int kPixelSize =
      std::ceil(std::min(kGeneratedImageSize.width, kGeneratedImageSize.height) * kPixelScale);
    const int kNumTilesX = std::ceil(kGeneratedImageSize.width / (kPixelSize * 1.0f));
    const int kNumTilesY = std::ceil(kGeneratedImageSize.height / (kPixelSize * 1.0f));
    cv::Mat tile_colors = cv::Mat::zeros(kNumTilesY, kNumTilesX, CV_8UC3);
    double best_distance = -1;
    cv::Mat candidate_tile_colors = tile_colors.clone();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> random_tile_row(0, kNumTilesY - 1);
    std::uniform_int_distribution<int> random_tile_col(0, kNumTilesX - 1);

    while(true) {
      cv::Mat candidate_image = generated_image.clone();
      /** Mutate candidate */
      cv::Vec3b& candidate_tile_color = candidate_tile_colors.at<cv::Vec3b>(random_tile_row(gen), random_tile_col(gen));
      std::normal_distribution<> random_b(candidate_tile_color.val[0], 128);
      std::normal_distribution<> random_g(candidate_tile_color.val[1], 128);
      std::normal_distribution<> random_r(candidate_tile_color.val[2], 128);
      candidate_tile_color.val[0] = std::max(0, std::min(255, static_cast<int>(random_b(gen))));
      candidate_tile_color.val[1] = std::max(0, std::min(255, static_cast<int>(random_g(gen))));
      candidate_tile_color.val[2] = std::max(0, std::min(255, static_cast<int>(random_r(gen))));

      for (int x = 0; x < kNumTilesX; ++x) {
        for (int y = 0; y < kNumTilesY; ++y) {
          const cv::Point kBeginPoint(x * kPixelSize, y * kPixelSize);
          const cv::Point kEndPoint(kBeginPoint.x + kPixelSize, kBeginPoint.y + kPixelSize);
          cv::rectangle(
            candidate_image,
            kBeginPoint,
            kEndPoint,
            cv::Scalar(candidate_tile_colors.at<cv::Vec3b>(y, x)),
            CV_FILLED);
        }
      }

      cv::Mat candidate_image_lab;
      cv::cvtColor(candidate_image, candidate_image_lab, CV_BGR2Lab);

      double distance = cv::norm(original_image_lab, candidate_image_lab);

      if (-1 == best_distance || distance < best_distance) {
        best_distance = distance;
        generated_image = candidate_image.clone();
        tile_colors = candidate_tile_colors.clone();
        cv::imshow(kGeneratedImageWindowName, generated_image);
      } else {
        candidate_tile_colors = tile_colors.clone();
      }
      /** Wait for ESC key */
      if (27 == static_cast<char>(cv::waitKey(1))) {
        break;
      }
    }

    cv::imwrite(kOutputImageFilename, generated_image);
  }


  cv::destroyAllWindows();
};
