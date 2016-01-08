#include <iostream>
#include <cstdlib>
#include <random>
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
  cv::Mat original_image_lab;
  cv::cvtColor(original_image, original_image_lab, CV_BGR2Lab);

  if (original_image.empty()) {
    std::cerr << "Could not read image \"" << argv[1] << "\"\n";
    return EXIT_FAILURE;
  }

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

    const int kTileWidth = 10;
    const int kTileHeight = 10;
    const cv::Size kGeneratedImageSize = generated_image.size();
    const int kNumTilesX = std::ceil(kGeneratedImageSize.width / (kTileWidth * 1.0f));
    const int kNumTilesY = std::ceil(kGeneratedImageSize.height / (kTileHeight * 1.0f));
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
          const cv::Point kBeginPoint(x * kTileWidth, y * kTileHeight);
          const cv::Point kEndPoint(kBeginPoint.x + kTileWidth, kBeginPoint.y + kTileHeight);
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
      cv::waitKey(1);
    }
  }

  cv::destroyAllWindows();
};
