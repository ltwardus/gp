#include <iostream>
#include <cstdlib>
#include <random>
#include <limits>
#include <thread>
#include <future>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

void print_help() {
  std::cerr << "Usage: gp [pixel scale] [input image] [output image]\n";
  std::cerr << "pixel scale - value in range (0, 1>, it is used to calculate the output image pixel size:\n";
  std::cerr << "              pixel size = ceil(image_shorter_dimension * pixel_scale))\n";
}

template<class T>
T clamp(const T&n, const T& lower, const T& upper) {
  return std::max(lower, std::min(n, upper));
}

void generate_end_evaluate_solution(
    cv::Mat& tiles,
    cv::Mat& image,
    cv::Mat& image_lab,
    double& distance,
    std::mt19937& rnd_engine,
    std::uniform_int_distribution<int>& generate_random_tile_row,
    std::uniform_int_distribution<int>& generate_random_tile_col,
    std::normal_distribution<> generate_random_color_channel_value,
    const cv::Mat& original_image_lab) {
  const int kPixelSize = std::ceil(image.cols / (tiles.cols * 1.0f));

  /** Mutate candidate */
  cv::Vec3b& random_tile =
    tiles.at<cv::Vec3b>(generate_random_tile_row(rnd_engine), generate_random_tile_col(rnd_engine));
  for (int i = 0; i < 3; ++i) {
    random_tile.val[i] =
      clamp(random_tile.val[i] + static_cast<int>(generate_random_color_channel_value(rnd_engine)), 0, 255);
  }

  for (int x = 0; x < tiles.cols; ++x) {
    for (int y = 0; y < tiles.rows; ++y) {
      const cv::Point kBeginPoint(x * kPixelSize, y * kPixelSize);
      const cv::Point kEndPoint(kBeginPoint.x + kPixelSize, kBeginPoint.y + kPixelSize);
      cv::rectangle(
        image,
        kBeginPoint,
        kEndPoint,
        cv::Scalar(tiles.at<cv::Vec3b>(y, x)),
        CV_FILLED);
    }
  }

  cv::cvtColor(image, image_lab, CV_BGR2Lab);

  distance = cv::norm(original_image_lab, image_lab);
}

int main(int argc, char* argv[]) {
  if (4 != argc) {
    print_help();
    return EXIT_FAILURE;
  }

  const float kPixelScale = std::max(std::numeric_limits<float>::min(), std::min(1.0f, std::stof(argv[1])));
  const std::string kInputImageFilename = argv[2];
  const std::string kOutputImageFilename = argv[3];

  const cv::Mat original_image = cv::imread(kInputImageFilename, CV_LOAD_IMAGE_COLOR);
  const cv::Mat original_image_lab = [&original_image]() {
    cv::Mat temp;
    cv::cvtColor(original_image, temp, CV_BGR2Lab);
    return temp;
  }();

  if (original_image.empty()) {
    std::cerr << "Could not read image \"" << kInputImageFilename << "\"\n";
    return EXIT_FAILURE;
  }

  std::cout << "Press ESC to save output image and exit ..." << std::endl;

  const std::string kOriginalImageWindowName = "Original image - Press ESC to save output image and exit ...";
  cv::namedWindow(kOriginalImageWindowName, CV_GUI_EXPANDED);
  cv::imshow(kOriginalImageWindowName, original_image);

  {
    cv::Mat output_image = original_image.clone();
    output_image = cv::Scalar();

    const std::string kOutputImageWindowName = "Output image - Press ESC to save output image and exit ...";
    cv::namedWindow(kOutputImageWindowName, CV_GUI_EXPANDED);
    cv::imshow(kOutputImageWindowName, output_image);

    const cv::Size kOutputImageSize = output_image.size();
    const int kPixelSize =
      std::ceil(std::min(kOutputImageSize.width, kOutputImageSize.height) * kPixelScale);
    const int kNumTilesX = std::ceil(kOutputImageSize.width / (kPixelSize * 1.0f));
    const int kNumTilesY = std::ceil(kOutputImageSize.height / (kPixelSize * 1.0f));
    cv::Mat best_tiles = cv::Mat(kNumTilesY, kNumTilesX, CV_8UC3, cv::Scalar(127, 127, 127));
    double best_distance = -1;

    std::uniform_int_distribution<int> random_tile_row(0, kNumTilesY - 1);
    std::uniform_int_distribution<int> random_tile_col(0, kNumTilesX - 1);

    const unsigned kNumWorkers = std::max(static_cast<unsigned>(1), std::thread::hardware_concurrency());

    std::vector<std::future<void>> results(kNumWorkers);
    std::vector<cv::Mat> tiles(kNumWorkers);
    std::vector<cv::Mat> images(kNumWorkers);
    std::vector<cv::Mat> images_lab(kNumWorkers);
    std::vector<double> distances(kNumWorkers, -1);
    std::vector<std::mt19937> rnd_engines(kNumWorkers);
    std::vector<std::uniform_int_distribution<int>>
      generate_random_tile_rows(kNumWorkers, std::uniform_int_distribution<int>(0, kNumTilesY - 1));
    std::vector<std::uniform_int_distribution<int>>
      generate_random_tile_cols(kNumWorkers, std::uniform_int_distribution<int>(0, kNumTilesX - 1));
    std::vector<std::normal_distribution<>>
      generate_random_color_channel_values(kNumWorkers, std::normal_distribution<>(0, 32));
    for (unsigned i = 0; i < kNumWorkers; ++i) {
      tiles[i] = best_tiles.clone();
      images[i] = output_image.clone();
      images_lab[i] = output_image.clone();
      rnd_engines[i] = std::mt19937(std::random_device{}());
    }

    std::chrono::system_clock::time_point last_time_gui_update;
    const int kGuiUpdateIntervalMs = 200;

    while(true) {
      for (unsigned i = 0; i < kNumWorkers; ++i) {
        tiles[i] = best_tiles.clone();

        results[i] =
          std::async(
            std::launch::async,
            generate_end_evaluate_solution,
            std::ref(tiles[i]),
            std::ref(images[i]),
            std::ref(images_lab[i]),
            std::ref(distances[i]),
            std::ref(rnd_engines[i]),
            std::ref(generate_random_tile_rows[i]),
            std::ref(generate_random_tile_cols[i]),
            std::ref(generate_random_color_channel_values[i]),
            std::cref(original_image_lab));
      }

      /** Wait for results and also get best candidate */
      double best_candidate_distance = -1;
      unsigned best_candidate_index = 0;
      for (unsigned i = 0; i < kNumWorkers; ++i) {
        results[i].get();
        if (-1 == best_candidate_distance || distances[i] < best_candidate_distance) {
          best_candidate_distance = distances[i];
          best_candidate_index = i;
        }
      }

      if (-1 == best_distance || best_candidate_distance < best_distance) {
        best_distance = best_candidate_distance;
        output_image = images[best_candidate_index].clone();
        best_tiles = tiles[best_candidate_index].clone();
      }

      {
        auto now = std::chrono::system_clock::now();
        const auto& gui_update_time_diff_ms =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_gui_update).count();
        if (gui_update_time_diff_ms >= kGuiUpdateIntervalMs) {
          cv::imshow(kOriginalImageWindowName, original_image);
          cv::imshow(kOutputImageWindowName, output_image);
          last_time_gui_update = now;
          /** Wait for ESC key */
          if (27 == static_cast<char>(cv::waitKey(1))) {
            break;
          }
        }
      }
    }

    cv::imwrite(kOutputImageFilename, output_image);
  }

  cv::destroyAllWindows();
};
