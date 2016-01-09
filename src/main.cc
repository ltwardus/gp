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
    cv::Mat best_tiles = cv::Mat(kNumTilesY, kNumTilesX, CV_8UC3, cv::Scalar(127, 127, 127));
    double best_distance = -1;

    std::uniform_int_distribution<int> random_tile_row(0, kNumTilesY - 1);
    std::uniform_int_distribution<int> random_tile_col(0, kNumTilesX - 1);

    const unsigned kNumWorkers = std::max(static_cast<unsigned>(1), std::thread::hardware_concurrency());

    std::vector<std::thread> workers(kNumWorkers);
    std::vector<std::mutex> mutexes(kNumWorkers);
    std::vector<std::condition_variable> condition_variables(kNumWorkers);
    std::vector<std::atomic_bool> work_available_flags(kNumWorkers);
    std::vector<std::atomic_bool> stop_working_flags(kNumWorkers);
    std::vector<double> distances(kNumWorkers, -1);
    std::vector<cv::Mat> tiles(kNumWorkers);
    std::vector<cv::Mat> images(kNumWorkers);
    for (unsigned i = 0; i < kNumWorkers; ++i) {
      tiles[i] = best_tiles.clone();
      images[i] = generated_image.clone();
    }

    for (unsigned i = 0; i < kNumWorkers; ++i) {
      workers[i] = std::thread([](
          std::mutex& m,
          std::condition_variable& cv,
          std::atomic_bool& work_available_flag,
          std::atomic_bool& stop_working_flag,
          cv::Mat& tiles,
          cv::Mat& image,
          double& distance,
          const cv::Mat& original_image_lab) {
        std::unique_lock<std::mutex> lock(m);

        std::mt19937 gen(std::random_device{}());
        std::uniform_int_distribution<int> generate_random_tile_row(0, tiles.rows - 1);
        std::uniform_int_distribution<int> generate_random_tile_col(0, tiles.cols - 1);
        std::normal_distribution<> generate_random_color_channel_value(0, 32);
        const int kPixelSize = std::ceil(image.cols / (tiles.cols * 1.0f));

        while(!stop_working_flag) {
          cv.wait(lock, [&work_available_flag, &stop_working_flag]() -> bool {
            return work_available_flag || stop_working_flag;
          });

          if (stop_working_flag) {
            return;
          }

          work_available_flag = false;

          /** Mutate candidate */
          cv::Vec3b& random_tile =
            tiles.at<cv::Vec3b>(generate_random_tile_row(gen), generate_random_tile_col(gen));
          for (int i = 0; i < 3; ++i) {
            random_tile.val[i] =
              clamp(random_tile.val[i] + static_cast<int>(generate_random_color_channel_value(gen)), 0, 255);
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

          cv::Mat image_lab;
          cv::cvtColor(image, image_lab, CV_BGR2Lab);

          distance = cv::norm(original_image_lab, image_lab);
        }
      },
      std::ref(mutexes[i]),
      std::ref(condition_variables[i]),
      std::ref(work_available_flags[i]),
      std::ref(stop_working_flags[i]),
      std::ref(tiles[i]),
      std::ref(images[i]),
      std::ref(distances[i]),
      std::cref(original_image_lab));
    }

    auto last_time_gui_update = std::chrono::system_clock::now();

    while(true) {
      for (unsigned i = 0; i < kNumWorkers; ++i) {
        std::unique_lock<std::mutex> lock(mutexes[i]);
        tiles[i] = best_tiles.clone();
        work_available_flags[i] = true;
        condition_variables[i].notify_all();
      }

      /** Wait for results and also get best candidate */
      double best_candidate_distance = -1;
      unsigned best_candidate_index = 0;
      for (unsigned i = 0; i < kNumWorkers; ++i) {
        std::unique_lock<std::mutex> lock(mutexes[i]);
        if (-1 == best_candidate_distance || distances[i] < best_candidate_distance) {
          best_candidate_distance = distances[i];
          best_candidate_index = i;
        }
      }

      if (-1 == best_distance || best_candidate_distance < best_distance) {
        std::unique_lock<std::mutex> lock(mutexes[best_candidate_index]);
        best_distance = best_candidate_distance;
        generated_image = images[best_candidate_index].clone();
        best_tiles = tiles[best_candidate_index].clone();
        cv::imshow(kGeneratedImageWindowName, generated_image);
      }

      {
        auto now = std::chrono::system_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time_gui_update).count() >= 1000) {
          last_time_gui_update = now;
          /** Wait for ESC key */
          if (27 == static_cast<char>(cv::waitKey(1))) {
            for (unsigned i = 0; i < kNumWorkers; ++i) {
              stop_working_flags[i] = true;
              condition_variables[i].notify_one();
              if (workers[i].joinable()) {
                workers[i].join();
              }
            }

            break;
          }
        }
      }
    }

    cv::imwrite(kOutputImageFilename, generated_image);
  }

  cv::destroyAllWindows();
};
