#pragma once

#include <vector>
#include <map>
#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <queue>
#include <filesystem>
#include <random>
#include <math.h>
#include <algorithm>
#include <utility>
#include <experimental/filesystem>
#include <windows.h>

#include <opencv2/opencv.hpp>

#define OUT

typedef std::pair<std::string, cv::Mat_<short>> t_image;
typedef std::vector<t_image> t_instance_data;
