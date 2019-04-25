#include "read_instances.h"
namespace fs = std::filesystem;

inline bool ends_with(std::string const& value, std::string const& ending)
{
	if (ending.size() > value.size()) return false;
	return std::equal(ending.rbegin(), ending.rend(), value.rbegin());//rbegin and rend iterates backwards
}

t_instance_data readDirData(std::string& dir, std::string& fileExtension)
{
	t_instance_data instanceData;
	t_image item;

	for (const auto & d : fs::recursive_directory_iterator(dir)) {
		std::stringstream ss;
		std::vector<std::string> dirSplit;
		ss << d.path();
		std::string path = ss.str(), _class;
		path = path.substr(1, path.size() - 2);
		if (ends_with(path, fileExtension)) {
			std::string token;
			ss.str(path);
			while (std::getline(ss, token, '\\')) {
				dirSplit.push_back(token);
			}
			_class = dirSplit[dirSplit.size() - 3];
			cv::Mat_<short> img = cv::imread(path, CV_32S);
			item = { _class, img};
			instanceData.push_back(item);
		}
	}

	return instanceData;
}
