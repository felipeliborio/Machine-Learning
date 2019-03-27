#include "read_instance.h"

t_instance_data readInstanceData(std::string & dir, std::string delimiter)
{
	t_instance_data output;
	std::ifstream infile(dir);

	std::string line;
	std::vector<std::string> aux;
	while (std::getline(infile, line)) {
		std::istringstream iss(line);
		std::string s = iss.str();
		size_t pos = 0;
		std::string token;
		while ((pos = s.find(delimiter)) != std::string::npos) {
			token = s.substr(0, pos);
			aux.push_back(token);
			s.erase(0, pos + delimiter.length());
		}
		aux.push_back(s);
		output.push_back(aux);
		aux.clear();
	}

	return output;
}
