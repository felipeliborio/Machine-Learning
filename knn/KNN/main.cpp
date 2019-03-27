/*
K-NN classifier
There's little to no exception handling. Incorrect entries WILL break the program.

MIT License

Copyright (c) <2019> <Felipe Libório>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "definitions.h"
#include "knn.h"
#include "read_instance.h"

namespace fs = std::filesystem;

void outErr();

/// knn (training_file, instances_file, k{int}, weighted{bool}, compare{bool}, regression{bool})
int main(int argc, char** argv)
{
	/*
	argc = 7;
	std::vector<std::string> argv;
	argv.resize(7);
	argv[1] = "./iris_training.txt";
	argv[2] = "./iris_test.txt";
	argv[3] = "5";
	argv[4] = "0";
	argv[5] = "1";
	argv[6] = "0";
	*/
	std::stringstream ss;
	std::string aux;
	if (argc > 2) {
		ss << argv[1];
		aux = ss.str();
		auto trainingSet = readInstanceData(aux);
		ss.str(std::string());
		ss << argv[2];
		aux = ss.str();
		auto instanceSet = readInstanceData(aux);

		FKNN knn(trainingSet, instanceSet);

		int k = 5;
		bool weighted = false;
		bool compare = false;
		bool regression = false;

		switch (argc) {//the order is inverted so to avoid code rewrite
		case 7:
			ss.str(std::string());
			ss << argv[6];
			aux = ss.str();
			regression = (bool)std::stoi(aux);
		case 6:
			ss.str(std::string());
			ss << argv[5];
			aux = ss.str();
			compare = (bool)std::stoi(aux);
		case 5:
			ss.str(std::string());
			ss << argv[4];
			aux = ss.str();
			weighted = (bool)std::stoi(aux);
		case 4:
			ss.str(std::string());
			ss << argv[3];
			aux = ss.str();
			k = std::stoi(aux);
		case 3:
			knn.run(k, weighted, compare, regression);
			break;
		default:
			outErr();
		}
	}
	else {
		outErr();
	}

	system("pause");
	return 0;
}

void outErr()
{
	std::cout << "Invalid entry!\n\
knn(training_file, instances_file, k{int}, weighted{0, 1}, compare{0, 1}, regression{0, 1})\n\
k{1} - neighbors\
weighted{0} - if true, uses a weighted classifier.\n\
compare{0} - if true, assumes that the classify file contains the answers\
and compare the results.\n\
regression{0} - 1 if k-NN should be used for regression, the property must be a number.\n\
Instance data should be separated by commas and linebreaks.\
\n\n";
}
