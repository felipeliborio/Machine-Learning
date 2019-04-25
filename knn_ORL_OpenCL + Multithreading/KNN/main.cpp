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
#include "read_instances.h"
#include "FHoldOutTest.h"

namespace fs = std::filesystem;

void outErr();

/// knn (database)
int main(int argc, char** argv)
{
	if (argc >= 0) {
		std::stringstream ss;
		std::string aux;
		//ss << argv[1];
		//aux = ss.str();
		//t_instance_data data = readInstanceData(aux);
		std::string dir = "./../att_faces";
		t_instance_data data = readDirData(dir, std::string(".pgm"));
		FHoldoutTest holdoutTest = FHoldoutTest(data);
		
		std::cout << "Holdout 1NN\n";
		holdoutTest.run(1, false, 100, 0.5);

		std::cout << "\n\nHoldout 3NN\n";
		holdoutTest.run(3, false, 100, 0.5);
	}
	else {
		outErr();
	}

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
