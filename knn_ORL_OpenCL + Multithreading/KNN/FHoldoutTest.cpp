#include "FHoldoutTest.h"


FHoldoutTest::FHoldoutTest(t_instance_data & data)
{
	_updateDataMap(data);
}


FHoldoutTest::~FHoldoutTest()
{
}


void FHoldoutTest::run(int K, bool weighted, int iterations, double rate_Train_Test)
{
	srand(0);//so the instances generated are the same for consecultive runs with different parameters
	double sum = 0;
	std::vector<double> accVec;
	std::cout << "iteration\taccuracy\n";
	for (int i = 0; i < iterations; i++) {
		t_instance_data training, test;
		_divideDataSet(training, test, rate_Train_Test);
		FKNN knn(training, test);
		double result = knn.run_accOnly(K, weighted);
		sum += result;
		accVec.push_back(result);
		std::cout << i + 1 << "\t\t" << result << "\n";
	}
	auto avg = sum / iterations;
	std::cout << "Average accuracy: " << avg << "\n";
	sum = 0;
	for (auto & a : accVec) {
		sum += pow(a - avg, 2);
	}
	double s = (s == 0) ? 0 : sqrt(sum / iterations);
	std::cout << "Standard deviation: " << s << "\n\n";
}

void FHoldoutTest::_updateDataMap(t_instance_data & data)
{
	for (auto& _class : data) {
		if (_dataMap.find(_class.first) == _dataMap.end()) {
			_dataMap[_class.first] = t_instance_data();
		}
		_dataMap[_class.first].push_back(_class);

	}
}

void FHoldoutTest::_divideDataSet(OUT t_instance_data & training, OUT t_instance_data & test, double rate)
{
	auto dM = _dataMap;
	for (auto & c : dM) {
		int eCount = c.second.size();
		while (c.second.size() > eCount - (eCount * rate)) {
			swap(c.second[rand() % c.second.size()], c.second[c.second.size() - 1]);
			training.push_back(c.second[c.second.size() - 1]);
			c.second.pop_back();
		}
		for (auto & e : c.second) {
			test.push_back(e);
		}
	}
}
