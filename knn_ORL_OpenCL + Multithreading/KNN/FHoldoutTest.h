#pragma once
#include "definitions.h"
#include "knn.h"

class FHoldoutTest
{
public:
	FHoldoutTest(t_instance_data & data);
	~FHoldoutTest();
	void run(int K = 50, bool weighted = true, int iterations = 100, double rate_Train_Test = 0.5);

private:
	std::map<std::string, t_instance_data> _dataMap;
	void _updateDataMap(t_instance_data & data);
	void _divideDataSet(OUT t_instance_data & training, OUT t_instance_data & test, double rate);
};

