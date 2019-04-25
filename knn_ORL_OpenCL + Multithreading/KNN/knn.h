#pragma once
#include "definitions.h"

class FKNN
{
public:
	/// (training, instances, k{int}, weighted{0, 1}, compare{0, 1}, regression{0, 1})
	FKNN(t_instance_data & trainingSet, t_instance_data & instanceSet);
	void run(int k = 1, bool weighted = false, bool compare = true);
	double run_accOnly(int k = 1, bool weighted = false);

private:
	t_instance_data _trainingSet;
	t_instance_data _instanceSet;

	std::vector<std::string> _outputVec;
	double _accuracy = 0;
	void _print(bool compare);

	void _startThreads(int& k, bool& weighted);
	void _runRange(int k, bool weighted, int start, int end);
	std::string FKNN::_assessItem(int k, int index, bool weighted);
	double _getDistance(int indexIS, int indexTS);
	std::string _getOutput(std::vector<std::pair<double, int>> & distanceVec, bool weighted);
	void _updateAccuracy();
};
