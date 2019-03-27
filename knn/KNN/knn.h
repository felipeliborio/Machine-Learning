#pragma once
#include "definitions.h"

class FKNN
{
public:
	/// (training, instances, k{int}, weighted{0, 1}, compare{0, 1}, regression{0, 1})
	FKNN(t_instance_data & trainingSet, t_instance_data & instanceSet);
	void run(int k = 1, bool weighted = false, bool compare = false, bool regression = false);

private:
	t_instance_data _trainingSet;
	t_instance_data _instanceSet;

	std::vector<std::string> _outputVec;
	double _accuracy;
	void _print(bool compare);

	std::string FKNN::_assessLine(int k, int index, bool weighted, bool regression);
	double _getDistance(int indexIS, int indexTS);
	std::string _getRegressionOutput(std::vector<std::pair<double, int>> & distanceVec, bool weighted);
	std::string _getClassificationOutput(std::vector<std::pair<double, int>> & distanceVec, bool weighted);
	std::string _getOutput(std::vector<std::pair<double, int>> & distanceVec,
		bool weighted, bool regression);
	void _updateAccuracy();
};
