#include "knn.h"

FKNN::FKNN(t_instance_data & trainingSet, t_instance_data & instanceSet)
{
	this->_trainingSet = trainingSet;
	this->_instanceSet = instanceSet;
	_outputVec.resize(instanceSet.size());
}

std::string FKNN::_getRegressionOutput(std::vector<std::pair<double, int>> & distanceVec, bool weighted)
{
	std::string output;
	double out = 0;
	for (auto & e : distanceVec) {
		out += std::stod(_trainingSet[e.second][_trainingSet[e.second].size() - 1]);
	}
	out /= distanceVec.size();
	if (weighted) {
		double aux = out;
		for (auto & e : distanceVec) {
			out += (1.0 / e.first) * (std::stod(_trainingSet[e.second][_trainingSet[e.second].size() - 1]) - aux);
		}
	}
	std::stringstream ss;
	ss << out;
	output = ss.str();
	return output;
}

std::string FKNN::_getClassificationOutput(std::vector<std::pair<double, int>> & distanceVec, bool weighted)
{
	std::string output;

	std::vector<std::pair<double, int>> votes;

	bool voted;
	for (auto & e : distanceVec) {
		voted = false;
		double vote = 1.0 / distanceVec.size();
		if (weighted) { vote *= 1.0 / e.first; }//this doesn't seem to be good enough
		for (int i = 0; i < votes.size(); i++) {
			if (votes[i].second == e.second) {
				votes[i].first += vote;
				voted = true;
				break;
			}
		}
		if (!voted) {
			votes.push_back(std::make_pair(vote, e.second));
		}
	}
	std::sort(votes.begin(), votes.end());
	auto aux = votes[votes.size() - 1].second;
	output = _trainingSet[aux][_trainingSet[aux].size() - 1];
	return output;
}

std::string FKNN::_getOutput(std::vector<std::pair<double, int>> & distanceVec, bool weighted, bool regression)
{
	std::string output;
	if (regression) {
		output = _getRegressionOutput(distanceVec, weighted);
	}
	else {
		output = _getClassificationOutput(distanceVec, weighted);
	}
	return output;
}

double FKNN::_getDistance(int indexIS, int indexTS)
{
	double sum = 0;
	for (int i = 0; i < _trainingSet[indexTS].size() - 1; i++) {
		sum += pow(std::stod(_trainingSet[indexTS][i]) - std::stod(_instanceSet[indexIS][i]), 2);
	}
	return sqrt(sum);
}

std::string FKNN::_assessLine(int k, int index, bool weighted, bool regression)
{
	std::vector<std::pair<double, int>> distanceVec;
	for (int i = 0; i < _trainingSet.size(); i++) {
		distanceVec.push_back(std::make_pair(_getDistance(index, i), i));
	}
	std::sort(distanceVec.begin(), distanceVec.end());
	distanceVec.resize(k);

	return _getOutput(distanceVec, weighted, regression);
}

void FKNN::run(int k, bool weighted, bool compare, bool regression)
{
	for (int i = 0; i < _instanceSet.size(); i++) {
		_outputVec[i] = _assessLine(k, i, weighted, regression);
	}
	if (compare) { _updateAccuracy(); }
	_print(compare);
}

void FKNN::_updateAccuracy()
{
	double hits = 0;
	for (int i = 0; i < _instanceSet.size(); i++) {
		if (_instanceSet[i][_instanceSet[i].size() - 1] == _outputVec[i]) {
			hits++;
		}
	}
	_accuracy = hits / _instanceSet.size();
}

void FKNN::_print(bool compare)
{
	for (int i = 0; i < _instanceSet.size(); i++) {
		for (auto e : _instanceSet[i]) {
			std::cout << e << "\t";
		}
		if (compare) { std::cout << "guess: "; }
		std::cout << _outputVec[i] << "\n";
	}
	if (compare) { std::cout << "\nAccuracy: " << _accuracy << "\n"; }
}
