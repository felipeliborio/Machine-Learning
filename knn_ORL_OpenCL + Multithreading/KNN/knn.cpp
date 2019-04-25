#include "knn.h"

FKNN::FKNN(t_instance_data & trainingSet, t_instance_data & instanceSet)
{
	this->_trainingSet = trainingSet;
	this->_instanceSet = instanceSet;
	_outputVec.resize(instanceSet.size());
}


std::string FKNN::_getOutput(std::vector<std::pair<double, int>> & kNeighbors, bool weighted)
{
	std::string output;

	std::map<std::string, double> votes;

	for (auto neighbor : kNeighbors) {
		if (votes.find(_trainingSet[neighbor.second].first) == votes.end()) {
			votes[_trainingSet[neighbor.second].first] = 0;
		}
		double vote = 1.0 / kNeighbors.size();
		if (weighted) { vote *= 1.0 / neighbor.first; }
		votes[_trainingSet[neighbor.second].first] += vote;
	}

	output = (*votes.begin()).first;
	double voteCount = (*votes.begin()).second;
	for (auto& candidate : votes) {
		if (candidate.second > voteCount) {
			output = candidate.first;
			voteCount = candidate.second;
		}
	}

	return output;
}


double FKNN::_getDistance(int indexIS, int indexTS)
{
	cv::Mat_<short> dif = _trainingSet[indexTS].second - _instanceSet[indexIS].second;
	return sqrt(cv::sum(dif.mul(dif))[0]);
}


std::string FKNN::_assessItem(int k, int index, bool weighted)
{
	std::vector<std::pair<double, int>> distanceVec;
	for (int i = 0; i < _trainingSet.size(); i++) {
		distanceVec.push_back(std::make_pair(_getDistance(index, i), i));
	}
	std::sort(distanceVec.begin(), distanceVec.end());
	distanceVec.resize(k);

	return _getOutput(distanceVec, weighted);
}


void FKNN::_runRange(int k, bool weighted, int start, int end)
{
	for (int i = start; i < end; i++) {
		_outputVec[i] = _assessItem(k, i, weighted);
	}
}


void FKNN::_startThreads(int& k, bool& weighted)
{
	double threads = std::thread::hardware_concurrency();//may return 0 when not able to detect
	double split = _instanceSet.size() / (threads > 0 ? threads : 1);
	std::vector<std::thread> tVec;
	for (double i = 0; i < threads; i++) {
		tVec.push_back(std::thread(&FKNN::_runRange, this, k, weighted, i * split,
			(i + 1 < threads) ? (i + 1) * split : _instanceSet.size()));
	}
	for (auto& thread : tVec) {
		thread.join();
	}
}


void FKNN::run(int k, bool weighted, bool compare)
{
	_startThreads(k, weighted);
	if (compare) { _updateAccuracy(); }
	_print(compare);
}


double FKNN::run_accOnly(int k, bool weighted)
{
	_startThreads(k, weighted);
	_updateAccuracy();
	return _accuracy;
}


void FKNN::_updateAccuracy()
{
	double hits = 0;
	for (int i = 0; i < _instanceSet.size(); i++) {
		if (_instanceSet[i].first == _outputVec[i]) {
			hits++;
		}
	}
	_accuracy = hits / _instanceSet.size();
}


void FKNN::_print(bool compare)
{
	for (int i = 0; i < _instanceSet.size(); i++) {
		std::cout << i + 1 << " " << _instanceSet[i].first << "\t";
		if (compare) { std::cout << "guess: "; }
		std::cout << _outputVec[i] << "\n";
	}
	if (compare) { std::cout << "\nAccuracy: " << _accuracy << "\n"; }
}
