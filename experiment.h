/* vim: set smartindent expandtab shiftwidth=4 softtabstop=4: */
#ifndef _EXPERIMENT_H_
#define _EXPERIMENT_H_

#include <vector>

const size_t DEFAULT_ALPHABET_SIZE = 2;

struct ExperimentParameters {
    size_t fromMaxStepCount;
    size_t countMaxStepCount;
    size_t mDeltaMaxStepCount;
    size_t fromStateCount;
    size_t countStateCount;
    size_t deltaStateCount;
    size_t iterationCount;
    size_t runCount;
    size_t seed;
    size_t threadBlockSize;
    bool stopActionUsed;
    bool halfTapeUsed;
    size_t alphabetSize;
    bool parallel;

    ExperimentParameters(): 
        fromMaxStepCount(0),
        countMaxStepCount(0),
        mDeltaMaxStepCount(0),
        fromStateCount(0),
        countStateCount(0),
        deltaStateCount(0),
        iterationCount(0),
        runCount(0),
        seed(0),
        threadBlockSize(0),
        stopActionUsed(false),
        halfTapeUsed(false),
        alphabetSize(DEFAULT_ALPHABET_SIZE),
        parallel(false) {
    }
};


void runExperiment(
        const ExperimentParameters &ep, 
        std::vector<size_t> &finishedMachineCounts, 
        std::vector<size_t> &brokenMachineCounts);

size_t getResultSize(const ExperimentParameters &ep);


#endif
