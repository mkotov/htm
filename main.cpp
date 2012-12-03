/* vim: set smartindent expandtab shiftwidth=4 softtabstop=4: */
// Данная программа определяет, какая часть машин Тьюринга заканчивает свою
// работу на ленте, заполненной нулями.  Пусть TM(n) --- множество машин
// Тьюринга с (n + 1) состояниями, при этом предполагается, что в программе
// машины имеются команды со всевозможными левыми частями. Пусть HTM(n) --- те
// машины из TM(n), которые останавливаются через некоторое конечное число
// шагов.  Нас интересует предел отношения |HTM(n)| / |TM(n)| при n стремящемся
// к бесконечности.  Так как у нас физически ограничена лента, то вместо этого
// предела ищем двойной предел |HTM(n, k)| / |TM(n)|, где k и n стремятся к
// бесконечности, и HTM(n, k) означает множество машин, которые останавливаются
// не более чем за k шагов.

#include <getopt.h>
#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cutil.h>
#include <cutil_inline_runtime.h>
#include <mpi.h>

#include "experiment.h"

void printHelp() {
    std::cerr 
        << "Используйте следующие параметры:" << std::endl
        << "--fromMaxStepCount    начальное максимальное число шагов машины;" << std::endl
        << "--countMaxStepCount   сколько раз увеличивать максимальное число шагов машины;" << std::endl
        << "--mDeltaMaxStepCount  мультипликативный шаг, с которым увеличивается " << std::endl
        << "                      максимальное число шагов машины;" << std::endl
        << "--fromStateCount      начальное число состояний машины;" << std::endl
        << "--countStateCount     сколько раз увеличивать конечное число состояний машины;" << std::endl
        << "--deltaStateCount     шаг, с которым увеличивается число состояний машины;" << std::endl
        << "--iterationCount      количество итераций для каждого числа шагов и числа " << std::endl
        << "                      состояний (будет изменено так, чтобы делилось на " << std::endl
        << "                      threadBlockSize);" << std::endl
        << "--seed                затравка для генератора случайных чисел, если не " << std::endl 
        << "                      указана, то берётся случайной;" << std::endl
        << "--threadBlockSize     число нитей на блок;" << std::endl
        << "--stopActionUsed      если в командах должно присутствовать действие S;" << std::endl
        << "--runCount            число прогонов для заданных maxStepCount и stateCount;" << std::endl
        << "--halfTapeUsed        использовать полубесконечную ленту;" << std::endl
        << "--alphabetSize        число символов в алфавите;" << std::endl
        << "--help, -h            вывести эту подсказку." << std::endl << std::endl;
}


void parseParams(
        const int argc, 
        char * const argv[],
        ExperimentParameters * const pep) {
    if (argc <= 1) {
        printHelp();
        exit(EXIT_FAILURE);
    } else {
        int c;
        while (true) {
            option longOptions[] = {
                {"fromMaxStepCount", required_argument, 0, 0},
                {"countMaxStepCount", required_argument, 0, 0},
                {"mDeltaMaxStepCount", required_argument, 0, 0},
                {"fromStateCount", required_argument, 0, 0},
                {"countStateCount", required_argument, 0, 0},
                {"deltaStateCount", required_argument, 0, 0}, 
                {"iterationCount", required_argument, 0, 0}, 
                {"seed", required_argument, 0, 0},
                {"help", no_argument, 0, 'h'},
                {"threadBlockSize", required_argument, 0, 0},
                {"stopActionUsed", no_argument, 0, 0},
                {"runCount", required_argument, 0, 0},
                {"halfTapeUsed", no_argument, 0, 0},
                {"alphabetSize", required_argument, 0, 0},
                {0, 0, 0, 0}
            };
            int optionIndex;
            c = getopt_long(argc, argv, "h:::", longOptions, &optionIndex);
            if (c == -1) {
                break;
            } else if (c == 0) {
                switch (optionIndex) {
                    case 0:
                        pep->fromMaxStepCount = atoi(optarg);
                        break;
                    case 1:
                        pep->countMaxStepCount = atoi(optarg);
                        break;
                    case 2:
                        pep->mDeltaMaxStepCount = atoi(optarg);
                        break;
                    case 3:
                        pep->fromStateCount = atoi(optarg);
                        break;
                    case 4:
                        pep->countStateCount = atoi(optarg);
                        break;
                    case 5:
                        pep->deltaStateCount = atoi(optarg);
                        break;
                    case 6:
                        pep->iterationCount = atoi(optarg);
                        break;
                    case 7:
                        pep->seed = atoi(optarg);
                        break;
                    case 9:
                        pep->threadBlockSize = atoi(optarg);
                        break;
                    case 10:
                        pep->stopActionUsed = true;
                        break;
                    case 11:
                        pep->runCount = atoi(optarg);
                        break;
                    case 12:
                        pep->halfTapeUsed = true;
                        break;
                    case 13:
                        pep->alphabetSize = atoi(optarg);
                        break;
                    default:
                        break;
                }
            } else if (c == 'h') {
                printHelp();
                exit(EXIT_SUCCESS);
            } else {
                printHelp();
                exit(EXIT_FAILURE);
            }
        }
    }
}


void testParams(const ExperimentParameters &ep) {
    if (ep.fromMaxStepCount == 0) {
        std::cerr << "fromMaxStepCount == 0" << std::endl;
        exit(EXIT_FAILURE); 
    }
    if (ep.fromStateCount == 0) {
        std::cerr << "fromStateCount == 0" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (ep.iterationCount == 0) {
        std::cerr << "iterationCount == 0" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (ep.runCount == 0) {
        std::cerr << "runCount == 0" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (ep.threadBlockSize == 0) {
        std::cerr << "threadBlockSize == 0" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (ep.alphabetSize < 2) {
        std::cerr << "alphabetSize < 2" << std::endl;
        exit(EXIT_FAILURE);
    }
}


void printParams(const ExperimentParameters &ep) {
    std::cout << "# fromMaxStepCount=" << ep.fromMaxStepCount << std::endl 
        << "# countMaxStepCount=" << ep.countMaxStepCount << std::endl
        << "# mDeltaMaxStepCount=" << ep.mDeltaMaxStepCount << std::endl
        << "# fromStateCount=" << ep.fromStateCount << std::endl
        << "# countStateCount=" << ep.countStateCount << std::endl
        << "# deltaStateCount=" << ep.deltaStateCount << std::endl
        << "# iterationCount=" << ep.iterationCount << std::endl
        << "# runCount=" << ep.runCount << std::endl
        << "# seed=" << ep.seed << std::endl
        << "# threadBlockSize=" << ep.threadBlockSize << std::endl
        << "# stopActionUsed=" << ep.stopActionUsed << std::endl
        << "# halfTapeUsed=" <<  ep.halfTapeUsed << std::endl
        << "# alphabetSize=" << ep.alphabetSize << std::endl;
}


// Число итераций должно быть кратно числу потоков на блок.
void fixParams(ExperimentParameters * const pep) {
    if (pep->iterationCount % pep->threadBlockSize != 0) {
        pep->iterationCount = (pep->iterationCount / pep->threadBlockSize + 1) * pep->threadBlockSize;
    }
    if (pep->seed == 0) {
        pep->seed = time(NULL);
    }
}	


void printResults(
        const ExperimentParameters &ep,
        const std::vector<size_t> &finishedMachineCounts,
        const std::vector<size_t> &brokenMachineCounts) {
    if (!ep.halfTapeUsed) {
        std::cout << "#stateCount maxStepCount finishedMachineCount / (iterationCount * runCount)" << std::endl;
    } else {
        std::cout << "#stateCount maxStepCount brokenMachineCount / (iterationCount * runCount)" << std::endl;
    }
    size_t stateCount = ep.fromStateCount;
    for (size_t i = 0; i <= ep.countStateCount; ++i) { 
        size_t maxStepCount = ep.fromMaxStepCount;
        for (size_t j = 0; j <= ep.countMaxStepCount; ++j) { 
            const size_t index = j + i * (ep.countMaxStepCount + 1);
            size_t count;
            if (!ep.halfTapeUsed) {
                count = finishedMachineCounts[index];
            } else {
                count = brokenMachineCounts[index];
            }
            const double part = static_cast<double>(count) / (ep.iterationCount * ep.runCount);
            std::cout << stateCount << " " << maxStepCount << " " << part << std::endl;
            maxStepCount *= ep.mDeltaMaxStepCount;
        }
        stateCount += ep.deltaStateCount;
        std::cout << std::endl;
    }
}


void finalize() {
    MPI_Finalize();
}


int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);	
    atexit(&finalize);
    int threadId;
    MPI_Comm_rank(MPI_COMM_WORLD, &threadId);
    ExperimentParameters ep;
    parseParams(argc, argv, &ep);
    if (threadId == 0) {
        testParams(ep);
    }
    fixParams(&ep);
    if (threadId == 0) { 
        printParams(ep);
    }
    const size_t resultSize = getResultSize(ep);
    std::vector<size_t> finishedMachineCounts(resultSize);
    std::vector<size_t> brokenMachineCounts(resultSize);
    runExperiment(ep, finishedMachineCounts, brokenMachineCounts);
    if (threadId == 0) {
        printResults(ep, finishedMachineCounts, brokenMachineCounts);
    }
    return EXIT_SUCCESS;
}

