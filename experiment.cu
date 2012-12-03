/* vim: set smartindent expandtab shiftwidth=4 softtabstop=4: */

#include <curand_kernel.h>
#include <cutil.h>
#include <cutil_inline_runtime.h>
#include <vector>
#include <mpi.h>
#include "experiment.h"

typedef size_t Symbol;
typedef size_t State;
typedef int Shift; 

struct Command {
    State newState;
    Symbol newSymbol;
    Shift shift;
};


__device__ __host__ __inline__ size_t getProgramSize(
        const size_t stateCount,
        const size_t alphabetSize) {
    return stateCount * alphabetSize;
}


__host__ Command *allocPrograms(
        const size_t stateCount, 
        const size_t alphabetSize,
        const size_t iterationCount) {
    Command *pPrograms = 0;
    const size_t programSize = getProgramSize(stateCount, alphabetSize);
    cutilSafeCall(cudaMalloc((void **)&pPrograms, 
        iterationCount * programSize * sizeof(Command)));
    return pPrograms;
}


__host__ void freePrograms(Command * const pPrograms) {
    cutilSafeCall(cudaFree(pPrograms));
}


__inline__ __device__ State getRandomState(
        curandState * const pRNGState, 
        const size_t stateCount) {
    return curand(pRNGState) % (stateCount + 1);
}


__inline__ __device__ Symbol getRandomSymbol(
        curandState * const pRNGState, 
        const size_t alphabetSize) {
    return curand(pRNGState) % alphabetSize;
}


__inline__ __device__ Shift getRandomShift(
        curandState * const pRNGState, 
        const bool stopActionUsed) {
    if (stopActionUsed) {
        return curand(pRNGState) % 3 - 1;
    } else {
        return 2 * (curand(pRNGState) % 2) - 1;
    }
}


__inline__ __device__ void createRandomCommand(
        curandState * const pRNGState, 
        const size_t stateCount, 
        const bool stopActionUsed,
        const size_t alphabetSize,
        Command * const pCommand) {
    pCommand->newState = getRandomState(pRNGState, stateCount);
    pCommand->newSymbol = getRandomSymbol(pRNGState, alphabetSize);
    pCommand->shift = getRandomShift(pRNGState, stopActionUsed);
}


__global__ void createRandomPrograms(
        const size_t stateCount, 
        const bool stopActionUsed, 
        const size_t alphabetSize,
        curandState * const pRNGStates, 
        Command * const pPrograms) {
    const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    curandState * const pRNGState = pRNGStates + index;
    const size_t programSize = getProgramSize(stateCount, alphabetSize);
    Command * const pProgram = pPrograms + programSize * index;
    for (size_t i = 0; i < programSize; ++i) {
        createRandomCommand(pRNGState, stateCount, stopActionUsed, 
            alphabetSize, pProgram + i);
    }
}


__host__ curandState *allocRNGStates(const size_t iterationCount) {
    curandState *pRNGStates = 0;
    cutilSafeCall(cudaMalloc((void **)&pRNGStates, 
        iterationCount * sizeof(curandState)));
    return pRNGStates;
}


__host__ void freeRNGStates(curandState * const pRNGStates) {
    cutilSafeCall(cudaFree(pRNGStates));
}


__global__ void createRNGStates(
        const size_t seed, 
        curandState * const pRNGStates) {
    const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, index, 0, pRNGStates + index);
}


__device__ __host__ __inline__ size_t getTapeSize(
        const size_t toMaxStepCount) {
    return 2 * toMaxStepCount + 1;
}


__host__ Symbol *allocTapes(
        const size_t toMaxStepCount, 
        const size_t iterationCount) {
    Symbol *pTapes = 0;
    const size_t tapeSize = getTapeSize(toMaxStepCount);
    cutilSafeCall(cudaMalloc((void **)&pTapes, 
        tapeSize * iterationCount * sizeof(Symbol)));
    return pTapes;
}


__host__ void freeTapes(Symbol * const pTapes) {
    cutilSafeCall(cudaFree(pTapes));
}


__host__ void initTapes(Symbol * const pTapes, 
        const size_t iterationCount, 
        const size_t toMaxStepCount) {
    const size_t tapeSize = getTapeSize(toMaxStepCount);
    cutilSafeCall(cudaMemset(pTapes, 0, 
        tapeSize * iterationCount * sizeof(Symbol)));
}


__host__ State *allocStates(const size_t iterationCount) {
    State *pStates = 0;
    cutilSafeCall(cudaMalloc((void **)&pStates, 
        iterationCount * sizeof(State)));
    return pStates;
}


__host__ void freeStates(State * const pStates) { 
    cutilSafeCall(cudaFree(pStates));
}


__host__ void initStates(
        State * const pStates, 
        const size_t iterationCount) {
    cutilSafeCall(cudaMemset(pStates, 0, iterationCount * sizeof(State)));
}


__host__ size_t *allocPositions(const size_t iterationCount) {
    size_t *pPositions = 0;
    cutilSafeCall(cudaMalloc((void **)&pPositions, 
        iterationCount * sizeof(size_t)));
    return pPositions;
}


__host__ void freePositions(size_t * const pPositions) {
    cutilSafeCall(cudaFree(pPositions));
}


__global__ void initPositions(
        size_t * const pPositions, 
        const size_t toMaxStepCount) {
    const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    pPositions[index] = toMaxStepCount;
}


__host__ size_t getMachineInStateCount(
        const size_t iterationCount, 
        const State * const pDeviceStates,
        const State state) {
    size_t * const pHostStates = new State[iterationCount];
    cutilSafeCall(cudaMemcpy(pHostStates, pDeviceStates, 
        iterationCount * sizeof(State), cudaMemcpyDeviceToHost));
    size_t machineInStateCount = 0;
    for (size_t i = 0; i < iterationCount; ++i) {
        if (pHostStates[i] == state) {
            machineInStateCount += 1;
        }
    }
    delete [] pHostStates;
    return machineInStateCount;
}


__global__ void runMachines(
        const size_t stateCount,
        const size_t toMaxStepCount,
        const size_t alphabetSize,
        const Command * const pPrograms,
        const size_t stepCount,
        const State finalState,
        Symbol * const pTapes,
        State * const pStates,
        size_t * const pPositions) {
    const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t programSize = getProgramSize(stateCount, alphabetSize);
    const size_t tapeSize = getTapeSize(toMaxStepCount);
    const Command * const pProgram = pPrograms + programSize * index;
    Symbol * const pTape = pTapes + tapeSize * index;
    State state = pStates[index];
    size_t position = pPositions[index];
    for (size_t i = 0; i < stepCount; ++i) {
        if (state == finalState) {
            break;	
        }
        Symbol * const pSymbol = pTape + position;
        const Command * const pCommand = 
            pProgram + alphabetSize * state + *pSymbol;
        *pSymbol = pCommand->newSymbol;
        position += pCommand->shift;
        state = pCommand->newState;		
    }
    pStates[index] = state;
    pPositions[index] = position;
}


__global__ void runMachinesOnHalfTape(
        const size_t stateCount,
        const size_t toMaxStepCount,
        const size_t alphabetSize,
        const Command * const pPrograms,
        const size_t stepCount,
        const State finalState,
        const State brokenState,
        Symbol * const pTapes,
        State * const pStates,
        size_t * const pPositions) {
    const size_t index = threadIdx.x + blockIdx.x * blockDim.x;
    const size_t programSize = getProgramSize(stateCount, alphabetSize);
    const size_t tapeSize = getTapeSize(toMaxStepCount);
    const Command * const pProgram = pPrograms + programSize * index;
    Symbol * const pTape = pTapes + tapeSize * index;
    State state = pStates[index];
    size_t position = pPositions[index];
    for (size_t i = 0; i < stepCount; ++i) {
        if (state == finalState || state == brokenState) {
            break;	
        }
        Symbol * const pSymbol = pTape + position;
        const Command * const pCommand = 
            pProgram + alphabetSize * state + *pSymbol;
        *pSymbol = pCommand->newSymbol;
        position += pCommand->shift;
        if (position < toMaxStepCount) {
            state = brokenState;
            break;
        }
        state = pCommand->newState;		
    }
    pStates[index] = state;
    pPositions[index] = position;
}


__host__ size_t getToMaxStepCount(const ExperimentParameters &ep) {
    size_t toMaxStepCount = ep.fromMaxStepCount;
    for (size_t i = 0; i <= ep.countMaxStepCount; ++i) {
            toMaxStepCount *= ep.mDeltaMaxStepCount;
    }
    return toMaxStepCount;
}


__host__ size_t getToStateCount(const ExperimentParameters &ep) {
    size_t toStateCount = ep.fromStateCount;
    for (size_t i = 0; i <= ep.countStateCount; ++i) {
            toStateCount += ep.deltaStateCount;
    }
    return toStateCount;
}


__host__ void runExperiment(
        const ExperimentParameters &ep,
        const size_t stateCount, 
        curandState * const pRNGStates,
        Command * const pPrograms,
        Symbol * const pTapes,
        State * const pStates,
        size_t * const pPositions,
        size_t * const finishedMachineCounts,
        size_t * const brokenMachineCounts) { 
    initStates(pStates, ep.iterationCount);
    const dim3 threads = dim3(ep.threadBlockSize, 1, 1);
    const dim3 blocks = dim3(ep.iterationCount / threads.x, 1, 1);
    initPositions<<<blocks, threads>>>(pPositions, getToMaxStepCount(ep));
    cutilSafeCall(cudaDeviceSynchronize());
    initTapes(pTapes, ep.iterationCount, getToMaxStepCount(ep));
    createRandomPrograms<<<blocks, threads>>>(stateCount, ep.stopActionUsed, 
        ep.alphabetSize, pRNGStates, pPrograms);
    cutilSafeCall(cudaDeviceSynchronize());
    const State finalState = stateCount;
    const State brokenState = stateCount + 1;
    size_t from = 0;
    size_t to = ep.fromMaxStepCount;
    for (size_t j = 0; j <= ep.countMaxStepCount; ++j) {
        if (!ep.halfTapeUsed) {
            runMachines<<<blocks, threads>>>(stateCount, getToMaxStepCount(ep),
                ep.alphabetSize, pPrograms, to - from, finalState, pTapes, 
                pStates, pPositions);
            cutilSafeCall(cudaDeviceSynchronize());
            finishedMachineCounts[j] += 
                getMachineInStateCount(ep.iterationCount, pStates, finalState);
        } else {
            runMachinesOnHalfTape<<<blocks, threads>>>(stateCount, 
                getToMaxStepCount(ep), ep.alphabetSize, 
                pPrograms, to - from, finalState, brokenState, pTapes,
                pStates, pPositions);
            cutilSafeCall(cudaDeviceSynchronize());
            finishedMachineCounts[j] +=
                getMachineInStateCount(ep.iterationCount, pStates, finalState);
            brokenMachineCounts[j] += 
                getMachineInStateCount(ep.iterationCount, pStates, brokenState);
        }
        from = to;
        to = to * ep.mDeltaMaxStepCount;
    }
}

__host__ void runExperiment (
        const ExperimentParameters &ep, 
        const size_t threadCount,
        const size_t threadId,
        size_t *finishedMachineCounts,
        size_t *brokenMachineCounts) {
    curandState * const pRNGStates = allocRNGStates(ep.iterationCount);
    const dim3 threads = dim3(ep.threadBlockSize, 1, 1);
    const dim3 blocks = dim3(ep.iterationCount / threads.x, 1, 1);
    createRNGStates<<<blocks, threads>>>(ep.seed, pRNGStates);
    cutilSafeCall(cudaDeviceSynchronize());
    Command * const pPrograms = allocPrograms(getToStateCount(ep), 
        ep.alphabetSize, ep.iterationCount);
    Symbol * const pTapes = allocTapes(getToMaxStepCount(ep), 
        ep.iterationCount);
    State * const pStates = allocStates(ep.iterationCount);
    size_t * const pPositions = allocPositions(ep.iterationCount);
    for (size_t i = 0; i <= ep.countStateCount; ++i) { 
        if (i % threadCount == threadId) {
            for (size_t run = 0; run < ep.runCount; ++run) {
                runExperiment(ep, ep.fromStateCount + i * ep.deltaStateCount,
                        pRNGStates, pPrograms, pTapes, pStates, pPositions, 
                        finishedMachineCounts + i * (ep.countMaxStepCount + 1), 
                        brokenMachineCounts + i * (ep.countMaxStepCount + 1));
            }
        }
    }
    freePositions(pPositions);
    freeStates(pStates);
    freeTapes(pTapes);
    freePrograms(pPrograms);
    freeRNGStates(pRNGStates);
}


size_t getResultSize(const ExperimentParameters &ep) {
    return (ep.countStateCount + 1) * (ep.countMaxStepCount + 1);
}


void runExperiment(
        const ExperimentParameters &ep, 
        std::vector<size_t> &finishedMachineCounts, 
        std::vector<size_t> &brokenMachineCounts) {
    int threadId;
    MPI_Comm_rank(MPI_COMM_WORLD, &threadId);
    int threadCount;
    MPI_Comm_size(MPI_COMM_WORLD, &threadCount);
    size_t localThreadId = atoi(getenv("OMPI_COMM_WORLD_LOCAL_RANK"));
    size_t deviceId = localThreadId;
    cutilSafeCall(cudaSetDevice(deviceId));
    if (threadId == 0) {
        std::cout << "# threadCount=" << threadCount << std::endl;
    }
    std::cout << "# thread" << threadId << ".deviceId=" << deviceId << std::endl;
    std::cout << "# thread" << threadId << ".localThreadId=" << localThreadId << std::endl;
    cudaDeviceProp deviceProperties;
    cutilSafeCall(cudaGetDeviceProperties(&deviceProperties, deviceId));
    std::cout << "# thread" << threadId << ".deviceName=" << deviceProperties.name << std::endl;
    char nodeName[256];
    gethostname(nodeName, sizeof(nodeName));
    std::cout << "# thread" << threadId << ".nodeName=" << nodeName << std::endl;

    size_t resultSize = getResultSize(ep);
    runExperiment(ep, threadCount, threadId, &finishedMachineCounts[0], 
        &brokenMachineCounts[0]);
    size_t *allFinishedMachineCounts;
    size_t *allBrokenMachineCounts;
    if (threadId == 0) {
        allFinishedMachineCounts = new size_t[resultSize * threadCount];
        allBrokenMachineCounts = new size_t[resultSize * threadCount];
        
    }
    MPI_Datatype MPI_SIZE_T;
    if (sizeof(size_t) == sizeof(unsigned long)) {
        MPI_SIZE_T = MPI_UNSIGNED_LONG;
    } else if (sizeof(size_t) == sizeof(unsigned int)) {
        MPI_SIZE_T = MPI_UNSIGNED;
    } else {
        std::cerr << "The type size_t has unknown size" << std::endl;
        exit(EXIT_FAILURE);
    }
    MPI_Gather(&finishedMachineCounts[0], resultSize, MPI_SIZE_T,
            allFinishedMachineCounts, resultSize, MPI_SIZE_T, 
            0, MPI_COMM_WORLD);
    MPI_Gather(&brokenMachineCounts[0], resultSize, MPI_SIZE_T,
            allBrokenMachineCounts, resultSize, MPI_SIZE_T, 
            0, MPI_COMM_WORLD);
    if (threadId == 0) {
        for (size_t i = 1; i < threadCount; ++i) {
            for (size_t j = 0; j < resultSize; ++j) {
                size_t index = j + i *resultSize;
                finishedMachineCounts[j] += allFinishedMachineCounts[index];
                brokenMachineCounts[j] += allBrokenMachineCounts[index];
            }
        }
        delete [] allFinishedMachineCounts;
        delete [] allBrokenMachineCounts;
    }
}

