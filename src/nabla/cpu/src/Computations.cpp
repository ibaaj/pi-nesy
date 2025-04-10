//
//  Computations.cpp
//



#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <future>


void ComputeNablaSingle(const std::vector<std::vector<float>>& A, 
                        const std::vector<float>& b, 
                        float& nablaResult, 
                        const float epsilon, int i) {
    int N = A.size();
    int M = A[0].size();
    float nablaMin = 1.0f;
    for (int j = 0; j < M; ++j) {
            float nabla_ij = std::fmax(0.0f, A[i][j] - b[i]);
            for (int k = 0; k < N; ++k) {
                float sigmaEpsilon = std::fmin(std::fmax(0.0f, (b[k] - b[i])) / 2.0f, std::fmax(0.0f, b[k] - A[k][j]));
                nabla_ij = std::fmax(sigmaEpsilon, nabla_ij);

                if (nabla_ij > 1.0f - epsilon)
                    break;
            }
            nablaMin = std::min(nablaMin, nabla_ij);
            if (nablaMin < epsilon)
                break;
        }
    nablaResult = nablaMin;
}

void ComputeNablaChunk(const std::vector<std::vector<float>>& A, 
                       const std::vector<float>& b, 
                       std::vector<float>& nablaResults, 
                       const float epsilon, 
                       int start, int end) {
    for (int i = start; i < end; ++i) {
        ComputeNablaSingle(A, b, nablaResults[i], epsilon, i);
    }
}

void ComputeNabla(const std::vector<std::vector<float>>& A, 
                  const std::vector<float>& b, 
                  std::vector<float>& nablaResults, 
                  const float epsilon) {
    int N = A.size();
    nablaResults.resize(N);

    // Use hardware concurrency as a basis for the number of chunks
    unsigned int maxThreads = std::thread::hardware_concurrency();
    int chunkSize = std::ceil(N / static_cast<float>(maxThreads));

    std::vector<std::future<void>> futures;

    for (int i = 0; i < N; i += chunkSize) {
        int end = std::min(i + chunkSize, N);
        futures.push_back(std::async(std::launch::async, ComputeNablaChunk, std::cref(A), std::cref(b), std::ref(nablaResults), epsilon, i, end));
    }

    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get();
    }
}


// Function to compute min-max for a single row
void ComputeMinMaxSingle(const std::vector<std::vector<float>>& A_in, 
                         const std::vector<float>& x_in, 
                         float& result, 
                         float epsilon, int i) {
    int M = x_in.size();
    float maxTerm;
    float res = 1.0f;
    for (int j = 0; j < M; ++j) {
        maxTerm = std::max(A_in[i][j], x_in[j]);
        res = std::min(res, maxTerm);
        if (res < epsilon) 
            break;
    }
    result = res;
}

// Helper function to compute a chunk of the min-max results
void ComputeMinMaxChunk(const std::vector<std::vector<float>>& A_in, 
                        const std::vector<float>& x_in, 
                        std::vector<float>& minMaxResults, 
                        float epsilon, int start, int end) {
    for (int i = start; i < end; ++i) {
        ComputeMinMaxSingle(A_in, x_in, minMaxResults[i], epsilon, i);
    }
}

void ComputeMinMax(const std::vector<std::vector<float>>& A_in, 
                   const std::vector<float>& x_in, 
                   std::vector<float>& minMaxResults, 
                   float epsilon) {
    int N = A_in.size();
    minMaxResults.resize(N);

    // Use hardware concurrency as a basis for the number of chunks
    unsigned int maxThreads = std::thread::hardware_concurrency();
    int chunkSize = std::ceil(N / static_cast<float>(maxThreads));

    std::vector<std::future<void>> futures;

    for (int i = 0; i < N; i += chunkSize) {
        int end = std::min(i + chunkSize, N);
        futures.push_back(std::async(std::launch::async, ComputeMinMaxChunk, std::cref(A_in), std::cref(x_in), std::ref(minMaxResults), epsilon, i, end));
    }

    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get(); // Blocking call to ensure completion
    }
}

void ComputePotentialMinSolutionSingle(const std::vector<std::vector<float>>& A, 
                                       const std::vector<float>& b, 
                                       float& colResult, 
                                       const float epsilon, int j) {
    int N = A.size();
    float col = -std::numeric_limits<float>::max();
    for (int i = 0; i < N; ++i) {
        float tmp1 = std::max(std::fabs(A[i][j]), std::fabs(b[i]));
        tmp1 = tmp1 * epsilon;
        tmp1 = std::fmax(tmp1, 1e-9);
        float epsilonTerm;

        if (std::fabs(A[i][j] - b[i]) <= tmp1) {
            epsilonTerm = 0.0; // Return 0 if they are approximately equal
        } else {
            if (A[i][j] < b[i]) {
                epsilonTerm = b[i];
            } else {
                epsilonTerm = 0.0;
            }
        }

        col = std::max(epsilonTerm, col);

        if (col > 1.0 - epsilon)
            break;
    }
    colResult = col;
}

// Helper function to compute a chunk of the potential min solutions
void ComputePotentialMinSolutionChunk(const std::vector<std::vector<float>>& A, 
                                      const std::vector<float>& b, 
                                      std::vector<float>& sol, 
                                      float epsilon, int start, int end) {
    for (int j = start; j < end; ++j) {
        ComputePotentialMinSolutionSingle(A, b, sol[j], epsilon, j);
    }
}

void ComputePotentialMinSolution(const std::vector<std::vector<float>>& A, 
                                  const std::vector<float>& b, 
                                  std::vector<float>& sol, 
                                  float epsilon) {
    int M = A[0].size();
    sol.resize(M);

    // Use hardware concurrency as a basis for the number of chunks
    unsigned int maxThreads = std::thread::hardware_concurrency();
    int chunkSize = std::ceil(M / static_cast<float>(maxThreads));

    std::vector<std::future<void>> futures;

    for (int j = 0; j < M; j += chunkSize) {
        int end = std::min(j + chunkSize, M);
        futures.push_back(std::async(std::launch::async, ComputePotentialMinSolutionChunk, std::cref(A), std::cref(b), std::ref(sol), epsilon, j, end));
    }

    // Wait for all tasks to complete
    for (auto& future : futures) {
        future.get(); // Blocking call to ensure completion
    }
}




void LowestApproxSolution(const std::vector<std::vector<float>>& A, 
                                 const std::vector<float>& b, 
                                 std::vector<float>& res, 
                                 const float nabla,
                                 const float epsilon)
{
    std::vector<float> modified_b;
    for(int i = 0; i < A.size(); ++i){
        
        modified_b.emplace_back(std::fmax(b[i] - nabla, 0.0f));
    }
    
    ComputePotentialMinSolution(A, modified_b, res, epsilon);
}


