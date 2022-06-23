#include "AnswerSpace.h"
#include "Array2D.h"
#include "Array2D_CUDA.h"
#include "FilterWords.h"
#include <chrono>
#include <cublas_v2.h>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>

constexpr float Array2D_CUDA::onef;
constexpr float Array2D_CUDA::zerof;

struct Suggestion
{
    size_t index;
    float expectedEntropyDiff;
};

template <typename T>
std::vector<Suggestion> sort_indexes(T* start, size_t n)
{
    // https://stackoverflow.com/a/12399290/5524761
    // initialize original index locations
    std::vector<Suggestion> idx(n);
    for (size_t i = 0; i < n; i++)
    {
        idx[i] = { i, start[i] };
    }

    std::stable_sort(idx.begin(), idx.end(),
                     [&start](Suggestion i1, Suggestion i2)
                     { return start[i1.index] < start[i2.index]; });

    return idx;
}

std::vector<Suggestion> Suggest(cublasHandle_t& cublas, Array2D<float> lookup)
{
    Array2D_CUDA lookup_cuda(lookup);
    Array2D_CUDA results_cuda(lookup); // Doesn't really matter we overwrite values anyway

    Array2D<float> maxEntropy(1, lookup.rows());
    for (int c = 0; c < maxEntropy.cols(); c++)
    {
        maxEntropy.Set(0, c, 0.0f);
    }
    Array2D_CUDA maxEntropy_cuda(maxEntropy);
    Array2D_CUDA rowSums_cuda(maxEntropy);

    Array2D<float> ones(1, lookup.cols());
    std::fill_n(ones.data(), lookup.cols(), 1.0f);
    Array2D_CUDA ones_cuda(ones);

    // Measure time
    auto start = std::chrono::high_resolution_clock::now();

    for (int colour = 0; colour < 243; colour++)
    {
        lookup_cuda.Equals((float)colour, &results_cuda);
        results_cuda.SumRowsBLAS(&rowSums_cuda, ones_cuda, cublas);
        rowSums_cuda.ReduceMax(&maxEntropy_cuda);
    }

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(stop - start);
    float durationMs = duration.count() / 1000000.0f;
    size_t numFilters = lookup.rows() * 243;
    size_t filtersPerSecond = numFilters * 1000ull / durationMs;
    printf("Calculated %llu filters in %f ms (%llu filters/s)\n", numFilters, durationMs,
           filtersPerSecond);

    maxEntropy_cuda.Log2();
    // maxEntropy_cuda.MultiplyScalar(1.0f / lookup.cols());
    maxEntropy_cuda.AddScalar(-log2f(lookup.cols()));

    maxEntropy_cuda.CopyToHost(&maxEntropy);
    // Argsort
    return sort_indexes(maxEntropy.data(), maxEntropy.numel());
}

int main()
{
    cublasHandle_t cublas;
    cublasCreate_v2(&cublas);

    Words words = ReadWords();
    AnswerSpace answerSpace(words.answers);
    if (!IsFileExist("lookup.bin"))
        GenerateLookupTable(words);

    Array2D<uint8_t> lookup_uint8(words.guesses.size(), words.answers.size());
    std::ifstream fp("lookup.bin", std::ios::binary);
    fp.read((char*)(lookup_uint8.data()), lookup_uint8.numel());
    fp.close();

    // Cast uint8 data to int32 (uint32 is also ok)
    Array2D<float> lookup(lookup_uint8.rows(), lookup_uint8.cols());
    for (int r = 0; r < lookup.rows(); r++)
    {
        for (int c = 0; c < lookup.cols(); c++)
        {
            lookup.Set(r, c, (float)lookup_uint8.Get(r, c));
        }
    }

    while (answerSpace.size() > 1)
    {
        if (answerSpace.size() < 10)
        {
            printf("%i possible answers - ", answerSpace.size());
            answerSpace.ShowValid();
            printf("Suggestions:\n");
        }
        else
        {
            printf("%i possible answers. Suggestions:\n", answerSpace.size());
        }
        auto suggestions = Suggest(cublas, lookup);
        for (int i = 0; i < 10; i++)
        {
            auto idx = suggestions[i];
            printf("%s (%f - %f = %f)\n", words.guesses[idx.index].c_str(), answerSpace.entropy(),
                   -idx.expectedEntropyDiff, answerSpace.entropy() + idx.expectedEntropyDiff);
        }

        std::string word;
        std::string colours_;
        printf("Word: ");
        std::getline(std::cin, word);
        printf("Colours: ");
        std::getline(std::cin, colours_);

        // char[] "01200" -> int[] { 0, 1, 2, 0, 0 } -> Colour[] { GREY, ... }
        Colour colours[5];
        for (int pos = 0; pos < 5; pos++)
        {
            switch (colours_[pos] - 48)
            {
            case 0:
                colours[pos] = GREY;
                break;
            case 1:
                colours[pos] = YELLOW;
                break;
            case 2:
                colours[pos] = GREEN;
                break;
            }
        }

        // :O rubbish code :D
        // CommitFilter overwrites the filter mask, so it needs to be last
        answerSpace.Filter(word.c_str(), colours);
        lookup.FilterCols(answerSpace.GetMask());
        answerSpace.CommitFilter();
    }
    answerSpace.ShowValid();

    return 0;
}
