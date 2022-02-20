#include "FilterWords.h"
#include "AnswerSpace.h"
#include "Array2D.h"
#include <fstream>
#include <iostream>

bool IsFileExist(const char* fileName)
{
    std::ifstream infile(fileName);
    return infile.good();
}

void GenerateLookupTable(Words& words)
{
    printf("lookup.bin not found. Generating lookup table for %llu answers and %llu guesses. "
           "Please wait.\n",
           words.answers.size(), words.guesses.size());
    AnswerSpace pos(words.answers);

    Array2D<uint8_t> output(words.guesses.size(), words.answers.size());

    for (int r = 0; r < output.rows(); r++)
    {
        for (int c = 0; c < output.cols(); c++)
        {
            output.Set(r, c, (uint8_t)255);
        }
    }

    for (int guess_n = 0; guess_n < words.guesses.size(); guess_n++)
    {
        auto guess = words.guesses[guess_n].c_str();
        for (int prog = 0; prog < 10; prog++)
        {
            if (guess_n == words.guesses.size() / 10 * prog - 1)
                printf("%i0%%\n", prog);
        }
        for (uint8_t _colours = 0; _colours < 243; _colours++)
        {
            Colour colours[5];
            Int2Colours(_colours, colours);

            // Cannot have yellow of character if already greyed exists
            bool greyed[26] = { false };
            bool valid = true;
            for (int _c = 0; _c < 5; _c++)
            {
                auto& c = guess[_c];
                if (colours[_c] == GREY)
                {
                    greyed[Char2Idx(c)] = true;
                }
                else if (colours[_c] == YELLOW && greyed[Char2Idx(c)])
                {
                    valid = false;
                    break;
                }
            }
            if (!valid)
                continue;

            pos.Filter(guess, colours);

            auto& mask = pos.GetMask();
            for (int answer_n = 0; answer_n < mask.size(); answer_n++)
            {
                if (mask[answer_n])
                    output.Set(guess_n, answer_n, (uint8_t)_colours);
            }

            pos.Reset();
        }
    }

    // Write output file
    printf("Done. Writing to lookup.bin\n");
    std::ofstream fp("lookup.bin", std::ios::out | std::ios::binary);
    fp.write((char*)(output.data()), output.rows() * output.cols());
    fp.close();
}
