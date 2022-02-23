#include "AnswerSpace.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <string>
#include <unordered_map>
#include <vector>

int Char2Idx(char c) { return c - 97; }
char Idx2Char(int idx) { return idx + 97; }

std::string WmatToWord(WMAT wmat)
{
    std::string outWord;
    for (int pos = 0; pos < 5; pos++)
    {
        for (int idx = 0; idx < 26; idx++)
        {
            if (wmat[pos][idx])
            {
                outWord += Idx2Char(idx);
                break;
            }
        }
    }
    return outWord;
}

Words ReadWords()
{
    std::vector<WMAT> answers;        // Possible answers
    std::vector<std::string> guesses; // Possible guesses

    std::ifstream guessFile("wordle_guesses.txt");
    for (std::string word; getline(guessFile, word);)
    {
        guesses.push_back(word);
    }

    std::ifstream answerFile("wordle_answers.txt");
    for (std::string word; getline(answerFile, word);)
    {
        answers.push_back(WMAT());
        WMAT& wordData = answers[answers.size() - 1];
        // Zero values
        for (auto& vec : wordData)
        {
            std::fill(vec.begin(), vec.end(), (uint8_t)0);
        }
        for (int pos = 0; pos < 5; pos++)
        {
            int idx = Char2Idx(word[pos]);
            wordData[pos][idx] = 1;
            wordData[5][idx] += 1;
        }
        guesses.push_back(word);
    }

    guessFile.close();
    answerFile.close();

    return Words{ answers, guesses };
}

uint8_t Colours2Int(Colour colours[5])
{
    uint8_t res = 0;
    for (int i = 0; i < 5; i++)
    {
        Colour c = colours[i];
        uint8_t mul;
        switch (c)
        {
        case GREY:
            mul = 0;
            break;
        case YELLOW:
            mul = 1;
            break;
        case GREEN:
            mul = 2;
            break;
        }
        // Calculate 3^i
        uint8_t base = 1;
        for (int _ = 0; _ < i; _++)
        {
            base *= (uint8_t)3;
        }
        res += mul * base;
    }
    return res;
}

void Int2Colours(uint8_t n, Colour colours[5])
{
    for (int i = 0; i < 5; i++)
    {
        uint8_t base = 1;
        for (int _ = 0; _ < i; _++)
        {
            base *= (uint8_t)3;
        }
        auto _colour = n / base % (uint8_t)3;
        Colour c;
        switch (_colour)
        {
        case 0:
            c = GREY;
            break;
        case 1:
            c = YELLOW;
            break;
        case 2:
            c = GREEN;
            break;
        }
        colours[i] = c;
    }
}

void AnswerSpace::Invalidate(int i)
{
    _validMask[i] = false;
    _validCount--;
}

void AnswerSpace::FilterCharCount(char c, uint8_t count, Comparison comp)
{
    if (comp == NEQ)
        throw std::invalid_argument("Cannot use NEQ comparison for FilterCharCount");

    int idx = Char2Idx(c);
    for (int i = 0; i < _answers.size(); i++)
    {
        if (!_validMask[i])
            continue;

        if (comp == EQ)
        {
            if (_answers[i][5][idx] != count)
                Invalidate(i);
        }
        else if (comp == GE)
        {
            if (_answers[i][5][idx] < count)
                Invalidate(i);
        }
    }
}

void AnswerSpace::FilterCharAt(char c, int pos, Comparison comp)
{
    if (comp == GE)
        throw std::invalid_argument("Cannot use GE comparison for FilterCharAt");

    int idx = Char2Idx(c);
    for (int i = 0; i < _answers.size(); i++)
    {
        if (!_validMask[i])
            continue;

        if (comp == EQ)
        {
            // We want to be 1
            if (!_answers[i][pos][idx])
                Invalidate(i);
        }
        else if (comp == NEQ)
        {
            // We want to be 0
            if (_answers[i][pos][idx])
                Invalidate(i);
        }
    }
}

AnswerSpace::AnswerSpace(std::vector<WMAT> answers)
{
    _validCount = answers.size();
    _validMask = std::vector<bool>(answers.size(), true);
    _answers = answers;
}

float AnswerSpace::entropy() { return log2(_validCount); }
int AnswerSpace::size() { return _validCount; }

void AnswerSpace::Reset()
{
    _validCount = _answers.size();
    _validMask = std::vector<bool>(_answers.size(), true);
}

void AnswerSpace::Filter(const char word[5], Colour colours[5])
{
    bool greys[26] = { false };
    std::vector<char> yellowUnique;
    int counts[26] = { 0 };

    for (int pos = 0; pos < 5; pos++)
    {
        char c = word[pos];
        int idx = Char2Idx(c);
        Colour colour = colours[pos];

        if (colour == GREY)
        {
            greys[idx] = true;
        }
        else if (colour == YELLOW)
        {
            counts[idx] += 1;
            // Defer the yellow filter to the next loop, making sure all the greens are done to
            // save compute
        }
        else // colour == GREEN
        {
            counts[idx] += 1;
            // Apply rule: Character must exist at position.
            FilterCharAt(c, pos, EQ);
        }
    }

    // Apply rule: Yellow means character is in wrong spot.
    for (int pos = 0; pos < 5; pos++)
    {
        if (colours[pos] == YELLOW)
        {
            // Character is not at this position
            FilterCharAt(word[pos], pos, NEQ);
        }
    }

    // Apply rule: Frequencies of each character.
    for (int idx = 0; idx < 26; idx++)
    {
        int count = counts[idx];

        char c = Idx2Char(idx);
        if (greys[idx])
        {
            FilterCharCount(c, count, EQ);
        }
        else if (count) // Can't be 0 if not grey, otherwise waste of compute
        {
            FilterCharCount(c, count, GE);
        }
    }
}

void AnswerSpace::CommitFilter()
{
    std::vector<WMAT> newAnswers;
    for (int i = 0; i < _validMask.size(); i++)
    {
        if (_validMask[i])
            newAnswers.push_back(_answers[i]);
    }
    _answers = newAnswers;
    _validMask = std::vector<bool>(_validCount, true);
}

void AnswerSpace::ShowValid()
{
    printf("Valid: [");
    bool first = true;
    for (int i = 0; i < _answers.size(); i++)
    {
        if (_validMask[i])
        {
            if (!first)
                printf(", ");
            first = false;
            printf("%s", WmatToWord(_answers[i]).c_str());
        }
    }
    printf("]\n");
}
