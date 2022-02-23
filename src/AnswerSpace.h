#ifndef CU_AnswerSpace_H
#define CU_AnswerSpace_H

#include <array>
#include <math.h>
#include <string>
#include <vector>

// 5+1=6, where 5 is num chars and 1 is sum of occurrence of char
typedef std::array<std::array<uint8_t, 26>, 6> WMAT;

int Char2Idx(char c);
char Idx2Char(int idx);

struct Words
{
    std::vector<WMAT> answers;
    std::vector<std::string> guesses;
};

std::string WmatToWord(WMAT wmat);
Words ReadWords();

enum Comparison
{
    GE,
    EQ,
    NEQ
};

enum Colour
{
    GREY,
    YELLOW,
    GREEN
};

uint8_t Colours2Int(Colour colours[5]);
void Int2Colours(uint8_t n, Colour colours[5]);

class AnswerSpace
{
private:
    int _validCount;
    std::vector<bool> _validMask;
    std::vector<WMAT> _answers;

    void Invalidate(int i);
    void FilterCharCount(char c, uint8_t count, Comparison comp);
    void FilterCharAt(char c, int pos, Comparison comp);

public:
    AnswerSpace(std::vector<WMAT> answers);

    float entropy();
    int size();
    void Reset();
    void Filter(const char word[5], Colour colours[5]);
    void CommitFilter();
    void ShowValid();
    std::vector<bool> GetMask() { return _validMask; }
};

#endif
