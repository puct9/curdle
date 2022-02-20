#ifndef CURDLE_FILTERWORDS_H_
#define CURDLE_FILTERWORDS_H_

#include "AnswerSpace.h"
#include <fstream>

bool IsFileExist(const char* fileName);

void GenerateLookupTable(Words& words);

#endif
