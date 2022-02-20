#ifndef CU_Array2D_H
#define CU_Array2D_H

#include <iostream>
#include <vector>

template <class T>
class Array2D
{
private:
    T* _data;
    int _r;
    int _c;

public:
    Array2D(int r, int c) : _r(r), _c(c) { _data = new T[r * c]; }
    Array2D(const Array2D<T>& other)
    {
        // Copy constructor
        _data = new T[other._r * other._c];
        std::copy_n(other._data, other._r * other._c, _data);
        _r = other._r;
        _c = other._c;
    }

    T* data() { return _data; }
    int rows() { return _r; }
    int cols() { return _c; }
    int numel() { return _r * _c; }
    T Get(int r, int c) { return _data[r * _c + c]; }
    void Set(int r, int c, T v) { _data[r * _c + c] = v; }

    void FilterCols(std::vector<bool>& filter)
    {
        // We need to know how many columns there will be
        int outC = 0;
        for (bool b : filter)
        {
            if (b)
                outC++;
        }
        Array2D<T> out(_r, outC);
        for (int r = 0; r < _r; r++)
        {
            int newC = 0;
            for (int oldC = 0; oldC < _c; oldC++)
            {
                if (filter[oldC])
                    out.Set(r, newC++, Get(r, oldC));
            }
        }

        // Copy new data into ours
        delete[] _data;
        _c = outC;
        _data = new T[_r * _c];
        std::copy_n(out._data, _r * _c, _data);
    }

    ~Array2D() { delete[] _data; }
};

#endif
