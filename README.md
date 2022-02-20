# GPU accelerated Wordle player

## Benchmark

System specs: Ryzen 5 3600, RTX 2060 Super. All tests are run on 1 CPU core.

Benchmark results are ordered by filters per second (filters/s). Finding and counting the resultant possible answers after being given a guess and colours counts as 1 filter. (e.g., what and how many words can be the answer if we guess "raise" and see grey, green, yellow, grey, grey?)

| Implementation | filters/s | Seconds to suggest best word |
| -------------- | --------- | ---------------------------- |
| My friend's    | 1.3k      | Too many |
| Naive (Python/NumPy) | 13k | Loads |
| Naive (C++)    | 170k      | A decent amount |
| Precomputed word-pair colours (Python/NumPy) | 250k | 10-13 |
| Precomputed word-pair colours (C++/CUDA) | 1.4B | 0.002 |

This method achieves the 3146121 filters to find the best word in the greedy expected information algorithm presented in [this video](https://www.youtube.com/watch?v=v68zYyaEmEA) in about 2ms, giving it an effective filtering speed of 1.3B-1.6B filters/s.
