#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>

std::tuple<std::vector<int>, std::vector<int>, int> schedule_min_c(std::vector<int> lengths, int thr) {
  int N = lengths.size();
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return lengths[a] < lengths[b]; });
  std::vector<int> D;
  std::vector<int> E;
  int total_area = 0;
  for (int i = 0; i < N; ++i) {
    total_area += lengths[idx[i]];
    int rect_area = lengths[idx[0]] * (i + 1);
    int penalty = total_area - rect_area;
    if (rect_area >= thr) {
      D.push_back(penalty);
      E.push_back(-1);
    } else {
      D.push_back(1 << 30);
      E.push_back(-1);
    }
  }
  for (int i = 0; i < N; ++i) {
    int total_area = 0;
    for (int j = i - 1; j >= 0; --j) {
      total_area += lengths[idx[j + 1]];
      int rect_area = lengths[idx[j + 1]] * (i - j);
      int penalty = total_area - rect_area;
      if (rect_area >= thr && D[i] > D[j] + penalty) {
        D[i] = D[j] + penalty;
        E[i] = j;
      }
    }
  }
  std::vector<int> blocks;
  int i = N - 1;
  if (E[i] == -1) {
    return make_tuple(idx, blocks, D[N - 1]);
  }
  while (i >= 0) {
    blocks.push_back(i - E[i]);
    i = E[i];
  }
  std::reverse(blocks.begin(), blocks.end());
  return make_tuple(idx, blocks, D[N - 1]);
}

std::tuple<std::vector<int>, std::vector<int>, int> schedule_max_c(std::vector<int> lengths, int thr) {
  int N = lengths.size();
  std::vector<int> idx(N);
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return lengths[a] < lengths[b]; });
  std::vector<int> D;
  std::vector<int> E;
  int total_area = 0;
  for (int i = 0; i < N; ++i) {
    total_area += lengths[idx[i]];
    int rect_area = lengths[idx[i]] * (i + 1);
    int penalty = rect_area - total_area;
    if (rect_area >= thr) {
      D.push_back(penalty);
      E.push_back(-1);
    } else {
      D.push_back(1 << 30);
      E.push_back(-1);
    }
  }
  for (int i = 0; i < N; ++i) {
    int total_area = 0;
    for (int j = i - 1; j >= 0; --j) {
      total_area += lengths[idx[j + 1]];
      int rect_area = lengths[idx[i]] * (i - j);
      int penalty = rect_area - total_area;
      if (rect_area >= thr && D[i] > D[j] + penalty) {
        D[i] = D[j] + penalty;
        E[i] = j;
      }
    }
  }
  std::vector<int> blocks;
  int i = N - 1;
  if (E[i] == -1) {
    return make_tuple(idx, blocks, D[N - 1]);
  }
  while (i >= 0) {
    blocks.push_back(i - E[i]);
    i = E[i];
  }
  std::reverse(blocks.begin(), blocks.end());
  return make_tuple(idx, blocks, D[N - 1]);
}

PYBIND11_MODULE(teamh_c_helper, m) {
  m.doc() = "teamh_c_helper";
  m.def("schedule_min_c", &schedule_min_c, "schedule_min_c");
  m.def("schedule_max_c", &schedule_max_c, "schedule_max_c");
}