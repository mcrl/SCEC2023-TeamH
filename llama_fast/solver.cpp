#include <cstdio>
#include <vector>
#include <algorithm>

void dfs(std::vector<std::vector<int>>& enc) {
  std::stable_sort(enc.begin(), enc.end());

  std::vector<std::vector<int>> score(enc.size());
  std::vector<std::vector<bool>> eq(enc.size());
  for (int i = 0; i < enc.size(); ++i) {
    score[i].resize(enc[i].size());
    eq[i].resize(enc[i].size());
  }

  for (int i = 0; i < enc.size(); ++i) {
    for (int j = 0; j < enc[i].size(); ++j) {
      if (i > 0 && enc[i - 1].size() > j) {
        eq[i][j] = enc[i - 1][j] == enc[i][j];
      } else {
        eq[i][j] = false;
      }
      if (j > 0) {
        eq[i][j] = eq[i][j] & eq[i][j - 1];
      }
    }
  }

  for (int i = 0; i < enc.size(); ++i) {
    for (int j = 0; j < enc[i].size(); ++j) {
      if (i == 0) {
        score[i][j] = 0;
      } else {
        if (enc[i - 1].size() <= j || score[i - 1][j] == -1) {
          score[i][j] = -1;
          continue;
        }
        score[i][j] = eq[i][j] ? score[i - 1][j] + j : score[i - 1][j];
      }
    }
  }

  const int bs_upperbound = 4096;
  const int bs_lowerbound = 4096;
  int max_score = -1, max_b, max_s = -1;

  for (int i = 0; i < enc.size(); ++i) {
    for (int j = 0; j < enc[i].size(); ++j) {
      if (max_score < score[i][j] && max_s <= j && i * j < bs_upperbound) {
        max_score = score[i][j];
        max_b = i;
        max_s = j;
      }
    }
  }
  printf("%d %d %d\n", max_b, max_s, max_score);
  if (max_score == -1) {
    return;
  }

  std::vector<std::vector<int>> new_enc(max_b);
  for (int i = 0; i < max_b; ++i) {
    new_enc[i].insert(new_enc[i].end(), enc[i].begin() + max_s,  enc[i].end());
  }
  dfs(new_enc);
}

int trie(std::vector<std::vector<int>> cur_enc) {
  int count = 1;
  for (auto it = cur_enc.begin(); it != cur_enc.end(); ) {
    if (it->size() == 0) {
      it = cur_enc.erase(it);
    } else {
      ++it;
    }
  }
  while (cur_enc.size() > 0) {
    int cur_v = cur_enc[0][0];
    std::vector<std::vector<int>> new_enc, child_enc;
    for (int i = 0; i < cur_enc.size(); ++i) {
      if (cur_enc[i][0] == cur_v) {
        child_enc.push_back(std::vector<int>(cur_enc[i].begin() + 1, cur_enc[i].end()));
      } else {
        new_enc.push_back(cur_enc[i]);
      }
    }
    cur_enc = new_enc;
    count += trie(child_enc);
  }
  return count;
}

int main() {
  FILE *fin = fopen("sorted_enc.txt", "r");
  int n, max_m = 0;
  fscanf(fin, "%d", &n);
  std::vector<std::vector<int>> enc(n);
  for (int i = 0; i < n; ++i) {
    int m;
    fscanf(fin, "%d", &m);
    max_m = std::max(m, max_m);
    for (int j = 0; j < m; ++j) {
      int x;
      fscanf(fin, "%d", &x);
      enc[i].push_back(x);
    }
  }

  //dfs(enc);
  int count = trie(enc);
  printf("%d\n", count);


  return 0;
}