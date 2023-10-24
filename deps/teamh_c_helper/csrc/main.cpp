#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/intrusive_ptr.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <cstdio>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unistd.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <semaphore.h>
#include <cassert>
#include <chrono>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#define CHECK_CUBLAS(call)                                                   \
  do {                                                                       \
    cublasStatus_t status_ = call;                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                  \
      fprintf(stderr, "CUBLAS error (%s:%d): %s, %s\n", __FILE__, __LINE__,  \
              cublasGetStatusName(status_), cublasGetStatusString(status_)); \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

static cublasHandle_t handle;

#define CHECK_CUDA(call)                                              \
  do {                                                                \
    cudaError_t status_ = call;                                       \
    if (status_ != cudaSuccess) {                                     \
      fprintf(stderr, "CUDA error (%s:%d): %s\n", __FILE__, __LINE__, \
              cudaGetErrorString(status_));                           \
      assert(false);                                                  \
    }                                                                 \
  } while (0)

#define CHECK_ERRNO(call) \
  do { \
    int code = call; \
    if (code != 0) { \
      fprintf(stderr, "ERRNO error (%s:%d): %s(%d)\n", __FILE__, __LINE__, strerror(errno), errno); \
      assert(false); \
    } \
  } while (0)

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

const int SHM_SIZE = 1024;
const size_t GPU_BUF_SIZE = 128 * 1024 * 1024;

sem_t* mysem;
sem_t* prevsem;
sem_t* mysem_rev;
sem_t* nextsem_rev;
cudaIpcMemHandle_t* myipchandle;
cudaIpcMemHandle_t* previpchandle;
int rank, world_size;
void* shared_ptr;
void* nextgpubuf;
void* mygpubuf;
cudaStream_t stream;
cudaIpcEventHandle_t* myeventhandle;
cudaIpcEventHandle_t* preveventhandle;
cudaEvent_t myevent;
cudaEvent_t prevevent;
cudaIpcEventHandle_t* myeventhandle_rev;
cudaIpcEventHandle_t* nexteventhandle_rev;
cudaEvent_t myevent_rev;
cudaEvent_t nextevent_rev;
cudaEvent_t tmpevent;

void init(int _rank, int _world_size) {
  rank = _rank;
  world_size = _world_size;

  int fd;
  fd = shm_open("/teamh", O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);
  assert(fd != -1);


  int ret = ftruncate(fd, SHM_SIZE);
  assert(ret != -1);

  shared_ptr = mmap(NULL, SHM_SIZE, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  assert(shared_ptr != MAP_FAILED);

  close(fd);

  // We initialize semaphore for each rank
  //mysem = (sem_t*)shared_ptr + rank;
  //prevsem = mysem - 1;
  //mysem_rev = ((sem_t*)shared_ptr + world_size) + rank;
  //nextsem_rev = mysem_rev + 1;

  char name[32];
  sprintf(name, "/teamh_mysem_%d", rank);
  mysem = sem_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR, 0);
  if (rank > 0) {
    sprintf(name, "/teamh_mysem_%d", rank - 1);
    prevsem = sem_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR, 0);
  }
  sprintf(name, "/teamh_mysem_rev_%d", rank);
  mysem_rev = sem_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR, 0);
  if (rank < world_size - 1) {
    sprintf(name, "/teamh_mysem_rev_%d", rank + 1);
    nextsem_rev = sem_open(name, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR, 0);
  }
  //CHECK_ERRNO(sem_init(mysem, 1, 0));
  //CHECK_ERRNO(sem_init(mysem_rev, 1, 0));

  if (rank < world_size - 1) {
    CHECK_CUDA(cudaSetDevice(rank + 1));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(rank, 0));
    CHECK_CUDA(cudaSetDevice(rank));
    CHECK_CUDA(cudaDeviceEnablePeerAccess(rank + 1, 0));
  }

  CHECK_CUBLAS(cublasCreate(&handle));
}

void init_comm() {
  int gpu_idx;
  CHECK_CUDA(cudaGetDevice(&gpu_idx));
  assert(rank == gpu_idx);

  // last rank has nothing to send
  void *baseaddr;
  baseaddr = (sem_t*)((sem_t*)shared_ptr + world_size) + world_size;
  myipchandle = (cudaIpcMemHandle_t*)baseaddr + rank;
  previpchandle = myipchandle - 1;

  baseaddr = (cudaIpcMemHandle_t*)baseaddr + world_size;
  myeventhandle = (cudaIpcEventHandle_t*)baseaddr + rank;
  preveventhandle = myeventhandle - 1;

  baseaddr = (cudaIpcEventHandle_t*)baseaddr + world_size;
  myeventhandle_rev = (cudaIpcEventHandle_t*)baseaddr + rank;
  nexteventhandle_rev = myeventhandle_rev + 1;

  if (rank < world_size - 1) {
    CHECK_CUDA(cudaSetDevice(rank + 1));
    CHECK_CUDA(cudaMalloc(&nextgpubuf, GPU_BUF_SIZE));
    CHECK_CUDA(cudaSetDevice(rank));
    CHECK_CUDA(cudaIpcGetMemHandle(myipchandle, nextgpubuf));

    CHECK_CUDA(cudaEventCreateWithFlags(&myevent, cudaEventInterprocess | cudaEventDisableTiming));
    CHECK_CUDA(cudaIpcGetEventHandle(myeventhandle, myevent));

    CHECK_ERRNO(sem_post(mysem));
  }

  if (rank > 0) {
    CHECK_ERRNO(sem_wait(prevsem));
    CHECK_CUDA(cudaIpcOpenMemHandle(&mygpubuf, *previpchandle, cudaIpcMemLazyEnablePeerAccess));
    CHECK_CUDA(cudaIpcOpenEventHandle(&prevevent, *preveventhandle));
  }

  if (rank > 0) {
    CHECK_CUDA(cudaEventCreateWithFlags(&myevent_rev, cudaEventInterprocess | cudaEventDisableTiming));
    CHECK_CUDA(cudaIpcGetEventHandle(myeventhandle_rev, myevent_rev));

    CHECK_ERRNO(sem_post(mysem_rev));
  }

  if (rank < world_size - 1) {
    CHECK_ERRNO(sem_wait(nextsem_rev));
    CHECK_CUDA(cudaIpcOpenEventHandle(&nextevent_rev, *nexteventhandle_rev));
  }

  CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  CHECK_CUDA(cudaEventCreate(&tmpevent));
  
}

void send(torch::Tensor& tensor_to_send, bool is_first) {
  assert(tensor_to_send.is_contiguous());
  void* ptr = tensor_to_send.data_ptr();
  // get torch tensor's storage's size
  size_t size = tensor_to_send.storage().nbytes();
  assert(size <= GPU_BUF_SIZE);

  // copy ptr to nextgpubuf using cudaMemcpyPeerAsync
  if (!is_first) {
    CHECK_ERRNO(sem_wait(nextsem_rev));
    CHECK_CUDA(cudaStreamWaitEvent(stream, nextevent_rev));
  }
  CHECK_CUDA(cudaEventRecord(tmpevent, 0));
  CHECK_CUDA(cudaStreamWaitEvent(stream, tmpevent));
  CHECK_CUDA(cudaMemcpyPeerAsync(nextgpubuf, rank + 1, ptr, rank, size, stream));
  //CHECK_CUDA(cudaMemcpyAsync(nextgpubuf, ptr, size, cudaMemcpyDefault, stream));
  c10::cuda::CUDACachingAllocator::recordStream(tensor_to_send.storage().data_ptr(), c10::cuda::getStreamFromExternal(stream, c10::cuda::current_device()));
  CHECK_CUDA(cudaEventRecord(myevent, stream));
  CHECK_ERRNO(sem_post(mysem));
  

  //printf("Rank %d send\n", rank);
  //CHECK_ERRNO(sem_post(mysem));
  //printf("Rank %d send done\n", rank);
}

void recv(torch::Tensor& tensor_to_recv) {
  assert(tensor_to_recv.is_contiguous());
  void* ptr = tensor_to_recv.data_ptr();
  size_t size = tensor_to_recv.storage().nbytes();
  assert(size <= GPU_BUF_SIZE);

  //printf("Rank %d recv\n", rank);
  //CHECK_ERRNO(sem_wait(prevsem));
  //printf("Rank %d recv done\n", rank);

  CHECK_ERRNO(sem_wait(prevsem));
  CHECK_CUDA(cudaStreamWaitEvent(0, prevevent));
  CHECK_CUDA(cudaMemcpyAsync(ptr, mygpubuf, size, cudaMemcpyDeviceToDevice, 0));
  CHECK_CUDA(cudaEventRecord(myevent_rev, 0));
  CHECK_ERRNO(sem_post(mysem_rev));
}

void finalize() {
  CHECK_ERRNO(sem_destroy(mysem));
  CHECK_ERRNO(sem_destroy(mysem_rev));
  munmap(shared_ptr, SHM_SIZE);

  if (rank < world_size - 1) {
    CHECK_CUDA(cudaFree(nextgpubuf));
  }

  if (rank > 0) {
    CHECK_CUDA(cudaIpcCloseMemHandle(mygpubuf));
  }
}

static std::chrono::steady_clock cpu_clock;
typedef std::chrono::time_point<std::chrono::steady_clock> tp;

std::chrono::time_point<std::chrono::steady_clock> get_time() {
  return cpu_clock.now();
}

size_t get_duration_us(
    std::chrono::time_point<std::chrono::steady_clock> start,
    std::chrono::time_point<std::chrono::steady_clock> end
    ) {
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
}

void test(int flag) {
  size_t nbytes = (size_t)10 * 1024 * 1024 * 1024;
  void *da, *db;
  CHECK_CUDA(cudaSetDevice(0));
  if (flag == 0)
    CHECK_CUDA(cudaDeviceEnablePeerAccess(1, 0));
  CHECK_CUDA(cudaMalloc(&da, nbytes));
  CHECK_CUDA(cudaSetDevice(1));
  if (flag == 1)
    CHECK_CUDA(cudaDeviceEnablePeerAccess(0, 0));
  CHECK_CUDA(cudaMalloc(&db, nbytes));

  for (int i = 0; i < 10; ++i) {
    CHECK_CUDA(cudaDeviceSynchronize());
    auto st = get_time();

    CHECK_CUDA(cudaMemcpyPeer(db, 1, da, 0, nbytes));

    CHECK_CUDA(cudaDeviceSynchronize());
    auto et = get_time();
    size_t us = get_duration_us(st, et);
    printf("Throughput: %f GB/s\n", (double)nbytes / us / 1e3);
  }
}

void cublas_nn(torch::Tensor& A, torch::Tensor& B, torch::Tensor& C, int M, int N, int K, float _beta) {
  __half alpha = __float2half(1.0f), beta = __float2half(_beta);
  int lda = K, ldb = N, ldc = N;
  // A = M by K
  // B = N by K
  // C = M by N

  // should do C^T = B^T (transposed) * A^T (normal)

  CHECK_CUBLAS(cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                           (const __half*)B.data_ptr(), ldb,
                           (const __half*)A.data_ptr(), lda, &beta,
                           (__half*)C.data_ptr(), ldc)); 
}

PYBIND11_MODULE(teamh_c_helper, m) {
  m.doc() = "teamh_c_helper";
  m.def("schedule_min_c", &schedule_min_c, "schedule_min_c");
  m.def("schedule_max_c", &schedule_max_c, "schedule_max_c");
  m.def("init", &init, "init");
  m.def("init_comm", &init_comm, "init_comm");
  m.def("send", &send, "send");
  m.def("recv", &recv, "recv");
  m.def("finalize", &finalize, "finalize");
  m.def("test", &test, "test");
  m.def("cublas_nn", &cublas_nn, "cublas_nn");
}