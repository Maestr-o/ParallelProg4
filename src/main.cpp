#include <CL/cl.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

#include <cmath>
#include <ctime>
#include <iomanip>
#include <iostream>

using namespace std;

typedef unsigned long long num;

void run();
void parallel(num n);  // параллельный алгоритм
void seq(num n);       // последовательный алгоритм
cl_program build(
    cl_context ctx, cl_device_id dev,
    const char* f);  // функция для компиляции и запуска ядра OpenCL

int main() {
  run();
  return 0;
}

void run() {
  double start, end;
  num n;
  cout << "Enter N: ";
  cin >> n;

  start = omp_get_wtime();
  seq(n);
  end = omp_get_wtime();
  cout << "Time: " << fixed << setprecision(3) << end - start << " sec" << endl;

  start = omp_get_wtime();
  parallel(n);
  end = omp_get_wtime();
  cout << "Time: " << fixed << setprecision(3) << end - start << " sec" << endl;
}

void seq(num n) {
  num r = 0, a, b, res;
  for (num i = n; i > 0; r++) i /= 10;
  r = pow(10, r / 2);
  a = r;
  b = r;
  num i;
  if (a * b < n + 1)
    i = n + 1;
  else
    i = a * b;

  while (true) {
    res = 0;
    for (num j = i; j > 0; j /= 10) {
      res *= 10;
      res += j % 10;
    }
    if (res == i && res > n) {
      a = r;
      b = r;
      for (a; a * b <= res; a++) {
        for (b; a * b <= res; b++) {
          if (a * b == res && a != b) {
            cout << endl << "Sequential" << endl;
            cout << a << " * " << b << " = " << res << endl;
            return;
          }
        }
        b = a;
      }
    }
    i++;
  }
}

void parallel(num n) {
  num r = 0, a, b;
  for (num i = n; i > 0; r++) i /= 10;
  r = pow(10, r / 2);
  a = r;
  b = r;
  num i;
  if (a * b < n + 1)
    i = n + 1;
  else
    i = a * b;

  cl_platform_id cpPlatform[2];
  cl_device_id device_id;
  cl_context context;
  cl_command_queue queue;
  cl_program program;
  cl_kernel kernel;
  cl_int err;

  size_t threads = 200;
  err = clGetPlatformIDs(2, cpPlatform, NULL);
  err = clGetDeviceIDs(cpPlatform[0], CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
  context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
  queue = clCreateCommandQueue(context, device_id, 0, &err);
  program = build(context, device_id, "kernel.cl");
  kernel = clCreateKernel(program, "krnl", &err);

  cl_mem X = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(num), NULL,
                            NULL);  // создание буфера для передачи данных
  cl_mem Y =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(num), NULL, NULL);
  cl_mem NMin =
      clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(num), NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(num),
                 &i);  // установка аргументов для ядра
  clSetKernelArg(kernel, 1, sizeof(num), &r);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &X);
  clSetKernelArg(kernel, 3, sizeof(cl_mem), &Y);
  clSetKernelArg(kernel, 4, sizeof(cl_mem), &NMin);
  num* x = (num*)malloc(sizeof(num));  // выделение памяти
  num* y = (num*)malloc(sizeof(num));
  num* z = (num*)malloc(sizeof(num));
  *x = 0;
  *y = 0;
  *z = 0;

  clEnqueueWriteBuffer(queue, X, CL_TRUE, 0, sizeof(num), x, 0, NULL,
                       NULL);  // запись данных в буферы OpenCL
  clEnqueueWriteBuffer(queue, Y, CL_TRUE, 0, sizeof(num), y, 0, NULL, NULL);
  clEnqueueWriteBuffer(queue, NMin, CL_TRUE, 0, sizeof(num), z, 0, NULL, NULL);
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &threads, NULL, 0, NULL, NULL);

  clFinish(queue);  // ожидание завершения выполнения ядра

  clEnqueueReadBuffer(
      queue, X, CL_TRUE, 0, sizeof(num), x, 0, NULL,
      NULL);  // чтение данных из буферов OpenCL обратно в хост-память
  clEnqueueReadBuffer(queue, Y, CL_TRUE, 0, sizeof(num), y, 0, NULL, NULL);
  clEnqueueReadBuffer(queue, NMin, CL_TRUE, 0, sizeof(num), z, 0, NULL, NULL);
  cout << endl << "Parallel" << endl;
  cout << *x << " * " << *y << " = " << *z << endl;
  free(x);  // очистка памяти
  free(y);
  free(z);
  clReleaseMemObject(X);  // освобождение буферов, программы, ядра, контекста
  clReleaseMemObject(Y);
  clReleaseMemObject(NMin);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);
}

cl_program build(
    cl_context ctx, cl_device_id dev,
    const char* f) {  // создание программы OpenCL, компиляция, логгирование
  cl_program program;
  FILE* program_handle;
  char *program_buffer, *program_log;
  size_t program_size, log_size;
  int err;

  program_handle = fopen(f, "rb");
  if (program_handle == NULL) {
    perror("Couldn't find the program file");
    exit(1);
  }
  fseek(program_handle, 0, SEEK_END);
  program_size = ftell(program_handle);
  rewind(program_handle);
  program_buffer = (char*)malloc(program_size + 1);
  program_buffer[program_size] = '\0';
  fread(program_buffer, sizeof(char), program_size, program_handle);
  fclose(program_handle);

  program = clCreateProgramWithSource(ctx, 1, (const char**)&program_buffer,
                                      &program_size, &err);
  if (err < 0) {
    perror("Couldn't create the program");
    exit(1);
  }
  free(program_buffer);

  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err < 0) {
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, 0, NULL,
                          &log_size);
    program_log = (char*)malloc(log_size + 1);
    program_log[log_size] = '\0';
    clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG, log_size + 1,
                          program_log, NULL);
    printf("%s\n", program_log);
    free(program_log);
    exit(1);
  }
  return program;
}
