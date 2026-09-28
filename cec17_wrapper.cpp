// Thin C wrapper around the CEC 2017 organizers' reference code (Awad et al., NTU).
// The organizer source file "cec17_test_func.cpp" is NOT redistributed here; setup_cec2017.sh fetches it.
#include <math.h>
#include <unistd.h>
#include "cec17_test_func.cpp"
double *OShift,*M,*y,*z,*x_bound;
int ini_flag=0,n_flag,func_flag,*SS;
extern "C" {
  int cec17_chdir(const char* dir){ return chdir(dir); }
  // Evaluate n points (row-major n x nx) of function func_num (1..30). Results in f[0..n-1].
  void cec17_eval_batch(double* X, double* f, int n, int nx, int func_num){
    for(int i=0;i<n;i++){ double v; cec17_test_func(X+(size_t)i*nx,&v,nx,1,func_num); f[i]=v; }
  }
}
