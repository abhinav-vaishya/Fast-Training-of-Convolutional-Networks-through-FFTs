#include <bits/stdc++.h>
#include<stdio.h>
#include <chrono> 
using namespace std::chrono;
using namespace std;

typedef complex<float> base;
int n,m,k;
template <typename T>
ostream &operator<<(ostream &o, vector<T> v)
{
    if (v.size() > 0)
        o << v[0];
    for (unsigned i = 1; i < v.size(); i++)
        o << " " << v[i];
    return o << endl;
}
__global__ void inplace_divide_invert(float *A_r,float *A_i, int n, int threads)
{
    int i = blockIdx.x * threads + threadIdx.x;
    if (i < n)
    {
        A_r[i] /= n;
        A_i[i] /= n;
    }
}
__global__ void bitrev_reorder(float *__restrict__ r_r,float *__restrict__ r_i, float *__restrict__ d_r, float *__restrict__ d_i, int s, size_t nthr, int n)
{
    int id = blockIdx.x * nthr + threadIdx.x;
    if (id < n and __brev(id) >> (32 - s) < n)
    {
        r_r[__brev(id) >> (32 - s)] = d_r[id];
        r_i[__brev(id) >> (32 - s)] = d_i[id];

    }
}
__device__ void inplace_fft_inner(float *__restrict__ A_r, float *__restrict__ A_i, int i, int j, int len, int n, bool invert)
{
    if (i + j + len / 2 < n and j < len / 2)
    {
        float u_r, v_r;
        float u_i, v_i;

        float angle = (2 * 3.14 * j) / (len * (invert ? 1.0 : -1.0));
        v_r = cos(angle);
        v_i = sin(angle);
        u_r = A_r[i + j];
        u_i = A_i[i + j];
        float temp_vr = v_r,temp_vi = v_i;
        v_r = A_r[i + j + len / 2]*temp_vr - A_i[i + j + len / 2]*temp_vi;
        v_i = A_i[i + j + len / 2]*temp_vr + A_r[i + j + len / 2]*temp_vi;
        A_r[i + j] = u_r + v_r;
        A_i[i + j] = u_i + v_i;
        A_r[i + j + len / 2] = u_r - v_r;
        A_i[i + j + len / 2] = u_i - v_i;
    }
}
__global__ void inplace_fft(float *__restrict__ A_r, float *__restrict__ A_i, int i, int len, int n, int threads, bool invert)
{
    int j = blockIdx.x * threads + threadIdx.x;
    inplace_fft_inner(A_r, A_i, i, j, len, n, invert);
    
}
__global__ void inplace_fft_outer(float *__restrict__ A_r, float *__restrict__ A_i, int len, int n, int threads, bool invert)
{
    int i = (blockIdx.x * threads + threadIdx.x)*len;
    for (int j = 0; j < len / 2; j++)
    {
        inplace_fft_inner(A_r, A_i, i, j, len, n, invert);
    }
}

void fft(vector<base> &a, bool invert, int balance = 10, int threads = 32)
{
    // Creating array from vector
    int n = (int)a.size();
    int data_size = n * sizeof(float);
    float *data_array_r = (float *)malloc(data_size);
    float *data_array_i = (float *)malloc(data_size);
    for (int i = 0; i < n; i++)
    {
        data_array_r[i] = a[i].real();
        data_array_i[i] = a[i].imag();
    }
    
    // Copying data to GPU
    float *A_r, *dn_r;
    float *A_i, *dn_i;
    cudaMalloc((void **)&A_r, data_size);
    cudaMalloc((void **)&A_i, data_size);
    cudaMalloc((void **)&dn_r, data_size);
    cudaMalloc((void **)&dn_i, data_size);
    cudaMemcpy(dn_r, data_array_r, data_size, cudaMemcpyHostToDevice);
    cudaMemcpy(dn_i, data_array_i, data_size, cudaMemcpyHostToDevice);
    // Bit reversal reordering
    int s = log2(n);

    
    bitrev_reorder<<<ceil(float(n) / threads), threads>>>(A_r,A_i, dn_r, dn_i, s, threads, n);
    
    float *result_r;
    float *result_i;
    result_r = (float *)malloc(data_size);
    result_i = (float *)malloc(data_size);

    
    
    // Synchronize
    cudaDeviceSynchronize();
    // Iterative FFT with loop parallelism balancing
    for (int len = 2; len <= n; len <<= 1)
    {
        if (n / len > balance)
        {
            inplace_fft_outer<<<ceil((float)n / threads / len), threads>>>(A_r,A_i, len, n, threads, invert);
        }
        else
        {
            for (int i = 0; i < n; i += len)
            {
                float repeats = len / 2;
                inplace_fft<<<ceil(repeats / threads), threads>>>(A_r,A_i, i, len, n, threads, invert);
                cudaMemcpy(result_r, A_r, data_size, cudaMemcpyDeviceToHost);
                cudaMemcpy(result_i, A_i, data_size, cudaMemcpyDeviceToHost);
            }
        }
    }
    
    
    if (invert)
        inplace_divide_invert<<<ceil(n * 1.00 / threads), threads>>>(A_r,A_i, n, threads);

    cudaMemcpy(result_r, A_r, data_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(result_i, A_i, data_size, cudaMemcpyDeviceToHost);
    
    // Saving data to vector<complex> in input.
    for (int i = 0; i < n; i++)
    {
        a[i] = base(result_r[i], result_i[i]);
    }
    // Free the memory blocks
    free(data_array_r);
    free(data_array_i);
    cudaFree(A_r);
    cudaFree(A_i);
    cudaFree(dn_r);
    cudaFree(dn_i);
    return;
}

/**
* Performs 2D FFT 
* takes vector of complex vectors, invert and verbose as argument
* performs inplace FFT transform on input vector
*/
void fft2D(vector<vector<base>> &a, bool invert, int balance, int threads)
{
    auto matrix = a;
    for (auto i = 0; i < matrix.size(); i++)
    {
        fft(matrix[i], invert, balance, threads);
    }

    a = matrix;
    
    
    matrix.resize(a[0].size());
    for (int i = 0; i < matrix.size(); i++)
        matrix[i].resize(a.size());
    
        // Transposing matrix
    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            matrix[j][i] = a[i][j];
        }
    }

    for (auto i = 0; i < matrix.size(); i++)
        fft(matrix[i], invert, balance, threads);
    

    for (int i = 0; i < a.size(); i++)
    {
        for (int j = 0; j < a[0].size(); j++)
        {
            a[j][i] = matrix[i][j];
        }
    }
    
}

#define N 100000
#define BALANCE 1024

int nextPowerOf2(int n)  
{  
    unsigned count = 0;  
    if (n && !(n & (n - 1)))  
        return n;  
      
    while( n != 0)  
    {  
        n >>= 1;  
        count += 1;  
    }  
    return 1 << count;  
}  

int main(int argc,char** argv)
{
    // cout<<n<<","<<k<<","<<time<<endl;
    n = atoi(argv[1]);
    k = atoi(argv[2]);
    int b_sz = atoi(argv[3]);
    m = n;
    int old_n = n;
    int old_k = k;
    // int old_m = m;
    int l = n+k-1;
    int new_n = nextPowerOf2(l);
    int new_m = nextPowerOf2(l);
    vector<vector<int>> image(new_n, vector<int>(new_m));
    vector<vector<int>> kernel(new_n, vector<int>(new_m));
    for (int i = 0; i < new_n; ++i)
    {
        for (int j = 0; j < new_m; ++j)
        {
            if(i < k && j < k)kernel[i][j] = i+j+2;
            else kernel[i][j] = 0;
            if(i < n && j < m)image[i][j] = (i+1)*(j+1);
            else image[i][j] = 0;
        }
    }
    n = new_n;
    m = new_m;
    
    vector<vector<base>> complex_image(image.size(), vector<base>(image[0].size()));
    vector<vector<base>> complex_kernel(image.size(), vector<base>(image[0].size()));
    vector<vector<base>> complex_out(image.size(), vector<base>(image[0].size()));
    for (auto i = 0; i < image.size(); i++)
        for (auto j = 0; j < image[0].size(); j++)
            complex_image[i][j] = image[i][j];
    for (auto i = 0; i < kernel.size(); i++)
        for (auto j = 0; j < kernel[0].size(); j++)
            complex_kernel[i][j] = kernel[i][j];
    
    auto t1 = std::chrono::high_resolution_clock::now();
    fft2D(complex_kernel, false, BALANCE, 10);
    for(int b=1;b<=b_sz;b++)
    {
        fft2D(complex_image, false, BALANCE, 10);
        for(auto i=0;i<n;i++)
            for(auto j=0;j<m;j++)
                complex_out[i][j] = complex_image[i][j]*complex_kernel[i][j];
        fft2D(complex_out, true, BALANCE, 10);
        
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    float t = chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
    cout<<old_n<<","<<old_k<<","<<b_sz<<","<<t/1000<<endl;
    // for(int i=0;i<n;i++)
    // {
    //     for(int j=0;j<m;j++)
    //     {
    //         cout<<image[i][j]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
    // cout<<endl;

    // for(int i=0;i<n;i++)
    // {
    //     for(int j=0;j<m;j++)
    //     {
    //         cout<<kernel[i][j]<<" ";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
    // cout<<endl;
    
    // for(auto i=0;i<n;i++)
    // {
    //     for(auto j=0;j<m;j++)
    //     {
        //         cout<<real(complex_image[i][j])<<"\t";
        //     }
        //     cout<<endl;
        // }
        // cout<<endl;
        // cout<<endl;
            
    // for(auto i=0;i<n;i++)
    // {
    //     for(auto j=0;j<m;j++)
    //     {
    //         cout<<real(complex_kernel[i][j])<<"\t";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
    // cout<<endl;

    // for(auto i=0;i<n;i++)
    // {
    //     for(auto j=0;j<m;j++)
    //     {
    //         cout<<real(complex_out[i][j])<<"\t";
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;
    // cout<<endl;

    // vector<vector<base>> final(old_n+old_k-1, vector<base>(old_m+old_k-1));
    // for(auto i=0;i<complex_out.size();i++)
    // {
    //     for(auto j=0;j<complex_out[0].size();j++)
    //     {
    //         cout<<real(complex_out[i][j])<<"\t";
    //         // final[i][j] = complex_out[i][j];
    //     }
    //     cout<<endl;
    // }
    // cout<<endl;

    return 0;
}