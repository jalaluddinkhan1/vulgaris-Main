/*
 * CUDA kernel for batched learnable wavelet convolution
 *
 * Input:   x       (B, C, T)  float32
 * Filters: filters (K, filter_len) float32  — Morlet-style bank
 * Dilations: (S,)  int32 — dilation rate per scale
 * Output:  out     (B, K*S, T) float32 — multi-scale wavelet coefficients
 *
 * Each thread computes one output element: (b, k*S + s, t).
 * Dilated 1-D convolution: out[b, k*S+s, t] = Σ_c Σ_f x[b, c, t - f*dilation[s]] * filter[k, f]
 * with zero-padding for out-of-bounds.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

/*
 * wavelet_conv_kernel
 *
 * Grid:  (B, K*S, ceil(T/TILE_T))
 * Block: (TILE_T,)
 */
#define TILE_T 128

__global__ void wavelet_conv_kernel(
    const float* __restrict__ input,    /* (B, C, T) */
    const float* __restrict__ filters,  /* (K, filter_len) */
    float* __restrict__ output,         /* (B, K*S, T) */
    int B, int C, int T,
    int K, int S, int filter_len,
    const int* __restrict__ dilations   /* (S,) */
) {
    int b       = blockIdx.x;
    int ks_idx  = blockIdx.y;   /* combined scale-filter index in [0, K*S) */
    int t_base  = blockIdx.z * TILE_T;
    int t_local = threadIdx.x;
    int t       = t_base + t_local;

    if (b >= B || ks_idx >= K * S || t >= T) return;

    int k   = ks_idx / S;
    int s   = ks_idx % S;
    int dil = dilations[s];

    /* Accumulate over input channels and filter taps */
    float acc = 0.0f;
    for (int c = 0; c < C; c++) {
        const float* x_row = input + b * C * T + c * T;
        const float* filt  = filters + k * filter_len;
        for (int f = 0; f < filter_len; f++) {
            int src = t - f * dil;
            float x_val = (src >= 0 && src < T) ? x_row[src] : 0.0f;
            acc += x_val * filt[f];
        }
    }

    output[b * (K * S) * T + ks_idx * T + t] = acc;
}

/* Host launcher */
void wavelet_conv_cuda(
    float* input, float* filters, float* output,
    int B, int C, int T, int K, int S, int filter_len,
    int* dilations
) {
    float *d_in, *d_filt, *d_out;
    int   *d_dil;

    size_t in_sz    = (size_t)B * C * T * sizeof(float);
    size_t filt_sz  = (size_t)K * filter_len * sizeof(float);
    size_t out_sz   = (size_t)B * K * S * T * sizeof(float);
    size_t dil_sz   = (size_t)S * sizeof(int);

    cudaMalloc(&d_in,   in_sz);
    cudaMalloc(&d_filt, filt_sz);
    cudaMalloc(&d_out,  out_sz);
    cudaMalloc(&d_dil,  dil_sz);

    cudaMemcpy(d_in,   input,    in_sz,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_filt, filters,  filt_sz, cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0,         out_sz);
    cudaMemcpy(d_dil,  dilations, dil_sz, cudaMemcpyHostToDevice);

    int n_tiles_t = (T + TILE_T - 1) / TILE_T;
    dim3 grid(B, K * S, n_tiles_t);
    dim3 block(TILE_T);

    wavelet_conv_kernel<<<grid, block>>>(
        d_in, d_filt, d_out,
        B, C, T, K, S, filter_len, d_dil
    );
    cudaDeviceSynchronize();

    cudaMemcpy(output, d_out, out_sz, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_filt);
    cudaFree(d_out);
    cudaFree(d_dil);
}

/* -----------------------------------------------------------------------
 * Python C extension wrapper
 * ----------------------------------------------------------------------- */

static PyObject* py_wavelet_conv(PyObject* self, PyObject* args) {
    PyObject *x_obj, *filt_obj, *dil_obj;
    if (!PyArg_ParseTuple(args, "OOO", &x_obj, &filt_obj, &dil_obj)) {
        return NULL;
    }

    PyArrayObject *x_arr = (PyArrayObject*)PyArray_FROM_OTF(
        x_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *f_arr = (PyArrayObject*)PyArray_FROM_OTF(
        filt_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *d_arr = (PyArrayObject*)PyArray_FROM_OTF(
        dil_obj, NPY_INT32, NPY_ARRAY_IN_ARRAY);

    if (!x_arr || !f_arr || !d_arr) {
        Py_XDECREF(x_arr); Py_XDECREF(f_arr); Py_XDECREF(d_arr);
        PyErr_SetString(PyExc_ValueError, "Invalid input arrays");
        return NULL;
    }

    if (PyArray_NDIM(x_arr) != 3) {
        Py_DECREF(x_arr); Py_DECREF(f_arr); Py_DECREF(d_arr);
        PyErr_SetString(PyExc_ValueError, "input must be 3-D (B, C, T)");
        return NULL;
    }
    if (PyArray_NDIM(f_arr) != 2) {
        Py_DECREF(x_arr); Py_DECREF(f_arr); Py_DECREF(d_arr);
        PyErr_SetString(PyExc_ValueError, "filters must be 2-D (K, filter_len)");
        return NULL;
    }

    int B  = (int)PyArray_DIM(x_arr, 0);
    int C  = (int)PyArray_DIM(x_arr, 1);
    int T  = (int)PyArray_DIM(x_arr, 2);
    int K  = (int)PyArray_DIM(f_arr, 0);
    int FL = (int)PyArray_DIM(f_arr, 1);
    int S  = (int)PyArray_SIZE(d_arr);

    npy_intp out_dims[3] = {B, K * S, T};
    PyArrayObject *out_arr = (PyArrayObject*)PyArray_SimpleNew(3, out_dims, NPY_FLOAT32);
    if (!out_arr) {
        Py_DECREF(x_arr); Py_DECREF(f_arr); Py_DECREF(d_arr);
        return PyErr_NoMemory();
    }

    wavelet_conv_cuda(
        (float*)PyArray_DATA(x_arr),
        (float*)PyArray_DATA(f_arr),
        (float*)PyArray_DATA(out_arr),
        B, C, T, K, S, FL,
        (int*)PyArray_DATA(d_arr)
    );

    Py_DECREF(x_arr);
    Py_DECREF(f_arr);
    Py_DECREF(d_arr);
    return (PyObject*)out_arr;
}

static PyMethodDef WaveletMethods[] = {
    {"wavelet_conv", py_wavelet_conv, METH_VARARGS,
     "wavelet_conv(x, filters, dilations) -> out\n"
     "Batched dilated wavelet convolution.\n"
     "x: (B,C,T) float32, filters: (K,FL) float32, dilations: (S,) int32.\n"
     "Returns out: (B, K*S, T) float32."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef wavelet_module = {
    PyModuleDef_HEAD_INIT, "wavelet_conv", NULL, -1, WaveletMethods
};

PyMODINIT_FUNC PyInit_wavelet_conv(void) {
    import_array();
    return PyModule_Create(&wavelet_module);
}
