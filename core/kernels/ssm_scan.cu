/*
 * CUDA parallel SSM scan — Blelloch work-efficient prefix scan
 * h_t = a_t * h_{t-1} + b_t  (diagonal SSM recurrence)
 *
 * Associative operator: (a1,b1) ⊕ (a2,b2) = (a1*a2, a1*b2 + b1)
 *
 * Inputs:
 *   a: (B, T, N) float32 — transition coefficients
 *   b: (B, T, N) float32 — input terms
 * Output:
 *   h: (B, T, N) float32 — all hidden states
 *
 * Each (batch, state) pair processed as independent 1-D scan.
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

struct ScanPair {
    float a, b;
};

__device__ inline ScanPair combine(ScanPair x, ScanPair y) {
    /* (a1,b1) ⊕ (a2,b2) = (a2*a1, a2*b1 + b2) — right-to-left composition */
    return {y.a * x.a, y.a * x.b + y.b};
}

/*
 * ssm_scan_kernel
 * Grid: (B, ceil(T/BLOCK_SIZE), N)
 * Each thread handles one (b, t_local, n) element.
 * For sequences longer than BLOCK_SIZE we use a two-pass approach:
 *   Pass 1 — each block scans its tile and writes the block aggregate.
 *   Pass 2 — scan block aggregates (on CPU or small GPU kernel).
 *   Pass 3 — propagate block prefix into each tile.
 * For simplicity this kernel handles sequences up to BLOCK_SIZE * BLOCK_SIZE
 * via shared memory hierarchical scan.
 */
__global__ void ssm_scan_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ h,
    int B, int T, int N
) {
    extern __shared__ ScanPair smem[];  /* BLOCK_SIZE pairs */

    int batch_idx = blockIdx.x;
    int state_idx = blockIdx.z;
    int tid       = threadIdx.x;
    int global_t  = blockIdx.y * BLOCK_SIZE + tid;

    /* Base offset into the (B, T, N) arrays */
    int base = batch_idx * T * N + state_idx;

    /* Load element or identity */
    ScanPair elem;
    if (global_t < T) {
        elem.a = a[batch_idx * T * N + global_t * N + state_idx];
        elem.b = b[batch_idx * T * N + global_t * N + state_idx];
    } else {
        elem.a = 1.0f;
        elem.b = 0.0f;
    }
    smem[tid] = elem;
    __syncthreads();

    /* Blelloch up-sweep (reduce) */
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < BLOCK_SIZE) {
            smem[idx] = combine(smem[idx - stride], smem[idx]);
        }
        __syncthreads();
    }

    /* Clear last element for exclusive scan */
    if (tid == BLOCK_SIZE - 1) {
        smem[tid] = {1.0f, 0.0f};
    }
    __syncthreads();

    /* Blelloch down-sweep */
    for (int stride = BLOCK_SIZE >> 1; stride >= 1; stride >>= 1) {
        int idx = (tid + 1) * (stride << 1) - 1;
        if (idx < BLOCK_SIZE) {
            ScanPair t_val = smem[idx - stride];
            smem[idx - stride] = smem[idx];
            smem[idx] = combine(t_val, smem[idx]);
        }
        __syncthreads();
    }

    /*
     * smem[tid] now holds the exclusive prefix.
     * Apply to get inclusive scan: h_t = combine(prefix, elem).
     * The recurrence h_t = a_t * h_{t-1} + b_t corresponds to inclusive scan
     * starting from h_0 = b_0 (with h_{-1} = 0).
     */
    if (global_t < T) {
        ScanPair prefix = smem[tid];
        ScanPair result = combine(prefix, elem);
        h[batch_idx * T * N + global_t * N + state_idx] = result.b;
    }
}

/*
 * Two-pass version for T > BLOCK_SIZE:
 * After the per-block scans, propagate block-level prefixes.
 */
__global__ void ssm_propagate_kernel(
    float* __restrict__ h,
    const float* __restrict__ block_a,   /* (B, n_blocks, N) prefix a values */
    const float* __restrict__ block_b,   /* (B, n_blocks, N) prefix b values */
    int B, int T, int N, int block_stride
) {
    int batch_idx = blockIdx.x;
    int state_idx = blockIdx.z;
    int global_t  = blockIdx.y * BLOCK_SIZE + threadIdx.x;
    int block_id  = blockIdx.y;

    if (global_t >= T || block_id == 0) return;

    /* The prefix accumulated from all earlier blocks */
    float pa = block_a[batch_idx * block_stride * N + (block_id - 1) * N + state_idx];
    float pb = block_b[batch_idx * block_stride * N + (block_id - 1) * N + state_idx];

    int idx = batch_idx * T * N + global_t * N + state_idx;
    /* combine prefix (pa,pb) with current value h[idx] as b-part */
    h[idx] = pa * h[idx] + pb;
}

/* Host-side launcher */
void ssm_scan_cuda(float* a, float* b, float* h, int B, int T, int N) {
    size_t total = (size_t)B * T * N;
    float *d_a, *d_b, *d_h;

    cudaMalloc(&d_a, total * sizeof(float));
    cudaMalloc(&d_b, total * sizeof(float));
    cudaMalloc(&d_h, total * sizeof(float));

    cudaMemcpy(d_a, a, total * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, total * sizeof(float), cudaMemcpyHostToDevice);

    int n_blocks_t = (T + BLOCK_SIZE - 1) / BLOCK_SIZE;
    dim3 grid(B, n_blocks_t, N);
    dim3 block(BLOCK_SIZE);
    size_t smem_size = BLOCK_SIZE * sizeof(ScanPair);

    if (n_blocks_t == 1) {
        /* Single-pass: fits in one block per (B, N) pair */
        ssm_scan_kernel<<<grid, block, smem_size>>>(d_a, d_b, d_h, B, T, N);
    } else {
        /*
         * Multi-pass: run per-block scans, collect block aggregates,
         * scan aggregates on CPU, then propagate back.
         */
        ssm_scan_kernel<<<grid, block, smem_size>>>(d_a, d_b, d_h, B, T, N);
        cudaDeviceSynchronize();

        /* Collect last element of each block's scan as block aggregate */
        size_t agg_size = (size_t)B * n_blocks_t * N;
        float *h_agg_a = (float*)malloc(agg_size * sizeof(float));
        float *h_agg_b = (float*)malloc(agg_size * sizeof(float));
        float *h_blk   = (float*)malloc(total * sizeof(float));

        cudaMemcpy(h_blk, d_h, total * sizeof(float), cudaMemcpyDeviceToHost);

        /* Extract block tail values from input arrays (on CPU) for aggregate */
        float *h_a = (float*)malloc(total * sizeof(float));
        float *h_b = (float*)malloc(total * sizeof(float));
        cudaMemcpy(h_a, d_a, total * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_b, d_b, total * sizeof(float), cudaMemcpyDeviceToHost);

        /* Compute block-level aggregates: product of a's and resulting b */
        for (int bat = 0; bat < B; bat++) {
            for (int blk = 0; blk < n_blocks_t; blk++) {
                for (int n = 0; n < N; n++) {
                    float agg_a = 1.0f, agg_b = 0.0f;
                    int t_start = blk * BLOCK_SIZE;
                    int t_end   = (t_start + BLOCK_SIZE < T) ? t_start + BLOCK_SIZE : T;
                    for (int t = t_start; t < t_end; t++) {
                        int idx = bat * T * N + t * N + n;
                        float ea = h_a[idx], eb = h_b[idx];
                        agg_b = ea * agg_b + eb;
                        agg_a = agg_a * ea;
                    }
                    int agg_idx = bat * n_blocks_t * N + blk * N + n;
                    h_agg_a[agg_idx] = agg_a;
                    h_agg_b[agg_idx] = agg_b;
                }
            }
        }

        /* Prefix-scan block aggregates on CPU (sequential, n_blocks_t is small) */
        float *prefix_a = (float*)calloc(agg_size, sizeof(float));
        float *prefix_b = (float*)calloc(agg_size, sizeof(float));
        for (int bat = 0; bat < B; bat++) {
            for (int n = 0; n < N; n++) {
                float cum_a = 1.0f, cum_b = 0.0f;
                for (int blk = 0; blk < n_blocks_t; blk++) {
                    int idx = bat * n_blocks_t * N + blk * N + n;
                    /* Exclusive prefix: store before combining */
                    prefix_a[idx] = cum_a;
                    prefix_b[idx] = cum_b;
                    float new_b = h_agg_a[idx] * cum_b + h_agg_b[idx];
                    float new_a = h_agg_a[idx] * cum_a;
                    cum_a = new_a;
                    cum_b = new_b;
                }
            }
        }

        /* Apply prefix to each block's scan result */
        float *d_pa, *d_pb;
        cudaMalloc(&d_pa, agg_size * sizeof(float));
        cudaMalloc(&d_pb, agg_size * sizeof(float));
        cudaMemcpy(d_pa, prefix_a, agg_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_pb, prefix_b, agg_size * sizeof(float), cudaMemcpyHostToDevice);

        ssm_propagate_kernel<<<grid, block>>>(d_h, d_pa, d_pb, B, T, N, n_blocks_t);
        cudaDeviceSynchronize();

        cudaFree(d_pa);
        cudaFree(d_pb);
        free(h_agg_a); free(h_agg_b); free(h_blk);
        free(h_a); free(h_b);
        free(prefix_a); free(prefix_b);
    }

    cudaMemcpy(h, d_h, total * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_h);
}

/* -----------------------------------------------------------------------
 * Python C extension wrapper
 * ----------------------------------------------------------------------- */

static PyObject* py_ssm_scan(PyObject* self, PyObject* args) {
    PyObject *a_obj, *b_obj;
    if (!PyArg_ParseTuple(args, "OO", &a_obj, &b_obj)) {
        return NULL;
    }

    PyArrayObject *a_arr = (PyArrayObject*)PyArray_FROM_OTF(
        a_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *b_arr = (PyArrayObject*)PyArray_FROM_OTF(
        b_obj, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);

    if (!a_arr || !b_arr) {
        Py_XDECREF(a_arr); Py_XDECREF(b_arr);
        PyErr_SetString(PyExc_ValueError, "Inputs must be float32 numpy arrays");
        return NULL;
    }

    if (PyArray_NDIM(a_arr) != 3 || PyArray_NDIM(b_arr) != 3) {
        Py_DECREF(a_arr); Py_DECREF(b_arr);
        PyErr_SetString(PyExc_ValueError, "Inputs must be 3-D arrays (B, T, N)");
        return NULL;
    }

    npy_intp B = PyArray_DIM(a_arr, 0);
    npy_intp T = PyArray_DIM(a_arr, 1);
    npy_intp N = PyArray_DIM(a_arr, 2);

    npy_intp dims[3] = {B, T, N};
    PyArrayObject *h_arr = (PyArrayObject*)PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!h_arr) {
        Py_DECREF(a_arr); Py_DECREF(b_arr);
        return PyErr_NoMemory();
    }

    float *a_ptr = (float*)PyArray_DATA(a_arr);
    float *b_ptr = (float*)PyArray_DATA(b_arr);
    float *h_ptr = (float*)PyArray_DATA(h_arr);

    ssm_scan_cuda(a_ptr, b_ptr, h_ptr, (int)B, (int)T, (int)N);

    Py_DECREF(a_arr);
    Py_DECREF(b_arr);
    return (PyObject*)h_arr;
}

static PyMethodDef SsmMethods[] = {
    {"ssm_scan", py_ssm_scan, METH_VARARGS,
     "ssm_scan(a, b) -> h\nParallel SSM prefix scan on GPU.\n"
     "a, b: (B, T, N) float32.  Returns h: (B, T, N) float32."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ssm_module = {
    PyModuleDef_HEAD_INIT, "ssm_scan", NULL, -1, SsmMethods
};

PyMODINIT_FUNC PyInit_ssm_scan(void) {
    import_array();
    return PyModule_Create(&ssm_module);
}
