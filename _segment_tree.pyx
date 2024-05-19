from cython.cimports.libc.stdlib import malloc, free
cimport cython
cimport numpy as np
from numpy cimport intp_t
import numpy as np


cdef struct data:
    int left_link
    int right_link
    int value


cdef struct node:
    node *left
    node *right
    data *arr
    int arr_size


cdef int merge(data *arr_l, int arr_l_size, data *arr_r, int arr_r_size, data **arr):
    cdef int arr_size = arr_l_size + arr_r_size
    arr[0] = <data*>malloc(arr_size * sizeof(data))
    cdef int l = 0, r = 0
    cdef i
    for i in range(arr_size):
        arr[0][i].left_link = l
        arr[0][i].right_link = r
        if l < arr_l_size and r < arr_r_size and arr_l[l].value <= arr_r[r].value or r == arr_r_size:
            arr[0][i].value = arr_l[l].value
            l += 1
        else:
            arr[0][i].value = arr_r[r].value
            r += 1
    return arr_size


cdef void initialize(node *t):
    t.left = NULL
    t.right = NULL
    t.arr = NULL
    t.arr_size = 0


cdef void add(node *t, int tl, int tr, intp_t[:] arr):
    if tr - tl == 1:
        t.arr = <data*>malloc(sizeof(data))
        t.arr[0].value = arr[tl]
        t.arr_size = 1
        return
    cdef int tc = (tl + tr) // 2
    t.left = <node*>malloc(sizeof(node))
    initialize(t.left)
    add(t.left, tl, tc, arr)
    t.right = <node*>malloc(sizeof(node))
    initialize(t.right)
    add(t.right, tc, tr, arr)
    t.arr_size = merge(t.left.arr, t.left.arr_size, t.right.arr, t.right.arr_size, &t.arr)


cdef int count(node *t, int tl, int tr, int l, int r, int p):
    if tl == l and tr == r:
        return r - l - p
    if p == tr - tl:
        return 0
    cdef int result = 0, tc = (tl + tr) // 2
    if t.left and l < tc:
        result += count(t.left, tl, tc, l, min(tc, r), t.arr[p].left_link)
    if t.right and r > tc:
        result += count(t.right, tc, tr, max(tc, l), r, t.arr[p].right_link)
    return result


cdef clean(node *t):
    if t.left:
        clean(t.left)
        free(t.left)
    if t.right:
        clean(t.right)
        free(t.right)
    free(t.arr)


cdef lower_bound(intp_t[:] arr, int n, int x):
    cdef l = 0, r = n - 1, result = 0, c
    while l <= r:
        c = (l + r) // 2
        if arr[c] < x:
            result = c + 1
            l = c + 1
        else:
            r = c - 1
    return result


cdef lower_bound_data(data *arr, int n, int x):
    cdef l = 0, r = n - 1, result = 0, c
    while l <= r:
        c = (l + r) // 2
        if arr[c].value < x:
            result = c + 1
            l = c + 1
        else:
            r = c - 1
    return result


cdef upper_bound(intp_t[:] arr, int n, int x):
    cdef l = 0, r = n - 1, result = 0, c
    while l <= r:
        c = (l + r) // 2
        if arr[c] <= x:
            result = c + 1
            l = c + 1
        else:
            r = c - 1
    return result


@cython.boundscheck(False)
@cython.wraparound(False)
def _pcc_array(intp_t[:] x, intp_t[:] y):
    cdef int n = x.size
    cdef intp_t[::1] res = np.zeros(n, dtype=np.intp)
    cdef int i, p, q
    cdef node *t = <node*>malloc(sizeof(node))
    initialize(t)
    add(t, 0, n, y)
    for i in range(n):
        p = lower_bound(x, n, x[i] + 1)
        if p < n:
            q = lower_bound_data(t.arr, n, y[i])
            res[i] += (n - p - count(t, 0, n, p, n, q))
        p = upper_bound(x, n, x[i] - 1)
        if p > 0:
            q = lower_bound_data(t.arr, n, y[i] + 1)
            res[i] += count(t, 0, n, 0, p, q)
    clean(t)
    free(t)
    return res
