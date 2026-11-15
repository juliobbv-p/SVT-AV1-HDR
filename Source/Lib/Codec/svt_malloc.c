/*
* Copyright(c) 2019 Intel Corporation
*
* This source code is subject to the terms of the BSD 2 Clause License and
* the Alliance for Open Media Patent License 1.0. If the BSD 2 Clause License
* was not distributed with this source code in the LICENSE file, you can
* obtain it at https://www.aomedia.org/license/software-license. If the Alliance for Open
* Media Patent License 1.0 was not distributed with this source code in the
* PATENTS file, you can obtain it at https://www.aomedia.org/license/patent-license.
*/
#include "svt_malloc.h"

#define LOG_TAG "SvtMalloc"
#include "svt_log.h"

void svt_print_alloc_fail_impl(const char* file, int line) {
    SVT_FATAL("allocate memory failed, at %s:%d\n", file, line);
}

#ifdef __ANDROID__
#include <pthread.h>
#include <string.h>

// Android's allocator (scudo) keeps every size class in its own region and hands out blocks in shuffled order,
// so objects allocated together (a context, its arrays, its descriptors) never share a page. Blocks of up to
// POOL_MAX bytes are instead carved in allocation order from chunks taken from malloc, with one free list per
// size class, which halves the encoder's D-TLB stall cycles on Cortex-A510. Each block starts after a header
// holding its size class (POOL_SYSTEM for blocks that come straight from malloc) and its requested size. The
// chunks go back to malloc when the last pooled block is freed.
#define POOL_MAX (64 << 10)
#define POOL_CHUNK (512 << 10)
#define POOL_CLASSES 106
#define POOL_SYSTEM POOL_CLASSES
#define POOL_FREED (POOL_CLASSES + 1)

typedef struct PoolHdr {
    size_t cls;
    size_t size;
} PoolHdr;

typedef struct PoolChunk {
    struct PoolChunk* next;
    size_t            pad;
} PoolChunk;

static pthread_mutex_t g_pool_mutex = PTHREAD_MUTEX_INITIALIZER;
static PoolHdr*        g_pool_free[POOL_CLASSES];
static PoolChunk*      g_pool_chunks;
static uint8_t*        g_pool_cur;
static size_t          g_pool_left;
static size_t          g_pool_live;

// Size class of a block of total bytes (header included): 16-byte steps up to 1 KB, 256-byte steps up to 8 KB,
// 4 KB steps up to POOL_MAX. Stores the class size in *size.
static size_t pool_class(size_t total, size_t* size) {
    if (total <= 1024) {
        *size = (total + 15) & ~(size_t)15;
        return (*size >> 4) - 1;
    }
    if (total <= 8192) {
        *size = (total + 255) & ~(size_t)255;
        return 59 + (*size >> 8);
    }
    *size = (total + 4095) & ~(size_t)4095;
    return 89 + (*size >> 12);
}

static void* pool_alloc(size_t n) {
    // A free block keeps its free-list link where the data goes.
    size_t       size;
    const size_t cls = pool_class(sizeof(PoolHdr) + (n > sizeof(PoolHdr*) ? n : sizeof(PoolHdr*)), &size);
    pthread_mutex_lock(&g_pool_mutex);
    PoolHdr* h = g_pool_free[cls];
    if (h) {
        g_pool_free[cls] = *(PoolHdr**)(h + 1);
    } else {
        if (g_pool_left < size) {
            PoolChunk* chunk = malloc(POOL_CHUNK);
            if (!chunk) {
                pthread_mutex_unlock(&g_pool_mutex);
                return NULL;
            }
            chunk->next   = g_pool_chunks;
            g_pool_chunks = chunk;
            g_pool_cur    = (uint8_t*)(chunk + 1);
            g_pool_left   = POOL_CHUNK - sizeof(PoolChunk);
        }
        h = (PoolHdr*)g_pool_cur;
        g_pool_cur += size;
        g_pool_left -= size;
    }
    g_pool_live++;
    pthread_mutex_unlock(&g_pool_mutex);
    h->cls  = cls;
    h->size = n;
    return h + 1;
}

static void* system_alloc(PoolHdr* h, size_t n) {
    if (!h) {
        return NULL;
    }
    h->cls  = POOL_SYSTEM;
    h->size = n;
    return h + 1;
}

void* svt_malloc(size_t size) {
    if (size > POOL_MAX - sizeof(PoolHdr)) {
        return size > SIZE_MAX - sizeof(PoolHdr) ? NULL : system_alloc(malloc(size + sizeof(PoolHdr)), size);
    }
    return pool_alloc(size);
}

void* svt_calloc(size_t count, size_t size) {
    if (size && count > (SIZE_MAX - sizeof(PoolHdr)) / size) {
        return NULL;
    }
    const size_t n = count * size;
    if (n > POOL_MAX - sizeof(PoolHdr)) {
        return system_alloc(calloc(1, n + sizeof(PoolHdr)), n);
    }
    void* p = pool_alloc(n);
    if (p) {
        memset(p, 0, n);
    }
    return p;
}

void svt_free(void* ptr) {
    if (!ptr) {
        return;
    }
    PoolHdr*     h   = (PoolHdr*)ptr - 1;
    const size_t cls = h->cls;
    if (cls == POOL_SYSTEM) {
        free(h);
        return;
    }
    // Not from these functions, or freed twice: continuing would corrupt the free lists.
    if (cls >= POOL_CLASSES) {
        abort();
    }
    h->cls = POOL_FREED;
    pthread_mutex_lock(&g_pool_mutex);
    *(PoolHdr**)ptr  = g_pool_free[cls];
    g_pool_free[cls] = h;
    if (!--g_pool_live) {
        while (g_pool_chunks) {
            PoolChunk* next = g_pool_chunks->next;
            free(g_pool_chunks);
            g_pool_chunks = next;
        }
        memset(g_pool_free, 0, sizeof(g_pool_free));
        g_pool_cur  = NULL;
        g_pool_left = 0;
    }
    pthread_mutex_unlock(&g_pool_mutex);
}

void* svt_realloc(void* ptr, size_t size) {
    if (!ptr) {
        return svt_malloc(size);
    }
    PoolHdr* h = (PoolHdr*)ptr - 1;
    if (h->cls == POOL_SYSTEM && size > POOL_MAX - sizeof(PoolHdr)) {
        return size > SIZE_MAX - sizeof(PoolHdr) ? NULL : system_alloc(realloc(h, size + sizeof(PoolHdr)), size);
    }
    void* p = svt_malloc(size);
    if (p) {
        memcpy(p, ptr, size < h->size ? size : h->size);
        svt_free(ptr);
    }
    return p;
}
#endif

#ifdef DEBUG_MEMORY_USAGE
#include <stdint.h>
#include <limits.h>
#include "definitions.h"
#include "svt_threads.h"

static EbHandle g_malloc_mutex;

static void malloc_mutex_cleanup(void) {
    svt_destroy_mutex(g_malloc_mutex);
}

static ONCE_ROUTINE(create_malloc_mutex) {
    g_malloc_mutex = svt_create_mutex();
    atexit(malloc_mutex_cleanup);
    ONCE_ROUTINE_EPILOG;
}

DEFINE_ONCE(g_malloc_once);

static EbHandle get_malloc_mutex() {
    svt_run_once(&g_malloc_once, create_malloc_mutex);
    return g_malloc_mutex;
}

// Simple hash function to speed up entry search
// Takes the top half and bottom half of the pointer and adds them together.
static inline uint32_t hash(const void* const p) {
    const uintptr_t bit_mask = ((uintptr_t)1 << sizeof(bit_mask) / 2 * CHAR_BIT) - 1, v = (uintptr_t)p;
    return (uint32_t)((v >> (sizeof(v) / 2 * CHAR_BIT)) + (v & bit_mask));
}

typedef struct MemoryEntry {
    void*       ptr;
    size_t      count;
    const char* file;
    EbPtrType   type;
    uint32_t    line;
} MemoryEntry;

//+1 to get a better hash result
#define MEM_ENTRY_SIZE (4 * 1024 * 1024 + 1)

static MemoryEntry g_mem_entry[MEM_ENTRY_SIZE];

#define TO_INDEX(v) ((v) % MEM_ENTRY_SIZE)
static bool g_add_mem_entry_warning    = true;
static bool g_remove_mem_entry_warning = true;

/*********************************************************************************
*
* @brief
*  compare and update current memory entry.
*
* @param[in] e
*  current memory entry.
*
* @param[in] param
*  param you set to for_each_mem_entry
*
*
* @returns  return true if you want get early exit in for_each_mem_entry
*
s*
********************************************************************************/

typedef bool (*Predicate)(MemoryEntry* e, void* param);

/*********************************************************************************
*
* @brief
*  Loop through mem entries.
*
* @param[in] bucket
*  the hash bucket
*
* @param[in] start
*  loop start position
*
* @param[in] pred
*  return true if you want early exit
*
* @param[out] param
*  param send to pred.
*
* @returns  return true if we got early exit.
*
*
********************************************************************************/
static bool for_each_hash_entry(MemoryEntry* bucket, uint32_t start, Predicate pred, void* param) {
    const uint32_t s = TO_INDEX(start);
    uint32_t       i = s;

    do {
        MemoryEntry* e = bucket + i;
        if (pred(e, param)) {
            return true;
        }
        i++;
        i = TO_INDEX(i);
    } while (i != s);
    return false;
}

static bool for_each_mem_entry(uint32_t start, Predicate pred, void* param) {
    bool     ret;
    EbHandle m = get_malloc_mutex();
    svt_block_on_mutex(m);
    ret = for_each_hash_entry(g_mem_entry, start, pred, param);
    svt_release_mutex(m);
    return ret;
}

static const char* mem_type_name(EbPtrType type) {
    static const char* name[EB_PTR_TYPE_TOTAL] = {
        "malloced memory", "calloced memory", "aligned memory", "mutex", "semaphore", "thread"};
    return name[type];
}

static bool add_mem_entry(MemoryEntry* e, void* param) {
    if (!e->ptr) {
        *e = *(MemoryEntry*)param;
        return true;
    }
    return false;
}

static bool remove_mem_entry(MemoryEntry* e, void* param) {
    MemoryEntry* item = param;
    if (e->ptr == item->ptr) {
        // The second case is a special case, we use EB_FREE to free calloced memory
        if (e->type == item->type || (e->type == EB_C_PTR && item->type == EB_N_PTR)) {
            e->ptr = NULL;
            return true;
        }
    }
    return false;
}

typedef struct MemSummary {
    size_t   amount[EB_PTR_TYPE_TOTAL];
    uint32_t occupied;
} MemSummary;

static bool count_mem_entry(MemoryEntry* e, void* param) {
    if (e->ptr) {
        MemSummary* sum = param;
        sum->amount[e->type] += e->count;
        sum->occupied++;
    }
    return false;
}

static inline void get_memory_usage_and_scale(size_t amount, double* const usage, char* const scale) {
    static const char scales[] = {' ', 'K', 'M', 'G', 'T'};
    size_t            i        = 1;
    for (; i < sizeof(scales) && amount >= (size_t)1 << (++i * 10);)
        ;
    *scale = scales[--i];
    *usage = (double)amount / (double)((size_t)1 << (i * 10));
}

//this need more memory and cpu
#define PROFILE_MEMORY_USAGE
#ifdef PROFILE_MEMORY_USAGE

//if we use a static array here, this size + sizeof(g_mem_entry) will exceed max size allowed on windows.
static MemoryEntry* g_profile_entry;

static bool add_location(MemoryEntry* e, void* param) {
    MemoryEntry* new_item = param;
    if (!e->ptr) {
        *e = *new_item;
        return true;
    }
    if (e->file == new_item->file && e->line == new_item->line) {
        e->count += new_item->count;
        return true;
    }
    // to next position.
    return false;
}

static bool collect_mem(MemoryEntry* e, void* param) {
    EbPtrType* type = param;
    if (e->ptr && e->type == *type) {
        for_each_hash_entry(g_profile_entry, 0, add_location, e);
    }
    //Loop entire bucket.
    return false;
}

static int compare_count(const void* a, const void* b) {
    const MemoryEntry* pa = a;
    const MemoryEntry* pb = b;
    return pb->count < pa->count ? -1 : pb->count != pa->count;
}

static void print_top_10_locations() {
    EbHandle  m    = get_malloc_mutex();
    EbPtrType type = EB_N_PTR;
    svt_block_on_mutex(m);
    g_profile_entry = calloc(MEM_ENTRY_SIZE, sizeof(*g_profile_entry));
    if (!g_profile_entry) {
        SVT_ERROR("not enough memory for memory profile");
        svt_release_mutex(m);
        return;
    }

    for_each_hash_entry(g_mem_entry, 0, collect_mem, &type);
    qsort(g_profile_entry, MEM_ENTRY_SIZE, sizeof(*g_profile_entry), compare_count);

    SVT_INFO("top 10 %s locations:\n", mem_type_name(type));
    for (int i = 0; i < 10; i++) {
        double       usage;
        char         scale;
        MemoryEntry* e = g_profile_entry + i;
        get_memory_usage_and_scale(e->count, &usage, &scale);
        SVT_INFO("(%.2lf %cB): %s:%d\n", usage, scale, e->file, e->line);
    }
    free(g_profile_entry);
    svt_release_mutex(m);
}
#endif //PROFILE_MEMORY_USAGE

static int g_component_count;

static bool print_leak(MemoryEntry* e, void* param) {
    if (e->ptr) {
        bool* leaked = param;
        *leaked      = true;
        SVT_ERROR("%s leaked at %s:%d\n", mem_type_name(e->type), e->file, e->line);
    }
    //loop through all items
    return false;
}

void svt_print_memory_usage() {
    MemSummary sum;
    double     usage;
    char       scale;
    memset(&sum, 0, sizeof(sum));

    for_each_mem_entry(0, count_mem_entry, &sum);
    SVT_INFO("SVT Memory Usage:\n");
    get_memory_usage_and_scale(sum.amount[EB_N_PTR] + sum.amount[EB_C_PTR] + sum.amount[EB_A_PTR], &usage, &scale);
    SVT_INFO("    total allocated memory:       %.2lf %cB\n", usage, scale);
    get_memory_usage_and_scale(sum.amount[EB_N_PTR], &usage, &scale);
    SVT_INFO("        malloced memory:          %.2lf %cB\n", usage, scale);
    get_memory_usage_and_scale(sum.amount[EB_C_PTR], &usage, &scale);
    SVT_INFO("        callocated memory:        %.2lf %cB\n", usage, scale);
    get_memory_usage_and_scale(sum.amount[EB_A_PTR], &usage, &scale);
    SVT_INFO("        allocated aligned memory: %.2lf %cB\n", usage, scale);

    SVT_INFO("    mutex count: %zu\n", sum.amount[EB_MUTEX]);
    SVT_INFO("    semaphore count: %zu\n", sum.amount[EB_SEMAPHORE]);
    SVT_INFO("    thread count: %zu\n", sum.amount[EB_THREAD]);
    SVT_INFO("    hash table fulless: %f, hash bucket is %s\n",
             (double)sum.occupied / MEM_ENTRY_SIZE,
             (double)sum.occupied / MEM_ENTRY_SIZE < .3 ? "healthy" : "too full");
#ifdef PROFILE_MEMORY_USAGE
    print_top_10_locations();
#endif
}

void svt_increase_component_count() {
    EbHandle m = get_malloc_mutex();
    svt_block_on_mutex(m);
    g_component_count++;
    svt_release_mutex(m);
}

void svt_decrease_component_count() {
    EbHandle m = get_malloc_mutex();
    svt_block_on_mutex(m);
    g_component_count--;
    if (!g_component_count) {
        bool leaked = false;
        for_each_hash_entry(g_mem_entry, 0, print_leak, &leaked);
        if (!leaked) {
            SVT_INFO("you have no memory leak\n");
        }
    }
    svt_release_mutex(m);
}

void svt_add_mem_entry_impl(void* ptr, EbPtrType type, size_t count, const char* file, uint32_t line) {
    if (for_each_mem_entry(hash(ptr),
                           add_mem_entry,
                           &(MemoryEntry){.ptr = ptr, .type = type, .count = count, .file = file, .line = line})) {
        return;
    }
    if (g_add_mem_entry_warning) {
        SVT_ERROR(
            "can't add memory entry.\n"
            "You have memory leak or you need increase MEM_ENTRY_SIZE\n");
        g_add_mem_entry_warning = false;
    }
}

void svt_remove_mem_entry(void* ptr, EbPtrType type) {
    if (!ptr) {
        return;
    }
    if (for_each_mem_entry(hash(ptr), remove_mem_entry, &(MemoryEntry){.ptr = ptr, .type = type})) {
        return;
    }
    if (g_remove_mem_entry_warning) {
        SVT_ERROR("something wrong. you freed a unallocated memory %p, type = %s\n", ptr, mem_type_name(type));
        g_remove_mem_entry_warning = false;
    }
}
#endif
