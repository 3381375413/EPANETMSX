/******************************************************************************
**  MODULE:        MSXSEGMENT_PROFILE.C
**  DESCRIPTION:   Read-only segment topology profiler for Hybrid Boundary-Core
**                 feasibility validation.
**
**  The profiler is disabled unless MSX_SEGMENT_PROFILE=1 is present.  It
**  observes the segment topology at quality-step boundaries and records
**  mutation notifications from the existing transport code.  It never writes
**  segment state or participates in the physical solve.
******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#include "msxsegment_profile.h"
#include "msxtypes.h"
#include "msxsegment_storage.h"

extern MSXproject MSX;

#define PROFILE_G_COUNT 6

static const int ProfileG[PROFILE_G_COUNT] = {1, 2, 4, 8, 16, 32};

typedef struct
{
    Pseg ptr;
    unsigned long long generation;
} SegmentToken;

typedef struct
{
    Pseg ptr;
    unsigned long long generation;
    int link;
    unsigned int stamp;
    unsigned char pre_mask;
    unsigned char post_mask;
} ShadowHashSlot;

typedef struct
{
    Pseg ptr;
    unsigned long long generation;
} GenerationEntry;

typedef struct
{
    unsigned long long *bins;
    size_t capacity;
    long long samples;
    unsigned long long maximum;
} BurstHistogram;

typedef struct
{
    long long react_visits;
    long long down_complete;
    long long down_partial;
    long long up_new;
    long long up_merge;
    long long reversals;
    long long boundary_ops;
    int max_segments;
} ProfileLink;

typedef struct
{
    int enabled;
    int opened;
    int nlinks;
    FILE *steps;
    FILE *links;

    int *pre_count;
    int *post_count;
    size_t *pre_offset;
    size_t *post_offset;
    SegmentToken *pre_tokens;
    SegmentToken *post_tokens;
    size_t pre_total;
    size_t post_total;
    size_t pre_token_capacity;
    size_t post_token_capacity;

    long long *step_visits;
    long long *step_down_complete;
    long long *step_down_partial;
    long long *step_up_new;
    long long *step_up_merge;
    long long *step_reversals;
    ProfileLink *link;

    GenerationEntry *generations;
    size_t generation_count;
    size_t generation_capacity;

    ShadowHashSlot *shadow_hash;
    size_t shadow_hash_capacity;
    unsigned int shadow_stamp;
    size_t *shadow_indices;
    size_t shadow_index_count;
    size_t shadow_index_capacity;
    int shadow_memory_ok;

    long long total_steps;
    long long total_react_visits;
    long long pre_core_segments[PROFILE_G_COUNT];
    long long core_segments[PROFILE_G_COUNT];
    double core_updates[PROFILE_G_COUNT];
    long long handoffs[PROFILE_G_COUNT];
    long long boundary_to_core[PROFILE_G_COUNT];
    long long core_to_boundary[PROFILE_G_COUNT];
    double coverage_sum[PROFILE_G_COUNT];
    long long coverage_steps[PROFILE_G_COUNT];
    long long step_pre_core_segments[PROFILE_G_COUNT];
    long long step_post_core_segments[PROFILE_G_COUNT];
    long long step_boundary_to_core[PROFILE_G_COUNT];
    long long step_core_to_boundary[PROFILE_G_COUNT];
    long long step_handoffs[PROFILE_G_COUNT];
    double step_core_updates[PROFILE_G_COUNT];
    BurstHistogram downstream_delete_burst;
    BurstHistogram upstream_create_burst;
    BurstHistogram combined_burst;
    double step_sim_time_sec;
} ProfileState;

static ProfileState P = {0};

static int profile_requested(void)
{
    const char *value = getenv("MSX_SEGMENT_PROFILE");
    return value && (strcmp(value, "1") == 0 ||
                     strcmp(value, "YES") == 0 ||
                     strcmp(value, "yes") == 0 ||
                     strcmp(value, "TRUE") == 0 ||
                     strcmp(value, "true") == 0);
}

static size_t next_power_of_two(size_t value)
{
    size_t result = 1;
    while (result < value && result <= SIZE_MAX / 2) result <<= 1;
    return result;
}

static size_t hash_pointer_generation(Pseg ptr, unsigned long long generation,
                                       int link)
{
    uintptr_t x = (uintptr_t)ptr;
    x ^= (uintptr_t)generation + (uintptr_t)0x9e3779b97f4a7c15ULL +
         (x << 6) + (x >> 2);
    x ^= (uintptr_t)(unsigned int)link * (uintptr_t)0x632be59bd9b4e019ULL;
    x ^= x >> 30;
    x *= (uintptr_t)0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= (uintptr_t)0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (size_t)x;
}

static size_t hash_pointer(Pseg ptr)
{
    uintptr_t x = (uintptr_t)ptr;
    x ^= x >> 30;
    x *= (uintptr_t)0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= (uintptr_t)0x94d049bb133111ebULL;
    x ^= x >> 31;
    return (size_t)x;
}

static int generation_rehash(size_t requested)
{
    size_t i, capacity = next_power_of_two(requested < 1024 ? 1024 : requested);
    GenerationEntry *old = P.generations;
    size_t old_capacity = P.generation_capacity;
    GenerationEntry *new_entries = (GenerationEntry *)calloc(capacity,
                                                               sizeof(GenerationEntry));
    if (!new_entries) return 0;
    P.generations = new_entries;
    P.generation_capacity = capacity;
    P.generation_count = 0;
    if (old)
    {
        for (i = 0; i < old_capacity; i++)
        {
            size_t slot;
            if (!old[i].ptr) continue;
            slot = hash_pointer(old[i].ptr) & (capacity - 1);
            while (new_entries[slot].ptr)
                slot = (slot + 1) & (capacity - 1);
            new_entries[slot] = old[i];
            P.generation_count++;
        }
        free(old);
    }
    return 1;
}

static GenerationEntry *generation_find(Pseg ptr, int create)
{
    size_t slot;
    if (!ptr) return NULL;
    if (!P.generation_capacity && !generation_rehash(1024)) return NULL;
    if (create && (P.generation_count + 1) * 10 >= P.generation_capacity * 7)
    {
        if (!generation_rehash(P.generation_capacity * 2)) return NULL;
    }
    slot = hash_pointer(ptr) & (P.generation_capacity - 1);
    while (P.generations[slot].ptr)
    {
        if (P.generations[slot].ptr == ptr) return &P.generations[slot];
        slot = (slot + 1) & (P.generation_capacity - 1);
    }
    if (!create) return NULL;
    P.generations[slot].ptr = ptr;
    P.generations[slot].generation = 0;
    P.generation_count++;
    return &P.generations[slot];
}

static unsigned long long generation_for(Pseg ptr)
{
    GenerationEntry *entry = generation_find(ptr, 0);
    if (ptr && !entry) P.shadow_memory_ok = 0;
    return entry ? entry->generation : 0;
}

void MSXsegProfile_segmentAllocated(Pseg seg)
{
    GenerationEntry *entry;
    if (!MSXsegProfile_enabled() || !seg) return;
    entry = generation_find(seg, 1);
    if (!entry)
    {
        P.shadow_memory_ok = 0;
        return;
    }
    if (entry->generation == ULLONG_MAX)
        entry->generation = 1;
    else
        entry->generation++;
}

static int reserve_tokens(SegmentToken **tokens, size_t *capacity, size_t needed)
{
    size_t next;
    SegmentToken *new_tokens;
    if (needed <= *capacity) return 1;
    next = *capacity ? *capacity : 1024;
    while (next < needed && next <= SIZE_MAX / 2) next <<= 1;
    if (next < needed) return 0;
    new_tokens = (SegmentToken *)realloc(*tokens, next * sizeof(SegmentToken));
    if (!new_tokens) return 0;
    *tokens = new_tokens;
    *capacity = next;
    return 1;
}

static int append_token(SegmentToken **tokens, size_t *count, size_t *capacity,
                        Pseg seg)
{
    if (!seg) return 1;
    if (!reserve_tokens(tokens, capacity, *count + 1)) return 0;
    (*tokens)[*count].ptr = seg;
    (*tokens)[*count].generation = generation_for(seg);
    (*count)++;
    return 1;
}

static int snapshot(int *count, size_t *offset, size_t *total,
                    SegmentToken **storage, size_t *capacity)
{
    int k;
    size_t start;
    Pseg seg;
    int expected, pos;
    *total = 0;
    for (k = 1; k <= P.nlinks; k++)
    {
        offset[k] = *total;
        start = *total;
        if (MSXsegStorage_isPipeRingLink(k))
        {
            expected = MSXsegStorage_pipeCount(k);
            for (pos = 0; pos < expected; pos++)
            {
                seg = MSXsegStorage_pipeSegFromHead(k, pos);
                if (!append_token(storage, total, capacity, seg)) return 0;
            }
        }
        else
        {
            expected = MSX.Link[k].nsegs;
            seg = expected > 0 ? MSX.FirstSeg[k] : NULL;
            pos = 0;
            while (seg && pos < expected)
            {
                if (!append_token(storage, total, capacity, seg)) return 0;
                seg = seg->prev;
                pos++;
            }
        }
        count[k] = (int)(*total - start);
    }
    return 1;
}

static int core_count(int n, int g)
{
    int core = n - 2 * g;
    return core > 0 ? core : 0;
}

static unsigned char core_mask(size_t index, int count)
{
    int g;
    unsigned char mask = 0;
    for (g = 0; g < PROFILE_G_COUNT; g++)
        if (index >= (size_t)ProfileG[g] &&
            index + (size_t)ProfileG[g] < (size_t)count)
            mask = (unsigned char)(mask | (unsigned char)(1u << g));
    return mask;
}

static void free_histogram(BurstHistogram *hist)
{
    free(hist->bins);
    memset(hist, 0, sizeof(*hist));
}

static int histogram_reserve(BurstHistogram *hist, size_t needed)
{
    size_t next;
    unsigned long long *bins;
    if (needed <= hist->capacity) return 1;
    next = hist->capacity ? hist->capacity : 64;
    while (next < needed && next <= SIZE_MAX / 2) next <<= 1;
    if (next < needed) return 0;
    bins = (unsigned long long *)realloc(hist->bins,
                                         next * sizeof(unsigned long long));
    if (!bins) return 0;
    memset(bins + hist->capacity, 0,
           (next - hist->capacity) * sizeof(unsigned long long));
    hist->bins = bins;
    hist->capacity = next;
    return 1;
}

static void histogram_add(BurstHistogram *hist, unsigned long long value)
{
    if (!histogram_reserve(hist, (size_t)value + 1))
    {
        P.shadow_memory_ok = 0;
        return;
    }
    hist->bins[value]++;
    hist->samples++;
    if (value > hist->maximum) hist->maximum = value;
}

static unsigned long long histogram_p99(const BurstHistogram *hist)
{
    long long rank, cumulative = 0;
    size_t i;
    if (hist->samples <= 0) return 0;
    rank = (99 * hist->samples + 99) / 100;
    for (i = 0; i <= (size_t)hist->maximum; i++)
    {
        cumulative += (long long)hist->bins[i];
        if (cumulative >= rank) return (unsigned long long)i;
    }
    return hist->maximum;
}

static size_t shadow_hash_size(size_t expected)
{
    return expected > SIZE_MAX / 4 ? SIZE_MAX : expected * 4 + 1;
}

static int shadow_rehash(size_t requested)
{
    size_t capacity = next_power_of_two(requested < 1024 ? 1024 : requested);
    ShadowHashSlot *new_slots;
    free(P.shadow_hash);
    new_slots = (ShadowHashSlot *)calloc(capacity, sizeof(ShadowHashSlot));
    if (!new_slots)
    {
        P.shadow_hash = NULL;
        P.shadow_hash_capacity = 0;
        return 0;
    }
    P.shadow_hash = new_slots;
    P.shadow_hash_capacity = capacity;
    P.shadow_stamp = 1;
    return 1;
}

static int shadow_prepare(size_t expected)
{
    size_t needed = shadow_hash_size(expected);
    if (!P.shadow_hash_capacity || P.shadow_hash_capacity < needed)
        return shadow_rehash(needed);
    P.shadow_stamp++;
    if (P.shadow_stamp == 0)
    {
        memset(P.shadow_hash, 0,
               P.shadow_hash_capacity * sizeof(ShadowHashSlot));
        P.shadow_stamp = 1;
    }
    return 1;
}

static size_t shadow_find(Pseg ptr, unsigned long long generation, int link,
                          int create)
{
    size_t slot;
    if (!P.shadow_hash || !P.shadow_hash_capacity || !ptr) return SIZE_MAX;
    slot = hash_pointer_generation(ptr, generation, link) &
           (P.shadow_hash_capacity - 1);
    while (P.shadow_hash[slot].stamp == P.shadow_stamp)
    {
        if (P.shadow_hash[slot].ptr == ptr &&
            P.shadow_hash[slot].generation == generation &&
            P.shadow_hash[slot].link == link)
            return slot;
        slot = (slot + 1) & (P.shadow_hash_capacity - 1);
    }
    if (!create) return SIZE_MAX;
    P.shadow_hash[slot].ptr = ptr;
    P.shadow_hash[slot].generation = generation;
    P.shadow_hash[slot].link = link;
    P.shadow_hash[slot].stamp = P.shadow_stamp;
    P.shadow_hash[slot].pre_mask = 0;
    P.shadow_hash[slot].post_mask = 0;
    if (P.shadow_index_count >= P.shadow_index_capacity)
    {
        size_t next = P.shadow_index_capacity ? P.shadow_index_capacity * 2 : 1024;
        size_t *new_indices = (size_t *)realloc(P.shadow_indices,
                                                next * sizeof(size_t));
        if (!new_indices)
        {
            P.shadow_memory_ok = 0;
            return SIZE_MAX;
        }
        P.shadow_indices = new_indices;
        P.shadow_index_capacity = next;
    }
    P.shadow_indices[P.shadow_index_count++] = slot;
    return slot;
}

static void build_exact_shadow(void)
{
    int k, g;
    size_t i, slot;
    unsigned char mask;
    if (!P.shadow_memory_ok) return;
    if (!shadow_prepare(P.pre_total + P.post_total))
    {
        P.shadow_memory_ok = 0;
        return;
    }
    P.shadow_index_count = 0;
    for (k = 1; k <= P.nlinks; k++)
    {
        for (i = 0; i < (size_t)P.pre_count[k]; i++)
        {
            mask = core_mask(i, P.pre_count[k]);
            if (!mask) continue;
            slot = shadow_find(P.pre_tokens[P.pre_offset[k] + i].ptr,
                               P.pre_tokens[P.pre_offset[k] + i].generation,
                               k, 1);
            if (slot != SIZE_MAX) P.shadow_hash[slot].pre_mask |= mask;
        }
    }
    for (k = 1; k <= P.nlinks; k++)
    {
        for (i = 0; i < (size_t)P.post_count[k]; i++)
        {
            mask = core_mask(i, P.post_count[k]);
            if (!mask) continue;
            slot = shadow_find(P.post_tokens[P.post_offset[k] + i].ptr,
                               P.post_tokens[P.post_offset[k] + i].generation,
                               k, 1);
            if (slot != SIZE_MAX) P.shadow_hash[slot].post_mask |= mask;
        }
    }
    for (i = 0; i < P.shadow_index_count; i++)
    {
        unsigned char entered, exited;
        ShadowHashSlot *entry = &P.shadow_hash[P.shadow_indices[i]];
        entered = (unsigned char)(entry->post_mask &
                                  (unsigned char)~entry->pre_mask);
        exited = (unsigned char)(entry->pre_mask &
                                 (unsigned char)~entry->post_mask);
        for (g = 0; g < PROFILE_G_COUNT; g++)
        {
            unsigned char bit = (unsigned char)(1u << g);
            if (entered & bit) P.step_boundary_to_core[g]++;
            if (exited & bit) P.step_core_to_boundary[g]++;
        }
    }
}

static void free_arrays(void)
{
    free(P.pre_count);
    free(P.post_count);
    free(P.pre_offset);
    free(P.post_offset);
    free(P.pre_tokens);
    free(P.post_tokens);
    free(P.step_visits);
    free(P.step_down_complete);
    free(P.step_down_partial);
    free(P.step_up_new);
    free(P.step_up_merge);
    free(P.step_reversals);
    free(P.link);
    free(P.generations);
    free(P.shadow_hash);
    free(P.shadow_indices);
    free_histogram(&P.downstream_delete_burst);
    free_histogram(&P.upstream_create_burst);
    free_histogram(&P.combined_burst);
    memset(&P, 0, sizeof(P));
}

static void write_steps_header(void)
{
    int g;
    fprintf(P.steps,
            "step_index,sim_time_sec,active_segments,active_links,max_segments_per_pipe,"
            "react_segment_visits,downstream_complete_deletes,downstream_partial_consumes,"
            "upstream_new_segments,upstream_merges,flow_reversals,boundary_operations,"
            "boundary_mutation_ratio,max_per_link_downstream_delete_burst,"
            "max_per_link_upstream_create_burst,max_per_link_combined_boundary_burst");
    for (g = 0; g < PROFILE_G_COUNT; g++)
        fprintf(P.steps, ",pre_core_segments_g%d,core_segments_g%d,core_coverage_g%d,"
                "boundary_to_core_g%d,core_to_boundary_g%d,handoff_g%d,core_segment_updates_g%d",
                ProfileG[g], ProfileG[g], ProfileG[g], ProfileG[g], ProfileG[g],
                ProfileG[g], ProfileG[g]);
    fputc('\n', P.steps);
}

static void write_links_header(void)
{
    fprintf(P.links,
            "link_index,total_react_segment_visits,max_segments,"
            "downstream_complete_deletes,downstream_partial_consumes,"
            "upstream_new_segments,upstream_merges,flow_reversals,boundary_operations\n");
}

int MSXsegProfile_open(void)
{
    size_t n;
    if (P.opened) return 0;
    memset(&P, 0, sizeof(P));
    if (!profile_requested()) return 0;
    P.enabled = 1;
    P.shadow_memory_ok = 1;
    P.nlinks = MSX.Nobjects[LINK];
    n = (size_t)P.nlinks + 1;
    P.pre_count = (int *)calloc(n, sizeof(int));
    P.post_count = (int *)calloc(n, sizeof(int));
    P.pre_offset = (size_t *)calloc(n, sizeof(size_t));
    P.post_offset = (size_t *)calloc(n, sizeof(size_t));
    P.step_visits = (long long *)calloc(n, sizeof(long long));
    P.step_down_complete = (long long *)calloc(n, sizeof(long long));
    P.step_down_partial = (long long *)calloc(n, sizeof(long long));
    P.step_up_new = (long long *)calloc(n, sizeof(long long));
    P.step_up_merge = (long long *)calloc(n, sizeof(long long));
    P.step_reversals = (long long *)calloc(n, sizeof(long long));
    P.link = (ProfileLink *)calloc(n, sizeof(ProfileLink));
    if (!P.pre_count || !P.post_count || !P.pre_offset || !P.post_offset ||
        !P.step_visits || !P.step_down_complete || !P.step_down_partial ||
        !P.step_up_new || !P.step_up_merge || !P.step_reversals || !P.link)
    {
        free_arrays();
        /* Profiling is observational; an optional allocation failure must
           never turn a successful physical run into an error. */
        return 0;
    }
    P.steps = fopen("msx_segment_profile_steps.csv", "wt");
    P.links = fopen("msx_segment_profile_links.csv", "wt");
    if (!P.steps || !P.links)
    {
        if (P.steps) fclose(P.steps);
        if (P.links) fclose(P.links);
        free_arrays();
        return 0;
    }
    write_steps_header();
    write_links_header();
    P.opened = 1;
    return 0;
}

void MSXsegProfile_reset(void)
{
    size_t n;
    if (!P.enabled || !P.opened) return;
    n = ((size_t)P.nlinks + 1) * sizeof(int);
    memset(P.pre_count, 0, n);
    memset(P.post_count, 0, n);
    n = ((size_t)P.nlinks + 1) * sizeof(size_t);
    memset(P.pre_offset, 0, n);
    memset(P.post_offset, 0, n);
    n = ((size_t)P.nlinks + 1) * sizeof(long long);
    memset(P.step_visits, 0, n);
    memset(P.step_down_complete, 0, n);
    memset(P.step_down_partial, 0, n);
    memset(P.step_up_new, 0, n);
    memset(P.step_up_merge, 0, n);
    memset(P.step_reversals, 0, n);
    memset(P.link, 0, ((size_t)P.nlinks + 1) * sizeof(ProfileLink));
    if (P.generations && P.generation_capacity)
        memset(P.generations, 0, P.generation_capacity * sizeof(GenerationEntry));
    P.generation_count = 0;
    P.total_steps = 0;
    P.total_react_visits = 0;
    memset(P.pre_core_segments, 0, sizeof(P.pre_core_segments));
    memset(P.core_segments, 0, sizeof(P.core_segments));
    memset(P.core_updates, 0, sizeof(P.core_updates));
    memset(P.handoffs, 0, sizeof(P.handoffs));
    memset(P.boundary_to_core, 0, sizeof(P.boundary_to_core));
    memset(P.core_to_boundary, 0, sizeof(P.core_to_boundary));
    memset(P.coverage_sum, 0, sizeof(P.coverage_sum));
    memset(P.coverage_steps, 0, sizeof(P.coverage_steps));
    free_histogram(&P.downstream_delete_burst);
    free_histogram(&P.upstream_create_burst);
    free_histogram(&P.combined_burst);
    P.shadow_memory_ok = 1;
}

int MSXsegProfile_enabled(void)
{
    return P.enabled && P.opened;
}

void MSXsegProfile_beginStep(double sim_time_sec)
{
    int k, g;
    if (!MSXsegProfile_enabled()) return;
    for (k = 1; k <= P.nlinks; k++)
    {
        P.step_visits[k] = 0;
        P.step_down_complete[k] = 0;
        P.step_down_partial[k] = 0;
        P.step_up_new[k] = 0;
        P.step_up_merge[k] = 0;
        P.step_reversals[k] = 0;
    }
    P.step_sim_time_sec = sim_time_sec;
    if (!snapshot(P.pre_count, P.pre_offset, &P.pre_total,
                  &P.pre_tokens, &P.pre_token_capacity))
        P.shadow_memory_ok = 0;
    for (g = 0; g < PROFILE_G_COUNT; g++)
    {
        P.step_pre_core_segments[g] = 0;
        P.step_boundary_to_core[g] = 0;
        P.step_core_to_boundary[g] = 0;
        P.step_handoffs[g] = 0;
        P.step_core_updates[g] = 0.0;
    }
    for (k = 1; k <= P.nlinks; k++)
    {
        size_t i;
        for (i = 0; i < (size_t)P.pre_count[k]; i++)
        {
            unsigned char mask = core_mask(i, P.pre_count[k]);
            for (g = 0; g < PROFILE_G_COUNT; g++)
                if (mask & (unsigned char)(1u << g))
                    P.step_pre_core_segments[g]++;
        }
    }
}

void MSXsegProfile_reactVisitsForLink(int k, int count)
{
    if (!MSXsegProfile_enabled() || k <= 0 || k > P.nlinks || count <= 0) return;
#ifdef _OPENMP
#pragma omp atomic
#endif
    P.step_visits[k] += count;
}

void MSXsegProfile_event(int k, int event_kind)
{
    if (!MSXsegProfile_enabled() || k <= 0 || k > P.nlinks) return;
    switch (event_kind)
    {
    case MSX_PROFILE_DOWNSTREAM_COMPLETE_DELETE:
#ifdef _OPENMP
#pragma omp atomic
#endif
        P.step_down_complete[k]++;
        break;
    case MSX_PROFILE_DOWNSTREAM_PARTIAL_CONSUME:
#ifdef _OPENMP
#pragma omp atomic
#endif
        P.step_down_partial[k]++;
        break;
    case MSX_PROFILE_UPSTREAM_NEW_SEGMENT:
#ifdef _OPENMP
#pragma omp atomic
#endif
        P.step_up_new[k]++;
        break;
    case MSX_PROFILE_UPSTREAM_MERGE:
#ifdef _OPENMP
#pragma omp atomic
#endif
        P.step_up_merge[k]++;
        break;
    case MSX_PROFILE_FLOW_REVERSAL:
#ifdef _OPENMP
#pragma omp atomic
#endif
        P.step_reversals[k]++;
        break;
    default:
        break;
    }
}

void MSXsegProfile_endStep(double sim_time_sec)
{
    int k, g;
    int active_total = 0, active_links = 0, max_segments = 0;
    long long visits = 0, down_complete = 0, down_partial = 0;
    long long up_new = 0, up_merge = 0, reversals = 0, boundary_ops = 0;
    unsigned long long max_delete_burst = 0, max_create_burst = 0;
    unsigned long long max_combined_burst = 0;
    if (!MSXsegProfile_enabled()) return;
    if (!snapshot(P.post_count, P.post_offset, &P.post_total,
                  &P.post_tokens, &P.post_token_capacity))
        P.shadow_memory_ok = 0;
    if (P.shadow_memory_ok) build_exact_shadow();

    for (k = 1; k <= P.nlinks; k++)
    {
        unsigned long long delete_burst = (unsigned long long)P.step_down_complete[k];
        unsigned long long create_burst = (unsigned long long)P.step_up_new[k];
        unsigned long long combined = (unsigned long long)(P.step_down_complete[k] +
            P.step_down_partial[k] + P.step_up_new[k] + P.step_up_merge[k] +
            P.step_reversals[k]);
        int active = P.post_count[k];
        if (active > 0) active_links++;
        active_total += active;
        if (active > max_segments) max_segments = active;
        visits += P.step_visits[k];
        down_complete += P.step_down_complete[k];
        down_partial += P.step_down_partial[k];
        up_new += P.step_up_new[k];
        up_merge += P.step_up_merge[k];
        reversals += P.step_reversals[k];
        boundary_ops += (long long)combined;
        histogram_add(&P.downstream_delete_burst, delete_burst);
        histogram_add(&P.upstream_create_burst, create_burst);
        histogram_add(&P.combined_burst, combined);
        if (delete_burst > max_delete_burst) max_delete_burst = delete_burst;
        if (create_burst > max_create_burst) max_create_burst = create_burst;
        if (combined > max_combined_burst) max_combined_burst = combined;

        P.link[k].react_visits += P.step_visits[k];
        P.link[k].down_complete += P.step_down_complete[k];
        P.link[k].down_partial += P.step_down_partial[k];
        P.link[k].up_new += P.step_up_new[k];
        P.link[k].up_merge += P.step_up_merge[k];
        P.link[k].reversals += P.step_reversals[k];
        P.link[k].boundary_ops += (long long)combined;
        if (active > P.link[k].max_segments) P.link[k].max_segments = active;
        for (g = 0; g < PROFILE_G_COUNT; g++)
        {
            int pre_core = core_count(P.pre_count[k], ProfileG[g]);
            double core_visit = P.pre_count[k] > 0
                ? (double)P.step_visits[k] * (double)pre_core /
                  (double)P.pre_count[k] : 0.0;
            P.step_core_updates[g] += core_visit;
            P.core_updates[g] += core_visit;
            P.core_segments[g] += core_count(P.post_count[k], ProfileG[g]);
        }
    }
    for (g = 0; g < PROFILE_G_COUNT; g++)
    {
        double coverage = visits > 0 ? P.step_core_updates[g] / (double)visits : 0.0;
        P.step_post_core_segments[g] = 0;
        for (k = 1; k <= P.nlinks; k++)
            P.step_post_core_segments[g] += core_count(P.post_count[k], ProfileG[g]);
        P.pre_core_segments[g] += P.step_pre_core_segments[g];
        P.coverage_sum[g] += coverage;
        P.coverage_steps[g]++;
        P.boundary_to_core[g] += P.step_boundary_to_core[g];
        P.core_to_boundary[g] += P.step_core_to_boundary[g];
        P.handoffs[g] += P.step_boundary_to_core[g] + P.step_core_to_boundary[g];
        P.step_handoffs[g] = P.step_boundary_to_core[g] + P.step_core_to_boundary[g];
    }
    P.total_steps++;
    P.total_react_visits += visits;

    fprintf(P.steps, "%lld,%.6f,%d,%d,%d,%lld,%lld,%lld,%lld,%lld,%lld,%lld,%.9f,%llu,%llu,%llu",
            P.total_steps, sim_time_sec, active_total, active_links, max_segments, visits,
            down_complete, down_partial, up_new, up_merge, reversals, boundary_ops,
            visits > 0 ? (double)boundary_ops / (double)visits : 0.0,
            max_delete_burst, max_create_burst, max_combined_burst);
    for (g = 0; g < PROFILE_G_COUNT; g++)
        fprintf(P.steps, ",%lld,%lld,%.9f,%lld,%lld,%lld,%.6f",
                P.step_pre_core_segments[g], P.step_post_core_segments[g],
                visits > 0 ? P.step_core_updates[g] / (double)visits : 0.0,
                P.step_boundary_to_core[g], P.step_core_to_boundary[g],
                P.step_handoffs[g], P.step_core_updates[g]);
    fputc('\n', P.steps);
    fflush(P.steps);
    P.step_sim_time_sec = sim_time_sec;

    for (k = 1; k <= P.nlinks; k++)
    {
        P.step_visits[k] = 0;
        P.step_down_complete[k] = 0;
        P.step_down_partial[k] = 0;
        P.step_up_new[k] = 0;
        P.step_up_merge[k] = 0;
        P.step_reversals[k] = 0;
    }
}

void MSXsegProfile_close(void)
{
    int k, g, recommended = 0;
    if (!P.opened) return;
    if (P.links)
    {
        for (k = 1; k <= P.nlinks; k++)
            fprintf(P.links, "%d,%lld,%d,%lld,%lld,%lld,%lld,%lld,%lld\n", k,
                    P.link[k].react_visits, P.link[k].max_segments,
                    P.link[k].down_complete, P.link[k].down_partial,
                    P.link[k].up_new, P.link[k].up_merge, P.link[k].reversals,
                    P.link[k].boundary_ops);
        fclose(P.links);
    }
    {
        FILE *summary = fopen("msx_segment_profile_summary.csv", "wt");
        if (summary)
        {
            fprintf(summary, "metric,value\n");
            fprintf(summary, "enabled,1\n");
            fprintf(summary, "quality_steps,%lld\n", P.total_steps);
            fprintf(summary, "total_react_segment_visits,%lld\n", P.total_react_visits);
            fprintf(summary, "downstream_delete_burst_p99,%llu\n",
                    histogram_p99(&P.downstream_delete_burst));
            fprintf(summary, "upstream_create_burst_p99,%llu\n",
                    histogram_p99(&P.upstream_create_burst));
            fprintf(summary, "combined_boundary_burst_p99,%llu\n",
                    histogram_p99(&P.combined_burst));
            fprintf(summary, "downstream_delete_burst_max,%llu\n",
                    P.downstream_delete_burst.maximum);
            fprintf(summary, "upstream_create_burst_max,%llu\n",
                    P.upstream_create_burst.maximum);
            fprintf(summary, "combined_boundary_burst_max,%llu\n",
                    P.combined_burst.maximum);
            fprintf(summary, "boundary_burst_samples,%lld\n", P.combined_burst.samples);
            fprintf(summary, "exact_shadow_identity,ptr_plus_generation\n");
            fprintf(summary, "exact_shadow_scope,per_link_logical_order\n");
            fprintf(summary, "shadow_memory_ok,%d\n", P.shadow_memory_ok);
            for (g = 0; g < PROFILE_G_COUNT; g++)
            {
                double cov = P.total_react_visits > 0
                    ? P.core_updates[g] / (double)P.total_react_visits : 0.0;
                double ratio = P.core_updates[g] > 0.0
                    ? (double)P.handoffs[g] / P.core_updates[g] : 0.0;
                double mean_cov = P.coverage_steps[g] > 0
                    ? P.coverage_sum[g] / (double)P.coverage_steps[g] : 0.0;
                fprintf(summary, "g%d_weighted_core_coverage,%.9f\n", ProfileG[g], cov);
                fprintf(summary, "g%d_mean_step_core_coverage,%.9f\n", ProfileG[g], mean_cov);
                fprintf(summary, "g%d_pre_core_segments_total,%lld\n", ProfileG[g],
                        P.pre_core_segments[g]);
                fprintf(summary, "g%d_core_segments_total,%lld\n", ProfileG[g],
                        P.core_segments[g]);
                fprintf(summary, "g%d_handoff_count,%lld\n", ProfileG[g], P.handoffs[g]);
                fprintf(summary, "g%d_boundary_to_core,%lld\n", ProfileG[g],
                        P.boundary_to_core[g]);
                fprintf(summary, "g%d_core_to_boundary,%lld\n", ProfileG[g],
                        P.core_to_boundary[g]);
                fprintf(summary, "g%d_core_segment_updates,%.6f\n", ProfileG[g],
                        P.core_updates[g]);
                fprintf(summary, "g%d_handoff_ratio,%.9f\n", ProfileG[g],
                        ratio);
                if (!recommended && cov >= 0.80 && ratio <= 0.05 &&
                    histogram_p99(&P.downstream_delete_burst) <=
                        (unsigned long long)ProfileG[g] &&
                    histogram_p99(&P.upstream_create_burst) <=
                        (unsigned long long)ProfileG[g] &&
                    histogram_p99(&P.combined_burst) <=
                        2ull * (unsigned long long)ProfileG[g])
                    recommended = ProfileG[g];
            }
            fprintf(summary, "recommended_g,%d\n", recommended);
            fclose(summary);
        }
    }
    if (P.steps) fclose(P.steps);
    free_arrays();
}
