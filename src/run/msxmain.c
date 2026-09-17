/*******************************************************************************
**  MODULE:        MSXMAIN.C
**  PROJECT:       EPANET-MSX
**  DESCRIPTION:   Main module of the EPANET Multi-Species Extension toolkit.
**  AUTHORS:       see AUTHORS
**  Copyright:     see AUTHORS
**  License:       see LICENSE
**  VERSION:       2.0.00
**  LAST UPDATE:   08/30/2022
**
**  EPANET-MSX is an extension of the EPANET program for modeling the fate
**  and transport of multiple interacting chemical species within a water
**  distribution system over an extended period of operation. This module
**  provides a main function for producing a stand-alone console
**  application version of EPANET-MSX. It is not needed when compiling
**  EPANET-MSX into a dynamic link library (DLL) of callable functions.
**
**  To use either the console version or the DLL a user must prepare a
**  regular EPANET input file that describes the pipe network layout
**  and its hydraulic properties as well as a special EPANET-MSX input file
**  that names the chemical species being modeled and specifies the reaction
**  rate and equilbrium expressions that define their chemical behavior. The
**  format of these files is described in the EPANET and EPANET-MSX Users
**  Manuals, respectively.
*******************************************************************************/

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <time.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOGDI
#include <windows.h>
#endif
#include "epanet2.h"                   // EPANET toolkit header file
#include "epanetmsx.h"                 // EPANET-MSX toolkit header file
#include <math.h>

static double wall_time_ms(void)
{
#ifdef _WIN32
    LARGE_INTEGER counter, frequency;
    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);
    return 1000.0 * (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1000.0 * (double)ts.tv_sec + (double)ts.tv_nsec / 1000000.0;
#endif
}

static void strip_eol(char *s)
{
    size_t n = strlen(s);
    while (n > 0 && (s[n-1] == '\n' || s[n-1] == '\r')) s[--n] = '\0';
}

static void finalize_timing_csv(
    const char *filename, double startup_ms, double epanet_open_ms,
    double msx_open_ms, double hydraulics_ms, double initialization_ms,
    double quality_step_ms, double report_ms, double binary_write_ms,
    double cleanup_ms, double process_total_ms)
{
    char header[8192];
    char total[8192];
    FILE *f = fopen(filename, "rt");
    if (!f) return;
    if (!fgets(header, sizeof(header), f) || !fgets(total, sizeof(total), f))
    {
        fclose(f);
        return;
    }
    fclose(f);
    strip_eol(header);
    strip_eol(total);
    f = fopen(filename, "wt");
    if (!f) return;
    fprintf(f, "%s,startup_ms,epanet_open_ms,msx_open_ms,hydraulics_ms,initialization_ms,quality_step_ms,report_ms,binary_write_ms,cleanup_ms,process_total_ms\n", header);
    fprintf(f, "%s,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f\n",
        total, startup_ms, epanet_open_ms, msx_open_ms, hydraulics_ms,
        initialization_ms, quality_step_ms, report_ms, binary_write_ms,
        cleanup_ms, process_total_ms);
    fprintf(f, "UNIT,count,s,flag,flag,flag,flag,flag,flag,flag,flag,flag,flag,flag,flag,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,count,count,count,rate_time,count,count,count,ms,ms,ms,ms,ms,ms,ms,count,count,count,count,rate_time,flag,code,code,code,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms\n");
    fprintf(f, "TITLE,quality_step_count,final_sim_time_seconds,gpu_runtime_enabled,gpu_strict_mode,react_on_gpu,advect_on_gpu,mix_on_gpu,release_on_gpu,disperse_on_gpu,ode_on_gpu,equil_on_gpu,formula_on_gpu,full_coupling_enabled,equil_embedded_in_ode,total_transport_step_wall_time,react_stage_wall_time,advect_stage_wall_time,mix_stage_wall_time,release_stage_wall_time,disperse_stage_wall_time,react_segment_pack_wall_time,host_to_device_copy_wall_time,ode_compute_wall_time,equil_compute_wall_time,formula_compute_wall_time,device_to_host_copy_wall_time,react_segment_unpack_wall_time,jit_compile_wall_time,nvrtc_compile_wall_time,ptx_cache_read_wall_time,cuda_module_load_wall_time,kernel_lookup_wall_time,rk5_function_eval_count,rk5_accepted_step_count,rk5_rejected_step_count,rk5_last_adaptive_hstep,rk5_bucket_count,rk5_bucket_launch_count,rk5_bucket_max_segment_count,rk5_bucket_reorder_wall_time,react_parallel_count_wall_time,react_serial_prefix_wall_time,react_segment_pack_only_wall_time,react_pinned_host_allocation_wall_time,react_device_workspace_allocation_wall_time,react_unpack_scatter_wall_time,ros2_function_eval_count,ros2_jacobian_count,ros2_accepted_step_count,ros2_rejected_step_count,ros2_last_adaptive_hstep,rk5_fast_mode_enabled,rk5_first_error_code,ros2_first_error_code,first_error_code,process_start_to_epanet_open,epanet_input_open,msx_input_open,hydraulics_solve_and_save,msx_quality_initialization,all_msxstep_calls,msx_report_generation,binary_result_write,msx_and_epanet_cleanup,process_start_to_cleanup_done\n");
    fclose(f);
#if 0
    return;
    fprintf(f, "TITLE,水质输运子步累计数,最终模拟时间_秒,GPU运行时是否启用,GPU严格模式,React是否在GPU运行,Advect是否在GPU运行,Mix是否在GPU运行,Release是否在GPU运行,Disperse是否在GPU运行,ODE是否在GPU运行,EQUIL是否在GPU运行,FORMULA是否在GPU运行,是否采用全耦合,平衡计算是否嵌入ODE,输运子步累计墙钟时间,React阶段墙钟时间,Advect阶段墙钟时间,Mix阶段墙钟时间,Release阶段墙钟时间,Disperse阶段墙钟时间,React分段打包墙钟时间,主机到设备复制墙钟时间,ODE计算墙钟时间,EQUIL计算墙钟时间,FORMULA计算墙钟时间,设备到主机复制墙钟时间,React分段解包墙钟时间,JIT编译墙钟时间,首个错误码,程序入口至EPANET打开前,EPANET输入打开,MSX输入打开,水力计算及结果保存,MSX水质初始化,全部MSXstep调用,MSX报告生成,二进制结果写盘,MSX和EPANET清理,程序入口至清理完成\n");
    fclose(f);
    return;
    fprintf(f, "UNIT,count,s,flag,flag,flag,flag,flag,flag,flag,flag,flag,flag,flag,flag,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms,code,ms,ms,ms,ms,ms,ms,ms,ms,ms,ms\n");
    fprintf(f, "TITLE,水质输运子步累计数,最终模拟时间_秒,GPU运行时是否启用,GPU严格模式,React是否在GPU运行,Advect是否在GPU运行,Mix是否在GPU运行,Release是否在GPU运行,Disperse是否在GPU运行,ODE是否在GPU运行,EQUIL是否在GPU运行,FORMULA是否在GPU运行,是否采用全耦合,平衡计算是否嵌入ODE,输运子步累计墙钟时间,React阶段墙钟时间,Advect阶段墙钟时间,Mix阶段墙钟时间,Release阶段墙钟时间,Disperse阶段墙钟时间,React分段打包墙钟时间,主机到设备复制墙钟时间,ODE计算墙钟时间,EQUIL计算墙钟时间,FORMULA计算墙钟时间,设备到主机复制墙钟时间,React分段解包墙钟时间,JIT编译墙钟时间,首个错误码,程序入口至EPANET打开前,EPANET输入打开,MSX输入打开,水力计算及结果保存,MSX水质初始化,全部MSXstep调用,MSX报告生成,二进制结果写盘,MSX和EPANET清理,程序入口至清理完成\n");
    fclose(f);
#endif
}

static void finalize_cpu_timing_csv(const char *filename, double process_total_ms)
{
    char header[2048];
    char total[2048];
    FILE *f = fopen(filename, "rt");
    if (!f) return;
    if (!fgets(header, sizeof(header), f) || !fgets(total, sizeof(total), f))
    {
        fclose(f);
        return;
    }
    fclose(f);
    strip_eol(header);
    strip_eol(total);
    f = fopen(filename, "wt");
    if (!f) return;
    fprintf(f, "%s,process_total_ms\n", header);
    fprintf(f, "%s,%.6f\n", total, process_total_ms);
    fclose(f);
}

int main(int argc, char* argv[])
/*
**  Purpose:
**    main function for the console version of EPANET-MSX.
**
**  Input:
**    argc = number of command line arguments
**    argv = array of command line arguments.
**
**  Returns:
**    an error code (or 0 for no error).
**
**  Notes:
**    The command line arguments are:
**     - the name of the regular EPANET input data file
**     - the name of the EPANET-MSX input file
**     - the name of a report file that will contain status
**       messages and output results
**     - optionally, the name of an output file that will
**       contain water quality results in binary format.
*/
{
    int    err, done = 1;
    double t, tleft;
    long   oldHour, newHour;
    char* inpFile, * repFile, * outFile;
    double totalStartMs = wall_time_ms();
    double stageStartMs;
    double startupMs = 0.0, epanetOpenMs = 0.0, msxOpenMs = 0.0;
    double hydraulicsMs = 0.0, initializationMs = 0.0, qualityStepMs = 0.0;
    double reportMs = 0.0, binaryWriteMs = 0.0, cleanupMs = 0.0;
    double processTotalMs = 0.0;

    // --- check command line arguments

    if (argc < 4 || argc > 5)
    {
        printf("\nInvalid command line arguments:\n\n");
        printf("usage: runepanetmsx <inp_file> <msx_file> <report_file> [binary_output_file]\n");
        return 0;
    }
    inpFile = argv[1];
    repFile = argv[3];
    if (argc == 5) {
        outFile = argv[4];
    }
    else {
        outFile = "";
    }
    remove("msx_gpu_timing.csv");

    // --- open EPANET file

    printf("\n... EPANET-MSX Version 2.0.0\n");                                  
    printf("\n  o Processing EPANET input file");
    startupMs = wall_time_ms() - totalStartMs;
    stageStartMs = wall_time_ms();
    /* Keep the actual copied/self-contained INP path for resident case hashing. */
    err = MSXENopen(inpFile, repFile, outFile);
    epanetOpenMs = wall_time_ms() - stageStartMs;
    printf("\nTIMING,module=ENopen,seconds=%.6f", epanetOpenMs / 1000.0);
    do
    {
        if (err)
        {
            printf("\n\n... Cannot read EPANET file; error code = %d\n", err);
            ENclose();
            return err;
        }

        // --- open the MSX input file

        printf("\n  o Processing MSX input file   ");
        stageStartMs = wall_time_ms();
        err = MSXopen(argv[2]);
        msxOpenMs = wall_time_ms() - stageStartMs;
        printf("\nTIMING,module=MSXopen,seconds=%.6f", msxOpenMs / 1000.0);
        if (err)
        {
            printf("\n\n... Cannot read EPANET-MSX file; error code = %d\n", err);
            break;
        }

        //--- solve hydraulics

        printf("\n  o Computing network hydraulics");
        stageStartMs = wall_time_ms();
        err = MSXsolveH();
        hydraulicsMs = wall_time_ms() - stageStartMs;
        printf("\nTIMING,module=MSXsolveH,seconds=%.6f", hydraulicsMs / 1000.0);
        if (err)
        {
            printf("\n\n... Cannot obtain network hydraulics; error code = %d\n", err);
            break;
        }

        //--- Initialize the multi-species analysis

        printf("\n  o Initializing network water quality");
        stageStartMs = wall_time_ms();
        err = MSXinit(1);
        initializationMs = wall_time_ms() - stageStartMs;
        printf("\nTIMING,module=MSXinit,seconds=%.6f", initializationMs / 1000.0);
        if (err)
        {
            printf("\n\n... Cannot initialize EPANET-MSX; error code = %d\n", err);
            break;
        }
        t = 0;
        oldHour = -1;
        newHour = 0;
        printf("\n");

        //--- Run the multi-species analysis at each time step

        do
        {
            if (oldHour != newHour)
            {
                printf("\r  o Computing water quality at hour %d", newHour);
                oldHour = newHour;
            }
            stageStartMs = wall_time_ms();
            err = MSXstep(&t, &tleft);
            qualityStepMs += wall_time_ms() - stageStartMs;
            newHour = (long)(t / 3600.);

        } while (!err && tleft > 0);
        printf("\nTIMING,module=MSXstep_total,seconds=%.6f", qualityStepMs / 1000.0);
        if (err)
        {
            printf("\n\n... EPANET-MSX runtime error; error code = %d\n", err);
            break;
        }
        else
            printf("\r  o Computing water quality at hour %d", (long)(t / 3600.));

        // --- report results

        printf("\n  o Reporting water quality results");
        stageStartMs = wall_time_ms();
        err = MSXreport();
        reportMs = wall_time_ms() - stageStartMs;
        printf("\nTIMING,module=MSXreport,seconds=%.6f", reportMs / 1000.0);
        if (err)
        {
            printf("\n\n... EPANET-MSX report writer error; error code = %d\n", err);
            break;
        }

        // --- save results to binary file if a file name was provided

        if (argc >= 5)
        {
            stageStartMs = wall_time_ms();
            err = MSXsaveoutfile(argv[4]);
            binaryWriteMs = wall_time_ms() - stageStartMs;
            printf("\nTIMING,module=MSXsaveoutfile,seconds=%.6f", binaryWriteMs / 1000.0);
            if (err > 0)
            {
                printf("\n\n... Cannot save EPANET-MSX results file; error code = %d\n", err);
                break;
            }
        }

    } while (!done);

    //--- Close both the multi-species & EPANET systems

    stageStartMs = wall_time_ms();
    MSXclose();
    ENclose();
    cleanupMs = wall_time_ms() - stageStartMs;
    processTotalMs = wall_time_ms() - totalStartMs;
    finalize_timing_csv("msx_gpu_timing.csv", startupMs, epanetOpenMs, msxOpenMs,
        hydraulicsMs, initializationMs, qualityStepMs, reportMs, binaryWriteMs,
        cleanupMs, processTotalMs);
    finalize_cpu_timing_csv("msx_cpu_timing.csv", processTotalMs);
    printf("\nTIMING,module=cleanup,seconds=%.6f", cleanupMs / 1000.0);
    printf("\nTIMING,module=total,seconds=%.6f", processTotalMs / 1000.0);
    if (!err) printf("\n\n... EPANET-MSX completed successfully.");
    printf("\n");
    return err;
}
