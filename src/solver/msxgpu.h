#ifndef MSXGPU_H
#define MSXGPU_H

#include "msxtypes.h"

double MSXgpu_wallTimeMs(void);
int MSXgpu_openTiming(void);
void MSXgpu_closeTiming(void);
int MSXcpu_openTiming(void);
void MSXcpu_closeTiming(void);
int MSXgpu_validateStrict(void);
void MSXgpu_beginStep(double simTimeSec);
void MSXgpu_endStep(int errorCode);
void MSXgpu_reactBegin(void);
void MSXgpu_reactEnd(void);
int MSXgpu_reactPipeSegments(double dt);
void MSXgpu_addOdeTime(double ms);
void MSXgpu_addEquilTime(double ms);
void MSXgpu_addFormulaTime(double ms);

#endif
