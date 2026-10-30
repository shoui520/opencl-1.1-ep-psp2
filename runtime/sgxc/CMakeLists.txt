# pvrsc: PVR_PSP2's USSE compiler (USC), shader patcher (USP) and assembler
# tables (useasm), built as a static library for the Vita.  The OpenCL
# runtime runs them on the device to turn UniFlex programs into SGX543 code.

set(PVR_PSP2_DIR "${CMAKE_CURRENT_LIST_DIR}/../../third_party/PVR_PSP2" CACHE PATH "PVR_PSP2 checkout")
set(P ${PVR_PSP2_DIR})

set(PVRSC_USC_SOURCES
  asm.c cdg.c cfa.c data.c dce.c debug.c dgraph.c domcalc.c dualissue.c efo.c
  execpred.c f16opt.c finalise.c groupinst.c hw.c icvt_c10.c icvt_core.c
  icvt_f16.c icvt_f16_vec.c icvt_f32.c icvt_f32_vec.c icvt_i32.c icvt_mem.c
  indexreg.c inst_usc.c intcvt.c iregalloc.c iselect.c layout.c pconvert.c
  precovr.c pregalloc.c regalloc.c reggroup.c regpack.c reorder.c ssa.c usc.c
  usc_utils.c usedef.c uspbin.c vec34.c)
list(TRANSFORM PVRSC_USC_SOURCES PREPEND ${P}/tools/intern/usc2/)

set(PVRSC_USEASM_SOURCES specialregs.c specialregs_vec.c useasm.c usedisasm.c useopt.c usetab.c utils.c)
list(TRANSFORM PVRSC_USEASM_SOURCES PREPEND ${P}/tools/intern/useasm/)

set(PVRSC_USP_SOURCES usp_finalise.c hwinst.c usp.c uspshader.c usp_inputdata.c
  usp_instblock.c usp_resultref.c usp_sample.c usp_texwrite.c)
list(TRANSFORM PVRSC_USP_SOURCES PREPEND ${P}/tools/intern/usp/)

add_library(pvrsc STATIC ${PVRSC_USC_SOURCES} ${PVRSC_USEASM_SOURCES} ${PVRSC_USP_SOURCES})

target_include_directories(pvrsc PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/compat
  ${P}/intermediates/sgxsupport
  ${P}/intermediates/errata
  ${P}/include/gpu_es4
  ${P}/include/gpu_es4/eurasia/include4
  ${P}/include/gpu_es4/eurasia/hwdefs
  ${P}/include/gpu_es4/eurasia/services4/include
  ${P}/tools/intern/usc2
  ${P}/tools/intern/usp
  ${P}/tools/intern/useasm
  ${P}/eurasiacon/include
  ${P}/common/tls
)

target_compile_definitions(pvrsc PUBLIC
  NDEBUG GEN_HW_CODE INCLUDE_SGX_FEATURE_TABLE INCLUDE_SGX_BUG_TABLE
  OUTPUT_USPBIN USER SUPPORT_SGX SUPPORT_SGX543 __psp2__)

target_compile_options(pvrsc PRIVATE -O2 -w -fno-strict-aliasing -include kernel.h)
