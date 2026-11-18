# Reproducible standalone build of the historical Flatboi SLIM binary.
#
# The shipped artifact came from ScrollPrize/villa revision
# fc25b4d1f160b494eee5e5f69726df2c1008bf26 and its own libs/flatboi/CMakeLists.txt.
# This project rebuilds the same target without the Qt application tree so it can be
# instrumented and compared against the shipped binary. Pass the matching villa
# checkout as -DVILLA_SOURCE=<dir>; see README.md for the exact recipe.
cmake_minimum_required(VERSION 3.22)
project(flatboi_recovery LANGUAGES C CXX)

include(FetchContent)
find_package(OpenMP REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

set(VILLA_SOURCE "" CACHE PATH "ScrollPrize/villa checkout at the recorded revision")
if(NOT VILLA_SOURCE)
  message(FATAL_ERROR "set -DVILLA_SOURCE=<villa checkout at fc25b4d1...>")
endif()
set(VILLA_LIBS "${VILLA_SOURCE}/volume-cartographer/libs")
if(NOT EXISTS "${VILLA_LIBS}/flatboi/flatboi.cpp")
  message(FATAL_ERROR "FLATBOI_SOURCE does not look like the recorded revision")
endif()

# libigl: same pin and the same overlay of volume-cartographer's local changes.
set(FLATBOI_LIBIGL_COMMIT "ae8f959ea26d7059abad4c698aba8d6b7c3205e8"
    CACHE STRING "libigl commit to fetch (matches cmake/FetchLibigl.cmake)")
FetchContent_Declare(libigl
    GIT_REPOSITORY https://github.com/libigl/libigl.git
    GIT_TAG        ${FLATBOI_LIBIGL_COMMIT}
    PATCH_COMMAND  ${CMAKE_COMMAND} -E copy_directory
                   ${VILLA_LIBS}/libigl_changes <SOURCE_DIR>
)
FetchContent_MakeAvailable(libigl)

# PaStiX 6.4.0 exactly as libs/flatboi/CMakeLists.txt configures it.
set(FLATBOI_PASTIX_TAG "v6.4.0" CACHE STRING "PaStiX tag to fetch")
FetchContent_Declare(pastix
    GIT_REPOSITORY https://gitlab.inria.fr/solverstack/pastix.git
    GIT_TAG        ${FLATBOI_PASTIX_TAG}
    GIT_SUBMODULES_RECURSE TRUE
)
set(PASTIX_INT64           OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_PARSEC     OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_STARPU     OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_MPI        OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_CUDA       OFF CACHE BOOL "" FORCE)
set(PASTIX_WITH_FORTRAN    OFF CACHE BOOL "" FORCE)
set(BUILD_TESTING          OFF CACHE BOOL "" FORCE)
set(_flatboi_saved_bsl "${BUILD_SHARED_LIBS}")
set(BUILD_SHARED_LIBS OFF)

FetchContent_GetProperties(pastix)
if(NOT pastix_POPULATED)
    FetchContent_Populate(pastix)
    set(RP_${CMAKE_PROJECT_NAME}_DICTIONNARY
        "${pastix_SOURCE_DIR}/cmake_modules/local_subs.py"
        CACHE INTERNAL "")
    set(RP_${CMAKE_PROJECT_NAME}_PRECISIONS "d" CACHE INTERNAL "")
    set(PASTIX_PRECISIONS "d" CACHE STRING "" FORCE)
    set(SPM_PRECISIONS    "d" CACHE STRING "" FORCE)
    file(READ "${pastix_SOURCE_DIR}/CMakeLists.txt" _pastix_top_cml)
    string(REPLACE "\${CMAKE_SOURCE_DIR}" "\${CMAKE_CURRENT_SOURCE_DIR}"
        _pastix_top_cml "${_pastix_top_cml}")
    foreach(_pat
        "add_subdirectory(test)"
        "add_subdirectory(example)"
        "add_subdirectory(wrappers)"
        "enable_testing()")
        string(REPLACE "${_pat}" "# stripped: ${_pat}" _pastix_top_cml "${_pastix_top_cml}")
    endforeach()
    file(WRITE "${pastix_SOURCE_DIR}/CMakeLists.txt" "${_pastix_top_cml}")
    file(READ "${pastix_SOURCE_DIR}/spm/CMakeLists.txt" _pastix_spm_cml)
    foreach(_pat
        "add_subdirectory(tests)"
        "add_subdirectory(examples)"
        "add_subdirectory(wrappers)")
        string(REPLACE "${_pat}" "# stripped: ${_pat}" _pastix_spm_cml "${_pastix_spm_cml}")
    endforeach()
    file(WRITE "${pastix_SOURCE_DIR}/spm/CMakeLists.txt" "${_pastix_spm_cml}")
    add_subdirectory(${pastix_SOURCE_DIR} ${pastix_BINARY_DIR})
endif()

set(BUILD_SHARED_LIBS "${_flatboi_saved_bsl}")
unset(_flatboi_saved_bsl)

add_executable(flatboi "${VILLA_LIBS}/flatboi/flatboi.cpp")
set_target_properties(flatboi PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
target_compile_definitions(flatboi PRIVATE SLIM_CACHED PASTIX6)
target_link_libraries(flatboi PRIVATE igl::core pastix spm OpenMP::OpenMP_CXX opencv_core opencv_imgproc opencv_imgcodecs)

# Native regressions for the corrected PaStiX shim (return codes, pattern identity,
# original-system residual). Run with: build/native_regression
add_executable(native_regression "${CMAKE_CURRENT_SOURCE_DIR}/native_regression.cpp")
set_target_properties(native_regression PROPERTIES AUTOMOC OFF AUTOUIC OFF AUTORCC OFF)
target_compile_definitions(native_regression PRIVATE PASTIX6)
target_link_libraries(native_regression PRIVATE igl::core pastix spm OpenMP::OpenMP_CXX)
