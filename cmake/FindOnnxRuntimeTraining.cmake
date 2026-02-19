# FindOnnxRuntimeTraining.cmake
# Finds ONNX Runtime Training library (from NuGet package)
#
# Sets:
#   OnnxRuntimeTraining_FOUND
#   OnnxRuntimeTraining_INCLUDE_DIRS
#   OnnxRuntimeTraining_LIBRARIES
#   OnnxRuntimeTraining_DLL_PATH

set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime-training")

find_path(OnnxRuntimeTraining_INCLUDE_DIR
    NAMES onnxruntime_training_cxx_api.h
    PATHS "${ORT_ROOT}/include"
    NO_DEFAULT_PATH
)

find_library(OnnxRuntimeTraining_LIBRARY
    NAMES onnxruntime
    PATHS "${ORT_ROOT}/lib"
    NO_DEFAULT_PATH
)

find_file(OnnxRuntimeTraining_DLL
    NAMES onnxruntime.dll
    PATHS "${ORT_ROOT}/bin"
    NO_DEFAULT_PATH
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OnnxRuntimeTraining
    REQUIRED_VARS
        OnnxRuntimeTraining_INCLUDE_DIR
        OnnxRuntimeTraining_LIBRARY
        OnnxRuntimeTraining_DLL
)

if(OnnxRuntimeTraining_FOUND)
    set(OnnxRuntimeTraining_INCLUDE_DIRS "${OnnxRuntimeTraining_INCLUDE_DIR}")
    set(OnnxRuntimeTraining_LIBRARIES "${OnnxRuntimeTraining_LIBRARY}")
    set(OnnxRuntimeTraining_DLL_PATH "${ORT_ROOT}/bin")

    if(NOT TARGET OnnxRuntimeTraining::OnnxRuntimeTraining)
        add_library(OnnxRuntimeTraining::OnnxRuntimeTraining SHARED IMPORTED)
        set_target_properties(OnnxRuntimeTraining::OnnxRuntimeTraining PROPERTIES
            IMPORTED_IMPLIB "${OnnxRuntimeTraining_LIBRARY}"
            IMPORTED_LOCATION "${OnnxRuntimeTraining_DLL}"
            INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntimeTraining_INCLUDE_DIR}"
        )
    endif()
endif()
