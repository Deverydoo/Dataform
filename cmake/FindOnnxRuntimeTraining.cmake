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

    # Read ORT_API_VERSION out of the header. Only one onnxruntime.dll can be
    # loaded per process, so this runtime has to serve Muninn as well — and the
    # C API is only backward compatible, meaning an older runtime cannot satisfy
    # a consumer compiled against newer headers. Surfacing the number here turns
    # a silent runtime null-deref into a configure-time message.
    file(STRINGS "${OnnxRuntimeTraining_INCLUDE_DIR}/onnxruntime_c_api.h" _ort_api_line
         REGEX "^#define ORT_API_VERSION[ \t]+[0-9]+")
    if(_ort_api_line MATCHES "([0-9]+)")
        set(OnnxRuntimeTraining_API_VERSION "${CMAKE_MATCH_1}")
    else()
        set(OnnxRuntimeTraining_API_VERSION "0")
    endif()
    unset(_ort_api_line)

    # Provider shim: present in some build configurations, absent in others
    # (a plain CPU build emits no shared provider). Reported so the consumer can
    # copy it only when it exists.
    if(EXISTS "${ORT_ROOT}/bin/onnxruntime_providers_shared.dll")
        set(OnnxRuntimeTraining_HAS_PROVIDERS_SHARED TRUE)
    else()
        set(OnnxRuntimeTraining_HAS_PROVIDERS_SHARED FALSE)
    endif()

    message(STATUS "OnnxRuntimeTraining: ORT_API_VERSION=${OnnxRuntimeTraining_API_VERSION}, "
                   "providers_shared=${OnnxRuntimeTraining_HAS_PROVIDERS_SHARED}")

    if(NOT TARGET OnnxRuntimeTraining::OnnxRuntimeTraining)
        add_library(OnnxRuntimeTraining::OnnxRuntimeTraining SHARED IMPORTED)
        set_target_properties(OnnxRuntimeTraining::OnnxRuntimeTraining PROPERTIES
            IMPORTED_IMPLIB "${OnnxRuntimeTraining_LIBRARY}"
            IMPORTED_LOCATION "${OnnxRuntimeTraining_DLL}"
            INTERFACE_INCLUDE_DIRECTORIES "${OnnxRuntimeTraining_INCLUDE_DIR}"
        )
    endif()
endif()
