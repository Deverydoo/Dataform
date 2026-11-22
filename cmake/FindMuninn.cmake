# FindMuninn.cmake
# Finds the Muninn faster-whisper library (built out of the Muninn-Faster-Whisper tree).
#
# Muninn is developed alongside Loki Studio and is not vendored into this repo, so the
# search covers three layouts, in order of precedence:
#   1. MUNINN_ROOT, set by the user at configure time.
#   2. third_party/muninn/, for a vendored or symlinked copy.
#   3. ../Muninn-Faster-Whisper/, the sibling checkout used on the development machine.
#
# Sets:
#   Muninn_FOUND
#   Muninn_INCLUDE_DIRS
#   Muninn_LIBRARIES
#   Muninn_DLL_PATH         directory holding muninn.dll and its runtime dependencies
#   Muninn::Muninn          imported target

set(_muninn_hints
    "${MUNINN_ROOT}"
    "$ENV{MUNINN_ROOT}"
    "${CMAKE_SOURCE_DIR}/third_party/muninn"
    "${CMAKE_SOURCE_DIR}/../Muninn-Faster-Whisper"
)

find_path(Muninn_INCLUDE_DIR
    NAMES muninn/streaming_transcriber.h
    HINTS ${_muninn_hints}
    PATH_SUFFIXES include
)

# The CUDA 13 tree is preferred when present: the streaming transcriber is only
# worth running on the GPU, and that is the build kept current on this machine.
find_library(Muninn_LIBRARY
    NAMES muninn
    HINTS ${_muninn_hints}
    PATH_SUFFIXES
        build_cuda13/Release/Release
        build_cuda13/Release
        build/Release/Release
        build/Release
        lib
)

find_file(Muninn_DLL
    NAMES muninn.dll
    HINTS ${_muninn_hints}
    PATH_SUFFIXES
        build_cuda13/Release/Release
        build_cuda13/Release
        build/Release/Release
        build/Release
        bin
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Muninn
    REQUIRED_VARS Muninn_INCLUDE_DIR Muninn_LIBRARY Muninn_DLL
)

if(Muninn_FOUND)
    set(Muninn_INCLUDE_DIRS "${Muninn_INCLUDE_DIR}")
    set(Muninn_LIBRARIES "${Muninn_LIBRARY}")
    get_filename_component(Muninn_DLL_PATH "${Muninn_DLL}" DIRECTORY)

    if(NOT TARGET Muninn::Muninn)
        add_library(Muninn::Muninn SHARED IMPORTED)
        set_target_properties(Muninn::Muninn PROPERTIES
            IMPORTED_IMPLIB "${Muninn_LIBRARY}"
            IMPORTED_LOCATION "${Muninn_DLL}"
            INTERFACE_INCLUDE_DIRECTORIES "${Muninn_INCLUDE_DIR}"
        )
    endif()
endif()

unset(_muninn_hints)
