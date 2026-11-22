cmake_minimum_required(VERSION 3.16)
project(Dataform VERSION 0.6.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 REQUIRED COMPONENTS Core Quick Qml Network Sql QuickDialogs2 Widgets Charts)

qt_standard_project_setup()

# --- Phase 2: ORT Training (optional) ---
option(DATAFORM_ENABLE_TRAINING "Enable ONNX Runtime Training support" ON)

# --- llama.cpp embedded inference (optional) ---
option(DATAFORM_ENABLE_LLAMACPP "Enable embedded llama.cpp inference for background tasks" OFF)

# --- Voice loop: Muninn speech recognition + local speech synthesis (optional) ---
#
# Off unless Muninn is actually present. Muninn lives in its own repository
# alongside Loki Studio, so a clone of this project on a fresh machine has to
# still configure and build.
option(DATAFORM_ENABLE_VOICE "Enable the spoken assistant loop (requires Muninn)" ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(DATAFORM_ENABLE_TRAINING)
    find_package(OnnxRuntimeTraining REQUIRED)
endif()

if(DATAFORM_ENABLE_VOICE)
    # Not REQUIRED: a missing Muninn downgrades to a text-only build with a clear
    # reason rather than failing configuration outright.
    find_package(Muninn QUIET)
    if(Muninn_FOUND)
        find_package(Qt6 REQUIRED COMPONENTS Multimedia)
        message(STATUS "DATAFORM: voice loop ON (muninn=${Muninn_LIBRARIES})")
    else()
        set(DATAFORM_ENABLE_VOICE OFF)
        message(WARNING
            "DATAFORM: Muninn was not found — the voice loop is disabled and the app "
            "will be text-only. Point -DMUNINN_ROOT=<path to Muninn-Faster-Whisper> at "
            "a built checkout, or configure with -DDATAFORM_ENABLE_VOICE=OFF to silence this.")
    endif()
endif()

if(DATAFORM_ENABLE_LLAMACPP)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
    add_subdirectory(third_party/llama.cpp llama)
endif()

# --- Core sources (Phase 0 + Phase 1) ---
set(DATAFORM_CORE_SOURCES
    src/main.cpp
    src/llmprovider.h
    src/llmprovider.cpp
    src/settingsmanager.h
    src/settingsmanager.cpp
    src/orchestrator.h
    src/orchestrator.cpp
    src/memorystore.h
    src/memorystore.cpp
    src/cryptoutil.h
    src/cryptoutil.cpp
    src/idlescheduler.h
    src/idlescheduler.cpp
    src/profilemanager.h
    src/profilemanager.cpp
    src/whyengine.h
    src/whyengine.cpp
    src/traitextractor.h
    src/traitextractor.cpp
    src/evalsuite.h
    src/evalsuite.cpp
    src/schemamigrator.h
    src/schemamigrator.cpp
    src/modelgeneration.h
    src/modelgeneration.cpp
    src/profilehealthmanager.h
    src/profilehealthmanager.cpp
    src/datalifecyclemanager.h
    src/datalifecyclemanager.cpp
    src/websearchengine.h
    src/websearchengine.cpp
    src/researchstore.h
    src/researchstore.cpp
    src/researchengine.h
    src/researchengine.cpp
    src/thoughtengine.h
    src/thoughtengine.cpp
    src/newsengine.h
    src/newsengine.cpp
    src/reminderengine.h
    src/reminderengine.cpp
    src/goaltracker.h
    src/goaltracker.cpp
    src/sentimenttracker.h
    src/sentimenttracker.cpp
    src/learningengine.h
    src/learningengine.cpp
    src/clipboardhelper.h
    src/clipboardhelper.cpp
    src/embeddingmanager.h
    src/embeddingmanager.cpp
    src/distillationmanager.h
    src/distillationmanager.cpp
    src/backgroundjobmanager.h
    src/backgroundjobmanager.cpp
    src/llmresponseparser.h
    src/llmresponseparser.cpp
    src/toolregistry.h
    src/toolregistry.cpp
    src/modelcatalog.h
    src/modelcatalog.cpp
    src/idlejobcoordinator.h
    src/idlejobcoordinator.cpp
    src/threadsafeprogress.h
    src/sessionguard.h
)

# --- Phase 2: Training sources (conditional) ---
set(DATAFORM_TRAINING_SOURCES
    src/tokenizer.h
    src/tokenizer.cpp
    src/trainingdatagenerator.h
    src/trainingdatagenerator.cpp
    src/orttrainingmanager.h
    src/orttrainingmanager.cpp
    src/adaptermanager.h
    src/adaptermanager.cpp
    src/reflectionengine.h
    src/reflectionengine.cpp
    src/populationstate.h
    src/populationstate.cpp
    src/weightmerger.h
    src/weightmerger.cpp
    src/evolutionengine.h
    src/evolutionengine.cpp
    src/lineagetracker.h
    src/lineagetracker.cpp
    src/ortinferencemanager.h
    src/ortinferencemanager.cpp
)

# --- llama.cpp sources (conditional) ---
set(DATAFORM_LLAMACPP_SOURCES
    src/llamacppmanager.h
    src/llamacppmanager.cpp
)

# --- Voice sources (conditional) ---
# Kept out of the target entirely when voice is off: these headers declare
# Q_OBJECT inside DATAFORM_VOICE_ENABLED, which AUTOMOC would otherwise moc into
# references to classes the preprocessor removed.
set(DATAFORM_VOICE_SOURCES
    src/voice/audiocapture.h
    src/voice/audiocapture.cpp
    src/voice/audioplayback.h
    src/voice/audioplayback.cpp
    src/voice/speechrecognizer.h
    src/voice/speechrecognizer.cpp
    src/voice/speechsynthesizer.h
    src/voice/speechsynthesizer.cpp
    src/voice/ttsbackends.h
    src/voice/ttsbackends.cpp
    src/voice/voicesession.h
    src/voice/voicesession.cpp
)
if(WIN32)
    list(APPEND DATAFORM_VOICE_SOURCES
        src/voice/sapiworker.h
        src/voice/sapiworker.cpp
    )
endif()

qt_add_executable(Dataform
    ${DATAFORM_CORE_SOURCES}
    $<$<BOOL:${DATAFORM_ENABLE_TRAINING}>:${DATAFORM_TRAINING_SOURCES}>
    $<$<BOOL:${DATAFORM_ENABLE_LLAMACPP}>:${DATAFORM_LLAMACPP_SOURCES}>
    $<$<BOOL:${DATAFORM_ENABLE_VOICE}>:${DATAFORM_VOICE_SOURCES}>
    dataform.rc
)

qt_add_qml_module(Dataform
    URI Dataform
    VERSION 1.0
    QML_FILES
        qml/Main.qml
        qml/ChatWindow.qml
        qml/SettingsPanel.qml
        qml/IdlePanel.qml
        qml/MemoryPanel.qml
        qml/LineagePanel.qml
        qml/ConversationSidebar.qml
        qml/ModelCatalogPanel.qml
        qml/VoiceBar.qml
        qml/VoiceSettingsPanel.qml
)

target_link_libraries(Dataform PRIVATE
    Qt6::Core
    Qt6::Quick
    Qt6::Qml
    Qt6::Network
    Qt6::Sql
    Qt6::QuickDialogs2
    Qt6::Widgets
    Qt6::Charts
)

# Compiler warnings
if(MSVC)
    target_compile_options(Dataform PRIVATE /W3)
endif()

if(WIN32)
    target_link_libraries(Dataform PRIVATE bcrypt)
endif()

# --- Phase 2: ORT Training linking ---
if(DATAFORM_ENABLE_TRAINING)
    target_link_libraries(Dataform PRIVATE OnnxRuntimeTraining::OnnxRuntimeTraining)
    target_compile_definitions(Dataform PRIVATE DATAFORM_TRAINING_ENABLED)

    # Copy ORT DLLs to output directory
    add_custom_command(TARGET Dataform POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OnnxRuntimeTraining_DLL_PATH}/onnxruntime.dll"
            "$<TARGET_FILE_DIR:Dataform>/onnxruntime.dll"
        COMMENT "Copying ONNX Runtime DLL to output directory"
    )

    # Only some build configurations emit the shared-provider shim; a plain CPU
    # build does not, and copying a file that is not there fails the build.
    if(OnnxRuntimeTraining_HAS_PROVIDERS_SHARED)
        add_custom_command(TARGET Dataform POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${OnnxRuntimeTraining_DLL_PATH}/onnxruntime_providers_shared.dll"
                "$<TARGET_FILE_DIR:Dataform>/onnxruntime_providers_shared.dll"
            COMMENT "Copying ONNX Runtime provider shim to output directory"
        )
    endif()
endif()

# --- llama.cpp linking ---
if(DATAFORM_ENABLE_LLAMACPP)
    target_link_libraries(Dataform PRIVATE llama)
    target_compile_definitions(Dataform PRIVATE DATAFORM_LLAMACPP_ENABLED)
    target_include_directories(Dataform PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    )
endif()

# --- Voice loop linking ---
if(DATAFORM_ENABLE_VOICE)
    target_link_libraries(Dataform PRIVATE Qt6::Multimedia Muninn::Muninn)
    target_compile_definitions(Dataform PRIVATE DATAFORM_VOICE_ENABLED)

    if(WIN32)
        # sapi.h lives in the Windows SDK; sapi.lib carries the CLSIDs/IIDs.
        target_link_libraries(Dataform PRIVATE sapi ole32)
    endif()

    # Muninn brings its own CTranslate2 and ONNX Runtime. Copying what sits beside
    # muninn.dll keeps the runtime set consistent with the build it came from,
    # rather than leaving the loader to find a mismatched copy on PATH.
    set(_muninn_runtime_dlls "")
    set(_muninn_dll_names muninn.dll ctranslate2.dll)

    # onnxruntime.dll is deliberately excluded from that list when training is on.
    # The ORT training package is a superset of the plain runtime Muninn was built
    # against, so one copy can serve both — but only if it is the training one.
    # Muninn's plain build would satisfy Silero VAD and then fail every training
    # symbol the reflection engine needs.
    if(NOT DATAFORM_ENABLE_TRAINING)
        list(APPEND _muninn_dll_names onnxruntime.dll)
    endif()

    foreach(_dll ${_muninn_dll_names})
        if(EXISTS "${Muninn_DLL_PATH}/${_dll}")
            list(APPEND _muninn_runtime_dlls "${Muninn_DLL_PATH}/${_dll}")
        endif()
    endforeach()

    if(_muninn_runtime_dlls)
        add_custom_command(TARGET Dataform POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_muninn_runtime_dlls} "$<TARGET_FILE_DIR:Dataform>"
            COMMENT "Copying Muninn runtime DLLs to output directory"
        )
    endif()

    # One onnxruntime.dll serves the whole process, so it has to be new enough for
    # Muninn. Muninn is compiled against ORT 1.23, whose ORT_API_VERSION is 23;
    # the C API is backward but not forward compatible, so an older runtime makes
    # Muninn's GetApi call return null and its neural VAD fail at the first
    # utterance. The stock training package from NuGet is 1.19 (API 19) and trips
    # this — see Docs/decisions/0002-onnxruntime-unification.md.
    set(DATAFORM_MUNINN_REQUIRED_ORT_API 23)

    if(DATAFORM_ENABLE_TRAINING)
        if(OnnxRuntimeTraining_API_VERSION LESS DATAFORM_MUNINN_REQUIRED_ORT_API)
            message(WARNING
                "DATAFORM: the ONNX Runtime in third_party/onnxruntime-training reports "
                "ORT_API_VERSION=${OnnxRuntimeTraining_API_VERSION}, but Muninn needs at least "
                "${DATAFORM_MUNINN_REQUIRED_ORT_API}. Neural VAD will fail at runtime. Build ORT from "
                "source with --enable_training_apis as described in "
                "Docs/decisions/0002-onnxruntime-unification.md, or accept energy VAD.")
            set(DATAFORM_MUNINN_NEURAL_VAD OFF)
        else()
            message(STATUS
                "DATAFORM: one ONNX Runtime (API ${OnnxRuntimeTraining_API_VERSION}) serves both the "
                "training tier and Muninn — neural VAD available.")
            set(DATAFORM_MUNINN_NEURAL_VAD ON)
        endif()
    else()
        # Without the training tier, Muninn's own runtime is copied and matches
        # what it was built against by construction.
        set(DATAFORM_MUNINN_NEURAL_VAD ON)
    endif()

    if(DATAFORM_MUNINN_NEURAL_VAD)
        target_compile_definitions(Dataform PRIVATE DATAFORM_MUNINN_NEURAL_VAD)
    endif()
endif()

set_target_properties(Dataform PROPERTIES
    WIN32_EXECUTABLE TRUE
)
