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

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

if(DATAFORM_ENABLE_TRAINING)
    find_package(OnnxRuntimeTraining REQUIRED)
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

qt_add_executable(Dataform
    ${DATAFORM_CORE_SOURCES}
    $<$<BOOL:${DATAFORM_ENABLE_TRAINING}>:${DATAFORM_TRAINING_SOURCES}>
    $<$<BOOL:${DATAFORM_ENABLE_LLAMACPP}>:${DATAFORM_LLAMACPP_SOURCES}>
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
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${OnnxRuntimeTraining_DLL_PATH}/onnxruntime_providers_shared.dll"
            "$<TARGET_FILE_DIR:Dataform>/onnxruntime_providers_shared.dll"
        COMMENT "Copying ONNX Runtime DLLs to output directory"
    )
endif()

# --- llama.cpp linking ---
if(DATAFORM_ENABLE_LLAMACPP)
    target_link_libraries(Dataform PRIVATE llama)
    target_compile_definitions(Dataform PRIVATE DATAFORM_LLAMACPP_ENABLED)
    target_include_directories(Dataform PRIVATE
        ${CMAKE_SOURCE_DIR}/third_party/llama.cpp/include
    )
endif()

set_target_properties(Dataform PROPERTIES
    WIN32_EXECUTABLE TRUE
)
